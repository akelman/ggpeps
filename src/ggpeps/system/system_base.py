from abc import ABC, abstractmethod
from typing import Optional

import sys
import logging
import itertools as it

import sympy
from scipy.linalg import block_diag

import jax
import numpy as np
from ggpeps import xnp as xnp
from ggpeps import xscipy as xscipy

import ggpeps
from ggpeps import utils
from ggpeps.lattice import Direction
from ggpeps.system.backend import backend
from ggpeps.system.config_base import Config2DBase
from ggpeps.modearray import generate_permutation_matrix

logger = logging.getLogger(ggpeps.LOGGER_NAME)


def maybe_jit(*jit_args, **jit_kwargs):
    def decorator(func):
        if ggpeps.PREFERRED_BACKEND == "jax":
            return jax.jit(func, *jit_args, **jit_kwargs)
        else:
            return func

    return decorator


################## System2DBase ######################


class System2DBase(ABC):
    """Base class for two dimensional systems.

    This class inherits from the abstract base class to enable abstract methods that
    must be overwritten in a child class.

    This class cannot be instantiated directly.
    """

    def __init__(self, cfg: Config2DBase) -> None:
        """Constructor of a Z2System2D system, for any groups, with any number of virtual or physical
        fermions per site per link. All of the ansatz information is stored in the provided configuration.

        Args:
            cfg (Config2DBase): Configuration containing all system-related parameters
        """
        self.cfg: Config2DBase = cfg
        self.initialize()

    def initialize(self) -> None:
        """Initialization function. This function also resets the system.
        If necessary, this would also be a good spot to copy data from the configuration.
        """

        # All variables that contain _vec are arrays of length nlayer in the first dimension.
        # Other types of vec are indicated by layervec, sitevec, etc.

        # Parameter based matrices
        self._tmat_layervec_unitcellvec: Optional[list[list[xnp.ndarray]]] = None
        self._tmat_layervec_sitevec: Optional[list[list[xnp.ndarray]]] = None
        self._gamma_dirac_layervec_sitevec: Optional[xnp.ndarray] = None
        self._gamma_maj_layervec_sitevec: Optional[xnp.ndarray] = None
        self._gamma_maj_sys_vec: Optional[xnp.ndarray] = None

        # Partial covariance matrices
        self._mat_a_vec: Optional[xnp.ndarray] = None
        self._mat_b_vec: Optional[xnp.ndarray] = None
        self._mat_d_vec: Optional[xnp.ndarray] = None
        self._det_mat_d_vec: Optional[xnp.ndarray] = None
        self._mat_d_inv_vec: Optional[xnp.ndarray] = None

        # Full covariance matrix (gamma_out) of the fermions
        self._ferm_covmat_vec: Optional[xnp.ndarray] = None

        # Parameter dependent quantities for the electric energy
        self.mod_link_ind = 0  # link index for which the electric energy is computed
        self._mat_a_mod_vec: Optional[xnp.ndarray] = None
        self._mat_b_mod_vec: Optional[xnp.ndarray] = None
        self._mat_d_mod_vec: Optional[xnp.ndarray] = None
        self._det_mat_d_mod_vec: Optional[xnp.ndarray] = None
        self._mat_d_mod_inv_vec: Optional[xnp.ndarray] = None
        # Electric energy intermediate values - if we compute the electric energy,
        # we store intermediate values to be reused in the gradient calculation
        self._covmat_out_virt_vec: Optional[list[xnp.ndarray]] = None
        self._norm_mod_vec: Optional[list[float]] = None
        self._lognorm_default_vec: Optional[xnp.ndarray] = None

        # Management of the gauge fields
        self._gamma_gauge_neutral_vec_dirs: Optional[xnp.ndarray] = (
            None  # vec for layers (choices of projectors may be different for each layer), dirs for directions
        )

        # In cases when different layers use the same projectors, all elements will point to the same gamma_in_sys:
        self._gamma_in_sys_vec: Optional[xnp.ndarray] = None

        neutral_gauge = self.cfg.gaugemgr.get_neutral_gauge_value()
        self._gaugefieldvec: xnp.ndarray = xnp.array([neutral_gauge] * self.cfg.lattice.nlinks)

        # Weight
        self._weight: Optional[float] = None

        # Gradients
        self._gamma_maj_sys_deriv: Optional[xnp.ndarray] = None
        self._el_energy_op_grad_vec: Optional[xnp.ndarray] = None  # first index is layer, second index is symbol
        self._mass_energy_op_grad_vec: Optional[xnp.ndarray] = None
        self._int_energy_op_grad_vec: Optional[xnp.ndarray] = None
        self._chem_energy_op_grad_vec: Optional[xnp.ndarray] = None

        # gradients of gamma_out for all symbols: _d_gamma_out_symbolvec[lay, uc_ind, symbol]
        # is the matrix of derivatives of gamma_out[lay] wrt the parameter at that (lay, uc_ind, symbol)
        self._d_gamma_out_symbolvec: Optional[xnp.ndarray] = None

        self._grad_over_norm_vec: Optional[xnp.ndarray] = None

        # Observables
        self._energy: Optional[float] = None
        self._el_energy_op: Optional[float] = None
        self._el_energy_op_vec: Optional[xnp.ndarray] = None
        self._mag_energy_op: Optional[float] = None
        self._mass_energy_op: Optional[float] = None
        self._mass_energy_op_vec: Optional[xnp.ndarray] = None
        self._int_energy_op: Optional[float] = None
        self._int_energy_op_vec: Optional[xnp.ndarray] = None
        self._chem_energy_op_vec: Optional[xnp.ndarray] = None
        self._all_occupations: Optional[xnp.ndarray] = None

        # Woodbury Update and Matrix Inversion
        self._wi_gamma_in_vec: Optional[list[utils.WoodburyInverter]] = None  # Tracks (D^-1 - gammain)^-1
        self._wi_gamma_out_vec: Optional[list[utils.WoodburyInverter]] = None  # Tracks (D - gammain)^-1
        self._incdet_vec: Optional[list[utils.IncLogAbsDeterminant]] = None  # Tracks det(D^-1 - gammain)

        self._wi_gamma_in_mod_vec: Optional[list[utils.WoodburyInverter]] = None  # Tracks (Dmod^-1 - gammain)^-1
        self._wi_gamma_out_mod_vec: Optional[list[utils.WoodburyInverter]] = None  # Tracks (Dmod - gammain)^-1
        self._incdet_mod_vec: Optional[list[utils.IncLogAbsDeterminant]] = None  # Tracks det(Dmod^-1 - gammain)

        return

    def invalidate_gauge_update(self) -> None:
        """Reset the values of computed quantitities to avoid spillover from previous computations.
        We do not need to reset quantities that are not dependent on the gauge fields,
        such as _gamma_maj_sys_vec, _mat_a_vec, etc.
        """

        self._ferm_covmat_vec = None
        self._d_gamma_out_symbolvec = None

        self._energy = None
        self._el_energy_op = None
        self._el_energy_op_vec = None
        self._mag_energy_op = None
        self._mass_energy_op = None
        self._mass_energy_op_vec = None
        self._int_energy_op = None
        self._int_energy_op_vec = None
        self._chem_energy_op_vec = None
        self._all_occupations = None

        self._el_energy_op_grad_vec = None
        self._mass_energy_op_grad_vec = None
        self._int_energy_op_grad_vec = None
        self._chem_energy_op_grad_vec = None
        self._grad_over_norm_vec = None

        self._covmat_out_virt_vec = None
        self._norm_mod_vec = None
        self._lognorm_default_vec = None
        return

    def _extract_partial_covmatvec(self, offset: int) -> tuple[xnp.ndarray, xnp.ndarray, xnp.ndarray]:
        mat_a_vec = self.gamma_maj_sys_vec[:, :offset, :offset]
        mat_b_vec = self.gamma_maj_sys_vec[:, :offset, offset:]
        mat_d_vec = self.gamma_maj_sys_vec[:, offset:, offset:]
        return mat_a_vec, mat_b_vec, mat_d_vec

    @property
    def symbolvec(self) -> list[sympy.Symbol]:
        """Return the symbolvec.

        Returns:
            list: Vector of analytic symbols
        """
        return self.cfg.symbolvec

    @property
    def tmat_symb(self) -> sympy.Matrix:
        """Return the symbolic version of the T matrix.
        This is stored in the config for each ansatz, because it is part of the definition of the ansatz,
        and is not specific to a particular state (i.e. particular parameters).
        """
        return self.cfg.tmat_symb

    def compute_tmat_deriv(self, symb: sympy.Symbol) -> xnp.ndarray:
        """Return the derivative of the T matrix with respect to the symbol

        Args:
            symb (sympy.Symbol): Symbol to be derived with respect to

        Returns:
            xnp.ndarray: Array of symbols
        """
        tmat_symb = self.cfg.tmat_symb
        # convert to numpy array, then to xnp (jax cannot convert from sympy directly)
        tmat_deriv = xnp.asarray(np.asarray(sympy.diff(tmat_symb, symb)).astype(complex))
        return tmat_deriv

    def _eval_tmat_symb(self, paramvec) -> xnp.ndarray:
        """Compute the numerical representation of the T matrix

        Args:
            paramvec (list): list of parameter values (numerical)

        Returns:
            xnp.ndarray: T matrix with numerical values
        """
        tmat_eval = self.cfg.tmat_symb.evalf(subs={self.symbolvec[i]: paramvec[i] for i in range(len(paramvec))})
        # convert to numpy array, then to xnp (jax cannot convert from sympy directly)
        return xnp.asarray(np.asarray(tmat_eval).astype(complex))

    @property
    def tmat_layervec_unitcellvec(self) -> list[list[xnp.ndarray]]:
        if self._tmat_layervec_unitcellvec is None:
            self.cfg.enforce_parameter_conditions(self.cfg.paramvec)
            self._tmat_layervec_unitcellvec = []
            for layer in range(self.cfg.nlayer):
                tmats = [self._eval_tmat_symb(self.cfg.paramvec[layer][ind]) for ind in range(self.cfg.unitcell_size)]
                self._tmat_layervec_unitcellvec.append(tmats)
        return self._tmat_layervec_unitcellvec

    @property
    def tmat_layervec_sitevec(self) -> list[list[xnp.ndarray]]:
        """
        Generate the T-matrix vector (single virtual fermion on the link).
        Analytically, this mode order is not advantageous,
        but it makes the reshuffling of the modes easier for gamma_in and M_D in the covariance matrix.

        Returns:
            xnp.ndarray: parameter matrix T
        """
        if self._tmat_layervec_sitevec is None:
            self.cfg.enforce_parameter_conditions(self.cfg.paramvec)
            self._tmat_layervec_sitevec = []
            for layer in range(self.cfg.nlayer):
                tmats = self.tmat_layervec_unitcellvec[layer]
                tmat_lay = [tmats[self.cfg.site_params_dict[site]] for site in range(self.cfg.lattice.size)]
                self._tmat_layervec_sitevec.append(tmat_lay)
        return self._tmat_layervec_sitevec

    @property
    def gamma_dirac_layervec_sitevec(self) -> xnp.ndarray:
        """Return the vector of covariance matrices in dirac modes.

        Returns:
            xnp.ndarray: Vector of covariance matrices in Dirac modes
        """
        if self._gamma_dirac_layervec_sitevec is None:

            gamma_dirac_layervec_sitevec = []
            for lay in range(self.cfg.nlayer):

                gamma_dirac_lay = [
                    xnp.array(utils.tmat_to_covariance_matrix(tmat)) for tmat in self.tmat_layervec_sitevec[lay]
                ]
                gamma_dirac_layervec_sitevec.append(gamma_dirac_lay)

            self._gamma_dirac_layervec_sitevec = xnp.array(gamma_dirac_layervec_sitevec)
        return self._gamma_dirac_layervec_sitevec

    @property
    def gamma_maj_layervec_sitevec(self) -> xnp.ndarray:
        r"""Return the covariance matrix in Majorana modes.
        The definition of Majorana modes used is
            \gamma_1 = c + c^\dagger
            \gamma_2 = i(c - c^\dagger)

        This is a get function.

        Returns:
            xnp.ndarray: list of covariance matrices in Majorana modes for all layers
        """
        if self._gamma_maj_layervec_sitevec is None:

            # We know that the gamma dirac matrices have all the same shape
            m, _ = self.gamma_dirac_layervec_sitevec[-1][0].shape
            smat = utils.generate_smat(m)

            # Vectorized operation over all layers and sites
            # note: since self.gamma_dirac_layervec_sitevec is already a vector over sites, here we are being
            #       slightly inneficent - we do the matrix multiplication for each entry, even though many of
            #       the gamma_dirac's are the same.
            self._gamma_maj_layervec_sitevec = xnp.real(smat @ self.gamma_dirac_layervec_sitevec @ xnp.transpose(smat))
        return self._gamma_maj_layervec_sitevec

    def _expand_gamma_maj_to_system(self, covmats_layervec_sitevec):
        """Expand the covariance matrix in Majorana modes to the full system.
        In order to obtain a structure that is convenient for further computations,
            (A    B)
            (-B^T D)
        we have to reorder the modes of the single-vertex matrix with respect to the full matrix.

        This method is overwritten for the U1 system.

        Args:
            covmats_layervec_sitevec (list[list[xnp.ndarray]]):
                list (per layer) of 2D covariance matrices of all sites; total shape (nlayer, nsites, nmodes, nmodes)

        Returns:
            xnp.ndarray: 2D covariance matrix of the full system
        """

        # Build permutation matrix to convert modes from site order to link order
        # be careful with the convention of the permutation matrix vs its transpose; this way works with the code below
        modes_link_order = self.get_link_based_mode_order()
        modes_site_order = self.get_site_based_mode_order()
        mat_perm_links = generate_permutation_matrix(modes_site_order, modes_link_order)
        sites_perm = xnp.eye(
            2 * self.cfg.lattice.nx * self.cfg.lattice.ny * self.cfg.nphysmodes_site
        )  # total number of physical fermionic majorana modes on all the sites together
        mat_perm = block_diag(sites_perm, mat_perm_links)

        # TODO: properly vectorize!
        gamma_maj_sys_vec = []
        for covmats_sitevec in covmats_layervec_sitevec:
            covmats = covmats_sitevec

            # Extract the parts of the covariance matrix
            offset = 2 * self.cfg.nphysmodes_site  # number of physical (majarona) modes per site
            amats = [covmats[site][:offset, :offset] for site in range(self.cfg.lattice.size)]
            bmats = [covmats[site][:offset, offset:] for site in range(self.cfg.lattice.size)]
            dmats = [covmats[site][offset:, offset:] for site in range(self.cfg.lattice.size)]
            # Expand them
            amat_sys = block_diag(*amats)
            bmat_sys = block_diag(*bmats)
            dmat_sys = block_diag(*dmats)
            # Reassemble them in the correct order
            mat_sys_unordered = xnp.block([[amat_sys, bmat_sys], [-xnp.transpose(bmat_sys), dmat_sys]])
            dest = xnp.transpose(mat_perm) @ mat_sys_unordered @ mat_perm
            gamma_maj_sys_vec.append(dest)
        return xnp.array(gamma_maj_sys_vec)

    @staticmethod
    @maybe_jit(
        static_argnames=[
            "lattice_size",
            "nphysmodes_site",
            "num_pg_layer",
            "num_fermionic_layer",
            "unitcell_size",
            "nparams",
        ]
    )
    def _compute_d_gamma_out_symbolvec(
        lattice_size: int,
        nphysmodes_site: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        unitcell_size: int,
        nparams: int,
        mat_b_vec: xnp.ndarray,
        gamma_out_inv_vec: xnp.ndarray,
        gamma_maj_sys_deriv_layvec_ucvec_symbvec: xnp.ndarray,
    ) -> xnp.ndarray:
        nlayer = num_pg_layer + num_fermionic_layer
        dim_gamma_out = 2 * lattice_size * nphysmodes_site
        shape = (nlayer, unitcell_size, nparams, dim_gamma_out, dim_gamma_out)
        d_gamma_out_symbolvec = xnp.zeros(shape)

        for layer in range(num_pg_layer, nlayer):
            for uc_ind in range(unitcell_size):
                offset = dim_gamma_out

                for symbol_ind in range(nparams):
                    # TODO: skip for zeroed params

                    mat_b = mat_b_vec[layer]
                    deriv_gamma_maj_sys = gamma_maj_sys_deriv_layvec_ucvec_symbvec[layer, uc_ind, symbol_ind]
                    d_mat_a, d_mat_b, d_mat_d = utils.extract_partial_covmats(deriv_gamma_maj_sys, offset)
                    diff_d_gamma_inv = gamma_out_inv_vec[layer]
                    d_gamma_out = (
                        d_mat_a
                        + d_mat_b @ diff_d_gamma_inv @ xnp.transpose(mat_b)
                        + mat_b @ diff_d_gamma_inv @ xnp.transpose(d_mat_b)
                        - mat_b @ diff_d_gamma_inv @ d_mat_d @ diff_d_gamma_inv @ xnp.transpose(mat_b)
                    )

                    d_gamma_out_symbolvec = backend.array_assign(
                        d_gamma_out_symbolvec, (layer, uc_ind, symbol_ind), d_gamma_out
                    )

        return d_gamma_out_symbolvec

    @property
    def d_gamma_out_symbolvec(self) -> xnp.ndarray:
        """Return a vector containing the derivatives of gamma_out for each symbol.

        The derivatives for pure-gauge layers are set to nan.

        Returns:
            array: d_gamma_out_symbolvec[layer, uc_ind, symbol] is the matrix of derivatives of gamma_out[lay]
                   wrt the parameter at that (lay, uc_ind, symbol)
        """
        if self._d_gamma_out_symbolvec is None:
            gamma_out_inv_vec = xnp.array([self.wi_gamma_out_vec[layer].inv() for layer in range(self.cfg.nlayer)])
            self._d_gamma_out_symbolvec = self._compute_d_gamma_out_symbolvec(
                self.cfg.lattice.size,
                self.cfg.nphysmodes_site,
                self.cfg.num_pg_layer,
                self.cfg.num_fermionic_layer,
                self.cfg.unitcell_size,
                len(self.cfg.symbolvec),
                self.mat_b_vec,
                gamma_out_inv_vec,
                self.gamma_maj_sys_deriv_layvec_ucvec_symbvec,
            )
        return self._d_gamma_out_symbolvec

    @property
    def gamma_maj_sys_vec(self):
        """Return the covariance matrix of the full system in Majorana modes.
        The mode order is changed to fit the mode order of gamma_in.
        See documentation of gamma_in for details.

        This is a get function.

        Returns:
            [xnp.ndarray]: Covariance matrix of the full system
        """
        if self._gamma_maj_sys_vec is None:
            self._gamma_maj_sys_vec = self._expand_gamma_maj_to_system(self.gamma_maj_layervec_sitevec)
        return self._gamma_maj_sys_vec

    @property
    def mat_a_vec(self):
        """Extract the matrix for physical-physical correlations.
        The mode ordering of this matrix is (p_1(0,0),p_2(0,0),p_1(1,0),p_2(1,0)....).
        It is formulated in terms of Majorana modes.
        The mode ordering of the sites is identical to the site convention defined in the lattice class.
        There is a vector of A matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            [xnp.ndarray]: Correlations of the physcial modes for the full system.
        """
        if self._mat_a_vec is None:
            offset = 2 * self.cfg.lattice.size * self.cfg.nphysmodes_site
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = self._extract_partial_covmatvec(offset)
        return self._mat_a_vec

    @property
    def mat_b_vec(self):
        """Extract the matrix for physical-virtual correlations.
        There is a vector of B matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            [xnp.ndarray]: Correlations of the physcial modes with the virtual modes for the full system.
        """
        if self._mat_b_vec is None:
            offset = 2 * self.cfg.lattice.size * self.cfg.nphysmodes_site
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = self._extract_partial_covmatvec(offset)
        return self._mat_b_vec

    @property
    def mat_d_vec(self):
        """Extract the matrix for virtual-virtual correlations (aka D).
        There is a vector of D matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            [xnp.ndarray]: Correlations of the virtual modes for the full system.
        """
        if self._mat_d_vec is None:
            offset = 2 * self.cfg.lattice.size * self.cfg.nphysmodes_site
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = self._extract_partial_covmatvec(offset)
        return self._mat_d_vec

    @property
    def det_mat_d_vec(self):
        """Compute the determinant of the virtual-virtual correlation matrix.
        There is a vector of D matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            xnp.ndarray: list of log-determinants
        """
        if self._det_mat_d_vec is None:
            _, self._det_mat_d_vec = xnp.linalg.slogdet(self.mat_d_vec)
        return self._det_mat_d_vec

    @property
    def mat_d_inv_vec(self):
        """Compute the vector of the inverses of the D matrix.
        The D matrix is the correlation matrix of virtual-virtual correlations.
        There is a vector of D matrices if multiple layers are used; len(vec) = # of layers
        This is a get function.

        Returns:
            xnp.ndarray: List of matrix inverses of the D matrix
        """
        if self._mat_d_inv_vec is None:
            self._mat_d_inv_vec = xnp.linalg.inv(self.mat_d_vec)
        return self._mat_d_inv_vec

    def mat_a_mod_vec(self, link_ind: int):
        """Extract the matrix for physical-physical correlations and one virtual mode.
        This shifted matrix is used for the computation of the electric energy.
        The mode ordering of this matrix is (p_1(0,0),p_2(0,0),p_1(1,0),p_2(1,0)....).
        It is formulated in terms of Majorana modes.
        The mode ordering of the sites is identical to the site convention defined in the lattice class.
        There is a vector of A matrices if multiple layers are used; len(vec)==# of copies
        "Physical" modes here includes the virtual modes on the link on which the electric energy is computed.
        This is a get function.

        Args:
            link_ind (int, optional): Index of the link to include in the physical-physical set.

        Returns:
            [xnp.ndarray]: Correlations of the physical modes for the full system.
        """
        if self._mat_a_mod_vec is None:
            self._mat_a_mod_vec, self._mat_b_mod_vec, self._mat_d_mod_vec = utils.extract_mod_covmats(
                self.gamma_maj_sys_vec,
                link_ind,
                self.cfg.lattice.size,
                self.cfg.nphysmodes_site,
                self.cfg.nvirtmodes_link,
            )
        return self._mat_a_mod_vec

    def mat_b_mod_vec(self, link_ind: int):
        """Extract the matrix for physical-virtual correlations.
        This matrix contains one link less than the original matrix (used for the electric energy computation)
        There is a vector of B matrices if multiple layers are used; len(vec)==# of copies
        "Physical" modes here includes the virtual modes on the link on which the electric energy is computed.
        This is a get function.

        Args:
            link_ind (int, optional): Index of the link to include in the physical-physical set.

        Returns:
            [xnp.ndarray]: Correlations of the physical modes with the virtual modes for the full system.
        """
        if self._mat_b_mod_vec is None:
            self._mat_a_mod_vec, self._mat_b_mod_vec, self._mat_d_mod_vec = utils.extract_mod_covmats(
                self.gamma_maj_sys_vec,
                link_ind,
                self.cfg.lattice.size,
                self.cfg.nphysmodes_site,
                self.cfg.nvirtmodes_link,
            )
        return self._mat_b_mod_vec

    def mat_d_mod_vec(self, link_ind: int):
        """Extract the matrix for virtual-virtual correlations.
        This matrix contains one link less than the original matrix (used for the electric energy computation)
        There is a vector of D matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Args:
            link_ind (int, optional): Index of the link to include in the physical-physical set.

        Returns:
            [xnp.ndarray]: Correlations of the virtual modes for the full system.
        """
        if self._mat_d_mod_vec is None:
            self._mat_a_mod_vec, self._mat_b_mod_vec, self._mat_d_mod_vec = utils.extract_mod_covmats(
                self.gamma_maj_sys_vec,
                link_ind,
                self.cfg.lattice.size,
                self.cfg.nphysmodes_site,
                self.cfg.nvirtmodes_link,
            )
        return self._mat_d_mod_vec

    def det_mat_d_mod_vec(self, link_ind: int):
        """
        Compute the determinant of the virtual-virtual correlation matrix for the modified matrix.
        There is a vector of determinants of D matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Args:
            link_ind (int, optional): Index of the link to include in the physical-physical set.

        Returns:
            list: list of log-determinants
        """
        if self._det_mat_d_mod_vec is None:
            _, self._det_mat_d_mod_vec = xnp.linalg.slogdet(self.mat_d_mod_vec(link_ind))
        return self._det_mat_d_mod_vec

    def mat_d_mod_inv_vec(self, link_ind: int):
        """Compute the inverse of modified D matrices.
        There is a vector of inverses of D matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Args:
            link_ind (int, optional): Index of the link to include in the physical-physical set.

        Returns:
            xnp.ndarray: List of inverses of modified D matrices
        """
        if self._mat_d_mod_inv_vec is None:
            self._mat_d_mod_inv_vec = xnp.linalg.inv(self.mat_d_mod_vec(link_ind))
        return self._mat_d_mod_inv_vec

    def covmat_out_virt_vec(self, link_ind: int) -> list[xnp.ndarray]:
        """Compute the convariance matrix of the state, including physical fermions and
        the virtual fermions on the link on which the electric energy is computed.
        This function returns a vector over layers of these covariance matrices.
        This is a get function.

        Returns:
            list[xnp.array]: a vector of covariance matrices
        """

        if self._covmat_out_virt_vec is None:
            covmat_out_virt_vec = []

            # Number of fermions = # of sites
            # Since we have 2 copies, we get 8 virtual fermions per site
            single_link_offset = 2 * self.cfg.nvirtmodes_link

            # TODO: vectorize!
            for layerind in range(self.cfg.nlayer):

                # We shift the first virtual link (0,0,X) towards the physical modes to trace out everything else
                mat_a = self.mat_a_mod_vec(link_ind)[
                    layerind
                ]  # dim: 2*nsites (for majorana) + 8 (= 4 virtual modes per link x2 for majorana)
                mat_b = self.mat_b_mod_vec(link_ind)[layerind]
                diff_d_gamma_inv = self.wi_gamma_out_mod_vec[layerind].inv()

                ###################### Calculation of <P> ########################
                covmat_out = mat_a + mat_b @ diff_d_gamma_inv @ xnp.transpose(mat_b)
                size = covmat_out.shape[1]
                covmat_out_virt = backend.slice_matrix(
                    covmat_out,
                    size - single_link_offset,
                    size,
                    size - single_link_offset,
                    size,
                )

                # The library pfapack (used in the el energy) is rather picky about the anti-symmetrization (to 1e-14)
                covmat_out_virt = utils.anti_symmetrize(covmat_out_virt)
                covmat_out_virt_vec.append(covmat_out_virt)

            self._covmat_out_virt_vec = covmat_out_virt_vec
        return self._covmat_out_virt_vec

    def norm_mod_vec(self, link_ind: int) -> list[float]:
        """Compute the normalization factor for the modified covariance matrix.
        This is used to compute the electric energy.
        This function returns a vector over layers of these normalization factors.
        This is a get function.

        Returns:
            list[float]: a vector of modified normalization factors
        """
        if self._norm_mod_vec is None:
            norm_mod_vec = []
            lognormvec_default = self.lognorm_default_vec

            for layerind in range(self.cfg.nlayer):
                # For the modified norm, we still have to take into account the other contributions
                # from the unmodified parts
                norm_mod = self._calculate_lognorm_inc(
                    [self.incdet_mod_vec[layerind]],
                    [self.det_mat_d_mod_vec(link_ind)[layerind]],
                    self.gamma_in_sys_mod_vec(link_ind)[layerind].shape[0],
                    all_factors=True,
                )

                norm_mod += utils.add_except(lognormvec_default, layerind)
                norm_mod_vec.append(norm_mod)

            self._norm_mod_vec = norm_mod_vec
        return self._norm_mod_vec

    @property
    def lognorm_default_vec(self) -> xnp.ndarray:
        """Compute the log of the norm of the state, with no modifications (e.g. those of the electric energy),
        including all factors.
        This function returns a vector over layers of these norms.
        This is a get function.

        Returns:
            array: a vector of norms
        """
        if self._lognorm_default_vec is None:
            self._lognorm_default_vec = self.calculate_lognormvec_inc(all_factors=True)
        return self._lognorm_default_vec

    def initialize_gamma_in_and_trackers(self):
        """Initialize gamma_in (the covariance matrix of the projectors),
        as well as the trackers (Woodbury inverters and incremental determinants), which depend on gamma_in.

        The mode-order in gamma_in_sys is dictated by the numbering of the links on the lattice.
        The numbering guarantees that we split the vertical from the horizontal links for easier gauging.

            |         |
            "5"       "7"
            |         |
            2 --"2"-- 3 --"3"--
            |         |
            "4"       "6"
            |         |
            0 --"0"-- 1 --"1"--

        The vertex indices are written as <number>, the link indices are written as "<number>".

        For a 2x2 system with 1 copy (one virtual fermions per site per link), gamma_in has the order
        {l_1, r_0, l_0, r_1, l_3, r_2, l_2, r_3, d_2, u_0, d_0, u_2, d_3, u_1, d_1, d_3}.

        For a 2x2 system with 2 copies (two virtual fermions per site per link), gamma_in has the order
        { l1_1, r2_0, l1_1, r2_0, l1_0, r2_1, l1_0, r2_1,
          l1_3, r2_2, l1_3, r2_2, l1_2, r2_3, l1_2, r2_3,
          d1_2, u2_0, d1_2, u2_0, d1_0, u2_2, d1_0, u2_2,
          d1_3, u2_1, d1_3, u2_1, d1_1, d2_3, d1_1, d2_3 }.

        The naming convention here is <mode letter><number of copy>_<vertex index>.
        (<number of copy> is ommitted for the 1 copy case).
        Each constituent in the lists above refers to two Majorana modes.

        This method overwrites an abstract method in System2DBase.
        """

        # Initialize empty lists
        gamma_in_sys_vec = []
        wi_gamma_in_vec, wi_gamma_out_vec, incdet_vec = [], [], []
        wi_gamma_in_mod_vec, wi_gamma_out_mod_vec, incdet_mod_vec = [], [], []

        # Initialize gamma_in_sys for the full system (and trackers)
        size = self.cfg.lattice.size  # number of sites
        id = xnp.eye(size)

        # TODO: vectorize!
        for layer in range(self.cfg.nlayer):
            neutral_gauge_X = xnp.kron(id, self.gamma_gauge_neutral_vec[layer][Direction.X])
            neutral_gauge_Y = xnp.kron(id, self.gamma_gauge_neutral_vec[layer][Direction.Y])
            gamma_in_sys = xscipy.linalg.block_diag(neutral_gauge_X, neutral_gauge_Y)
            gamma_in_sys_vec.append(gamma_in_sys)

            wi_gamma_in_vec.append(utils.WoodburyInverter(self.mat_d_inv_vec[layer] - gamma_in_sys))
            wi_gamma_out_vec.append(utils.WoodburyInverter(self.mat_d_vec[layer] - gamma_in_sys))
            incdet_vec.append(utils.IncLogAbsDeterminant(self.mat_d_inv_vec[layer] - gamma_in_sys))
        gamma_in_sys_vec = xnp.array(gamma_in_sys_vec)

        # Initialize the modified gamma_in_sys and trackers for the full system
        # We do this in a separate loop, so that we can use the full gamma_in_sys_vec which is built in the previous loop.
        ind = self.mod_link_ind  # index of the link to exclude
        gamma_in_sys_mod_vec = self._extract_gamma_in_sys_mod_vec(ind, gamma_in_sys_vec)
        for layer in range(self.cfg.nlayer):
            gamma_in_sys_mod = gamma_in_sys_mod_vec[layer]
            wi_gamma_in_mod_vec.append(utils.WoodburyInverter(self.mat_d_mod_inv_vec(ind)[layer] - gamma_in_sys_mod))
            wi_gamma_out_mod_vec.append(utils.WoodburyInverter(self.mat_d_mod_vec(ind)[layer] - gamma_in_sys_mod))
            incdet_mod_vec.append(utils.IncLogAbsDeterminant(self.mat_d_mod_inv_vec(ind)[layer] - gamma_in_sys_mod))

        return (
            gamma_in_sys_vec,
            (wi_gamma_in_vec, wi_gamma_out_vec, incdet_vec),
            (wi_gamma_in_mod_vec, wi_gamma_out_mod_vec, incdet_mod_vec),
        )

    @property
    def gamma_in_sys_vec(self):
        """Get function to return the gauged gamma_in_sys_vec, the covariance matrices
        of the links for the whole system for each layer.

        Returns:
            xnp.ndarray: vector of gauged covariance matrices of the system
        """
        if self._gamma_in_sys_vec is None:
            self._gamma_in_sys_vec, full_tuple, mod_tuple = self.initialize_gamma_in_and_trackers()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            (
                self._wi_gamma_in_mod_vec,
                self._wi_gamma_out_mod_vec,
                self._incdet_mod_vec,
            ) = mod_tuple
        return self._gamma_in_sys_vec

    @property
    def incdet_vec(self):
        """Return the vector of incremental determinants for the different layers.
        The length of the list is equal to the number of layers.
        This is a get function.

        Returns:
            list: List of incremental determinant trackers
        """
        if self._incdet_vec is None:
            self._gamma_in_sys_vec, full_tuple, mod_tuple = self.initialize_gamma_in_and_trackers()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            (
                self._wi_gamma_in_mod_vec,
                self._wi_gamma_out_mod_vec,
                self._incdet_mod_vec,
            ) = mod_tuple
        return self._incdet_vec

    @property
    def wi_gamma_in_vec(self):
        """Return the vector of Woodbury inverters for (D^-1 - gammain)^-1 for the different layers.
        The length of the list is equal to the number of layers.
        This is a get function.

        Returns:
            list: List of Woodbury inverters
        """
        if self._wi_gamma_in_vec is None:
            self._gamma_in_sys_vec, full_tuple, mod_tuple = self.initialize_gamma_in_and_trackers()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            (
                self._wi_gamma_in_mod_vec,
                self._wi_gamma_out_mod_vec,
                self._incdet_mod_vec,
            ) = mod_tuple
        return self._wi_gamma_in_vec

    @property
    def wi_gamma_out_vec(self):
        """Return the vector of Woodbury inverters for (D - gammain)^-1 for the different layers.
        The length of the list is equal to the number of layers.
        This is a get function.

        Returns:
            list: List of Woodbury inverters
        """
        if self._wi_gamma_out_vec is None:
            self._gamma_in_sys_vec, full_tuple, mod_tuple = self.initialize_gamma_in_and_trackers()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            (
                self._wi_gamma_in_mod_vec,
                self._wi_gamma_out_mod_vec,
                self._incdet_mod_vec,
            ) = mod_tuple
        return self._wi_gamma_out_vec

    def gamma_in_sys_mod_vec(self, link_ind: int):
        """Get function to return the gauged gamma_in_sys_vec with a single link modification
        (to compute the electric energy),
        the covariance matrix of the links for the whole system.

        Args:
            link_ind (int, optional): Index of the link to exclude.

        Returns:
            xnp.ndarray: Gauged, modified covariance matrices of the system for each layer
        """
        gamma_in_sys_mod_vec = self._extract_gamma_in_sys_mod_vec(link_ind, self.gamma_in_sys_vec)
        return gamma_in_sys_mod_vec

    def _extract_gamma_in_sys_mod_vec(self, link_ind: int, gamma_in_sys_vec: xnp.ndarray):
        gamma_in_sys_mod_vec = []

        # Calculate the indices of gamma_in to extract
        virt_start = 2 * self.cfg.nvirtmodes_link * link_ind  # start of virtual modes for the link
        virt_end = virt_start + 2 * self.cfg.nvirtmodes_link  # end of virtual modes for the link
        size = gamma_in_sys_vec.shape[1]  # size of gamma_in_sys

        inds = [k for k in range(virt_start, virt_end)]  # inds virt modes on link
        virt_inds = xnp.asarray([k for k in range(size) if k not in inds])  # all other virtual modes

        rows, cols = xnp.ix_(virt_inds, virt_inds)
        gamma_in_sys_mod_vec = gamma_in_sys_vec[:, rows, cols]
        return gamma_in_sys_mod_vec

    @property
    def incdet_mod_vec(self):
        """Return the vector of incremental determinants for the modified matrices for the different layers.
        The length of the list is equal to the number of layers.
        This is a get function.

        Returns:
            list: List of incremental determinant trackers
        """
        if self._incdet_mod_vec is None:
            self._gamma_in_sys_vec, full_tuple, mod_tuple = self.initialize_gamma_in_and_trackers()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            (
                self._wi_gamma_in_mod_vec,
                self._wi_gamma_out_mod_vec,
                self._incdet_mod_vec,
            ) = mod_tuple
        return self._incdet_mod_vec

    @property
    def wi_gamma_in_mod_vec(self):
        """Return the vector of Woodbury inverters for (D^-1 - gammain_mod)^-1 for the different layers.
        The length of the list is equal to the number of layers.
        This is a get function.

        Returns:
            list: List of Woodbury inverters
        """
        if self._wi_gamma_in_mod_vec is None:
            self._gamma_in_sys_vec, full_tuple, mod_tuple = self.initialize_gamma_in_and_trackers()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            (
                self._wi_gamma_in_mod_vec,
                self._wi_gamma_out_mod_vec,
                self._incdet_mod_vec,
            ) = mod_tuple
        return self._wi_gamma_in_mod_vec

    @property
    def wi_gamma_out_mod_vec(self):
        """Return the vector of Woodbury inverters for (D - gammain_mod)^-1 for the different layers.
        The length of the list is equal to the number of layers.
        This is a get function.

        Returns:
            list: List of Woodbury inverters
        """
        if self._wi_gamma_out_mod_vec is None:
            self._gamma_in_sys_vec, full_tuple, mod_tuple = self.initialize_gamma_in_and_trackers()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            (
                self._wi_gamma_in_mod_vec,
                self._wi_gamma_out_mod_vec,
                self._incdet_mod_vec,
            ) = mod_tuple
        return self._wi_gamma_out_mod_vec

    ################## Computation of derivatives ######################

    def compute_gamma_dirac_deriv(self, symb: sympy.Symbol, layerind: int, uc_ind: int):
        """Return the numerical derivative of the gamma_dirac, the Dirac covariance matrix of one fiducial state.

        Args:
            symb (sympy.Symbol): Symbol with respect to which we derive
            layerind (int): index of the layer

        Returns:
            xnp.ndarray: Derivative of gamma_dirac wrt to symb
        """
        deriv_t = self.compute_tmat_deriv(symb)
        tmat = self.tmat_layervec_unitcellvec[layerind][uc_ind]
        tmatc = xnp.conjugate(tmat)
        idttinv_minus = xnp.linalg.inv(xnp.eye(deriv_t.shape[0]) - tmat @ tmatc)
        idtt_plus = xnp.eye(deriv_t.shape[0]) + tmat @ tmatc
        d_idtt_minus = -(deriv_t @ tmatc + tmat @ xnp.conjugate(deriv_t))
        d_idtt_plus = -d_idtt_minus
        d_lt = idttinv_minus @ d_idtt_minus @ idttinv_minus @ tmat - idttinv_minus @ deriv_t
        d_rt = 0.5 * idttinv_minus @ d_idtt_plus @ idttinv_minus @ idtt_plus + 0.5 * idttinv_minus @ d_idtt_plus
        d_lb = -xnp.conjugate(d_rt)
        d_rb = -xnp.conjugate(d_lt)
        return 1.0j * xnp.block([[d_lt, d_rt], [d_lb, d_rb]])

    def compute_gamma_maj_deriv(self, symb: sympy.Symbol, layerind: int, uc_ind: int):
        """Return the numerical derivative of the gamma_maj, the Majorana covariance matrix of one fiducial state.

        Args:
            symb (sympy.Symbol): Symbol with respect to which we derive
            layerind (int): index of the layer

        Returns:
            xnp.ndarray: Derivative of gamma_maj wrt to symb
        """
        gamma_dirac_deriv = self.compute_gamma_dirac_deriv(symb, layerind, uc_ind)
        m, _ = gamma_dirac_deriv.shape
        smat = utils.generate_smat(m)
        return xnp.real(smat @ gamma_dirac_deriv @ xnp.transpose(smat))

    def _generate_gamma_maj_sys_deriv(self) -> xnp.ndarray:
        """Internal function to generate an array of all possible derivatives of gamma_maj_sys, the system-wide
        covariance matrix of the fiducial state.
        The dimensions of the array are
            (nlayer, unitcell_size, n_symbols, gamma_maj_dim, gamma_maj_dim)
        where
            gamma_maj_dim = 2 * nsites * (nphysmodes_site + 4 * ncopy)

        Returns:
            array: derivatives of gamma_maj_sys
        """
        lay_vec = []
        for lay in range(self.cfg.nlayer):
            uc_vec = []
            for uc_ind in range(self.cfg.unitcell_size):
                arr = []
                for symb in self.symbolvec:
                    gamma_maj_deriv = self.compute_gamma_maj_deriv(symb, lay, uc_ind)

                    gamma_maj_derivs_sitevec = []
                    for site in range(self.cfg.lattice.size):
                        if self.cfg.site_params_dict[site] == uc_ind:
                            gamma_maj_derivs_sitevec.append(gamma_maj_deriv)
                        else:
                            gamma_maj_derivs_sitevec.append(xnp.zeros_like(gamma_maj_deriv))

                    gamma_maj_sys_derivs = self._expand_gamma_maj_to_system([gamma_maj_derivs_sitevec])[0]
                    arr.append(gamma_maj_sys_derivs)
                uc_vec.append(arr)
            lay_vec.append(uc_vec)
        dest = xnp.array(lay_vec)
        return dest

    @property
    def gamma_maj_sys_deriv_layvec_ucvec_symbvec(self) -> xnp.ndarray:
        if self._gamma_maj_sys_deriv is None:
            self._gamma_maj_sys_deriv = self._generate_gamma_maj_sys_deriv()
        return self._gamma_maj_sys_deriv

    def gamma_maj_sys_deriv_vec(self, symb: sympy.Symbol) -> xnp.ndarray:
        """Return a list of derivatives of all layers of gamma_maj_sys with respect to a given symbol.
        This is a get function.

        Args:
            symb (sympy.Symbol): Symbol with repsect to which we derive

        Returns:
            xnp.ndarray: List of derived gamma_maj_sys
        """
        if symb not in self.symbolvec:
            raise ValueError(f"Symbol {symb} is not in the symbol vector.")

        arr = self.gamma_maj_sys_deriv_layvec_ucvec_symbvec[:, :, self.symbolvec.index(symb), :, :]
        return arr

    @classmethod
    @maybe_jit(
        static_argnames=[
            "cls",
            "lattice_size",
            "nphysmodes_site",
            "nlayer",
            "unitcell_size",
            "nparams",
            "zeroed_params",
        ]
    )
    def compute_grad_norm_vec(
        cls,
        lattice_size: int,
        nphysmodes_site: int,
        nlayer: int,
        unitcell_size: int,
        nparams: int,
        zeroed_params: list,
        wi_gamma_in_inv_vec: xnp.ndarray,
        gamma_in_sys_vec: xnp.ndarray,
        mat_d_inv_vec: xnp.ndarray,
        gamma_maj_sys_deriv_layvec_ucvec_symbvec: xnp.ndarray,
    ) -> xnp.ndarray:
        """Compute the gradient of the norm for all layers with respect to all parameters.

        Returns:
            xnp.ndarray: Vector of gradients of the norm with respect to all parameters of all layers and unit cells.
        """
        dest_grad = xnp.zeros((nlayer, unitcell_size, nparams))

        for layerind in range(nlayer):
            for uc_ind in range(unitcell_size):

                for symbol_ind in range(nparams):
                    if (layerind, uc_ind, symbol_ind) not in zeroed_params:
                        # the derivative calculation is computationally expensive
                        # we can skip it for parameters that are forced by the ansatz to be zero

                        # Compute gradient
                        grad = cls._compute_grad_over_norm(
                            lattice_size,
                            nphysmodes_site,
                            wi_gamma_in_inv_vec,
                            gamma_in_sys_vec,
                            mat_d_inv_vec,
                            gamma_maj_sys_deriv_layvec_ucvec_symbvec,
                            layerind,
                            uc_ind,
                            symbol_ind,
                        )

                        # Save
                        inds = (layerind, uc_ind, symbol_ind)
                        dest_grad = backend.array_assign(dest_grad, inds, grad)

        return dest_grad

    ################## Weight management ######################

    @property
    def weight(self):
        """Return the Monte Carlo weight of the current configuration.
        This function is a get function.
        If the value does not exist, it will be calculated.
        Each subsequent Monte Carlo step will update the weight.
        A full computation is only necessary in the beginning of the procedure.

        Returns:
            float: Weight of the MC configuration
        """
        if self._weight is None:
            self._weight = self.calculate_lognorm_inc()
        return self._weight

    @weight.setter
    def weight(self, val):
        """Setter of the weight"""
        self._weight = val

    def calculate_weight_attempt(self, link_ind: int, theta: float, all_factors=False):
        """
        Compute the weight of an update attempt in which the link index link_ind is substituted for theta
        The inclusion of all constant pre-factors can be switched on and off.

        For D2n gauge groups, we overwrite this function in the system implementation.

        Args:
            link_ind (int): Link index
            theta (float): New gauge field value
            all_factors (bool, optional): Include all constant factors. Defaults to False.

        Returns:
            float: Logarithm of the weight of the proposed configuration
        """
        # There are two directions per vertex and two Majoranas per link
        ind_mat = 2 * self.cfg.nvirtmodes_link * link_ind
        coord, dir = self.cfg.lattice.ind2coord_dir(link_ind)
        rotmat = self.generate_rotmat(self.cfg.ncopy, theta, coord, dir)
        gamma_neutral_gauge_vec = self.gamma_gauge_neutral_vec
        gamma_in_subst_layers = [
            rotmat @ gamma_neutral_gauge[dir] @ xnp.transpose(rotmat)
            for gamma_neutral_gauge in gamma_neutral_gauge_vec
        ]
        updates = [
            self.calculate_update_gamma_in(ind_mat, gamma_in_subst, gamma_in_sys)
            for gamma_in_subst, gamma_in_sys in zip(gamma_in_subst_layers, self.gamma_in_sys_vec)
        ]
        return self.update_lognorm_inc(ind_mat, updates, all_factors)

    def _calculate_lognorm(
        self, gamma_in_sys_vec: list[xnp.ndarray], mat_d_vec: list[xnp.ndarray], all_factors: bool = False
    ) -> float:
        # This is still the plain formula, without any update mechanism
        normvec = backend.calculate_lognormvec(gamma_in_sys_vec, mat_d_vec, all_factors=all_factors)
        return xnp.sum(normvec)

    def calculate_lognorm(self, all_factors=False):
        """Compute the logarithm of the norm

        Args:
            all_factors (bool, optional): Include all constant prefactors. Defaults to False.

        Returns:
            float: Logarithm of the norm
        """
        return self._calculate_lognorm(self.gamma_in_sys_vec, self.mat_d_vec, all_factors=all_factors)

    def calculate_lognormvec(self, all_factors=False):
        """Compute the logarithm of the norm for each layer

        Args:
            all_factors (bool, optional): Include all constant prefactors. Defaults to False.

        Returns:
            float: Logarithm of the norm
        """
        return backend.calculate_lognormvec(self.gamma_in_sys_vec, self.mat_d_vec, all_factors=all_factors)

    @classmethod
    def _calculate_lognorm_inc(cls, incdet_vec, det_mat_d_vec, n, all_factors: bool = False):
        det_vec = xnp.array([incdet.det() for incdet in incdet_vec])
        lognormvec = cls._calculate_lognormvec_inc(det_vec, det_mat_d_vec, n, all_factors=all_factors)
        return xnp.sum(lognormvec)

    @staticmethod
    @maybe_jit(static_argnames=["n", "all_factors"])
    def _calculate_lognormvec_inc(det_vec, det_mat_d_vec, n, all_factors: bool = False) -> xnp.ndarray:
        dest = []
        for ind in range(len(det_vec)):
            detval = det_vec[ind]
            if all_factors:
                detval -= n * xnp.log(2)
                detval += det_mat_d_vec[ind]
            # The factor 0.5 is the sqrt of the formula. We are storing the logarithm of the norm.
            # The addition of the cumval is the multiplication of the indpendent PEPS
            dest.append(0.5 * detval)
        return xnp.array(dest)

    def calculate_lognormvec_inc(self, all_factors=False) -> xnp.ndarray:
        """Compute the logarithm of the norm for all layers by incrementally updating the previous value
        (using IncDet and Woodbury)

        Args:
            all_factors (bool, optional): Include all pre-factors in the computation. Defaults to False.

        Returns:
            list: Vector of the incrementally updated norms for all layers
        """
        det_vec = xnp.array([incdet.det() for incdet in self.incdet_vec])
        res = self._calculate_lognormvec_inc(
            det_vec,
            self.det_mat_d_vec,
            self.gamma_in_sys_vec[0].shape[0],
            all_factors=all_factors,
        )
        return res

    def calculate_lognorm_inc(self, all_factors=False):
        """Update the logarithm of the norm incrementally (using IncDet and Woodbury)

        Args:
            all_factors (bool, optional): Include all pre-factors in the computation. Defaults to False.

        Returns:
            float: Logarithm of the norm (computed from IncDet and Woodbury)
        """
        normvec = self.calculate_lognormvec_inc(all_factors=all_factors)
        return xnp.sum(normvec)

    def update_lognorm_inc(self, offset, updates, all_factors=False):
        """Updat the logarithm of the norm incrementally with the given update.

        Args:
            offset (int): Offset into the matrix.
            update (xnp.ndarray): Update matrix to replace the current sub-matrix
            all_factors (bool, optional): Include all constant pre-factors. Defaults to False.

        Returns:
            float: Updated logarithmic value of the norm
        """
        cumval = 0
        for ind in range(self.cfg.nlayer):
            detval = self.incdet_vec[ind].update_index(
                self.wi_gamma_in_vec[ind].inv(),
                updates[ind],
                offset,
                offset,
                store=False,
            )
            if all_factors:
                detval -= self.gamma_in_sys_vec[0].shape[0] * xnp.log(2)
                detval += xnp.linalg.slogdet(self.mat_d_vec[ind])[1]
            # The factor 0.5 is the sqrt of the formula. We are storing the logarithm of the norm.
            # The addition of the cumval is the multiplication of the indpendent PEPS
            cumval += 0.5 * detval
        return cumval

    @staticmethod
    @maybe_jit(static_argnames=["lattice_size", "nphysmodes_site"])
    def _compute_grad_over_norm(
        lattice_size: int,
        nphysmodes_site: int,
        gamma_in_inv_vec: xnp.ndarray,
        gamma_in_sys_vec: xnp.ndarray,
        mat_d_inv_vec: xnp.ndarray,
        gamma_maj_sys_deriv: xnp.ndarray,
        layerind: int,
        uc_ind: int,
        symb_ind: int,
    ) -> float:
        """Compute the quotient of derivative of the norm over the norm itself.
        We can avoid a lot of factors by computing the quotient directly.

        Args:
            lattice_size (int): Size of the lattice (number of sites)
            nphysmodes_site (int): Number of physical modes per site
            gamma_in_inv_vec (xnp.ndarray): Vector of inverses of gamma_in
            gamma_in_sys_vec (xnp.ndarray):
            mat_d_inv_vec (xnp.ndarray): Vector of inverses of D matrices
            gamma_maj_sys_deriv (xnp.ndarray): Derivatives of the Majorana covariance
                matrix of the fulll system, shape (nlayer, unitcell_size, n_symbols, gamma
            layerind (int): layer index
            uc_ind (int): unit cell index
            symb_ind (int): index into symbolvec of parameter wrt which to take the derivative
                            (of the given lay, uc_ind)

        Returns:
            float: Value of the gradient divided by the norm of the state
        """
        diff = gamma_in_inv_vec[layerind]
        # 2 phys. Majorana modes per vertex, this is indepent of the number of copies or layers
        offset = 2 * lattice_size * nphysmodes_site
        # Extract only the part of the virtual-virtual correlations
        _, _, deriv_d = utils.extract_partial_covmats(gamma_maj_sys_deriv[layerind, uc_ind, symb_ind], offset)
        mat_d_inv = mat_d_inv_vec[layerind]

        # TODO: We might save one matrix-matrix multiplication here
        # The deriv_d and mat_d_inv are constant
        res = utils.compute_grad_over_norm(gamma_in_sys_vec[layerind], diff, deriv_d, mat_d_inv)
        return res

    def compute_grad_over_norm(self, layerind: int, uc_ind: int, symb_ind: int) -> float:
        """This is a wrapper for the compute_grad_over_norm function.
        We use a wrapper so that the inner computation function can be jit-compiled."""
        wi_gamma_in_inv_vec = xnp.array([self.wi_gamma_in_vec[layerind].inv() for layerind in range(self.cfg.nlayer)])
        res = self._compute_grad_over_norm(
            self.cfg.lattice.size,
            self.cfg.nphysmodes_site,
            wi_gamma_in_inv_vec,
            self.gamma_in_sys_vec,
            self.mat_d_inv_vec,
            self.gamma_maj_sys_deriv_layvec_ucvec_symbvec,
            layerind,
            uc_ind,
            symb_ind,
        )
        return res

    @property
    def grad_over_norm_vec(self) -> xnp.ndarray:
        """Compute the gradient of the norm over all parameters for all layers and unit cells.

        Returns:
            xnp.ndarray: Vector of gradients of the norm with respect to all parameters
        """
        if self._grad_over_norm_vec is None:
            wi_gamma_in_inv_vec = xnp.array(
                [self.wi_gamma_in_vec[layerind].inv() for layerind in range(self.cfg.nlayer)]
            )
            self._grad_over_norm_vec = self.compute_grad_norm_vec(
                self.cfg.lattice.size,
                self.cfg.nphysmodes_site,
                self.cfg.nlayer,
                self.cfg.unitcell_size,
                len(self.symbolvec),
                self.cfg.zeroed_params,
                wi_gamma_in_inv_vec,
                self.gamma_in_sys_vec,
                self.mat_d_inv_vec,
                self.gamma_maj_sys_deriv_layvec_ucvec_symbvec,
            )
        return self._grad_over_norm_vec

    ################## Local Gauge ######################

    @property
    def gaugefieldvec(self):
        """Return the vector of gauge fields.

        Returns:
            xnp.ndarray: Gauge fields of the simulation
        """
        return self._gaugefieldvec

    @gaugefieldvec.setter
    def gaugefieldvec(self, val):
        print(
            "Do not set the gaugefieldvec explicitly. Use 'update_gauge_ind'.",
            file=sys.stderr,
        )
        # TODO: log error

    @property
    def gamma_gauge_neutral_vec(self):
        if self._gamma_gauge_neutral_vec_dirs is None:
            self._gamma_gauge_neutral_vec_dirs = self._generate_gamma_gauge_neutral_dict()
        return self._gamma_gauge_neutral_vec_dirs

    def _generate_gamma_gauge_neutral_dict(self):
        """Define the ungauged covariance matrix of a single link.
        The substitution method must ensure a consistent order of the modes.
        The direction parameter controls which covariance matrix is retrieved,
        since these can differ between directions.
        """
        gamma_gauge_neutral_vec_dirs = self.cfg.generate_gamma_gauge_neutral_dict()
        return xnp.array(gamma_gauge_neutral_vec_dirs)

    @classmethod
    @abstractmethod
    def generate_rotmat(cls, ncopy, theta, coord, dir):
        """Generate the matrix to rotate gamma_in_neutral according to a given gauge field value.

        The mode order is (the same as for gamma_in_neutral and) provided explicitly in each subclass.
        We order first by link, then by copy, then by color.

        For pure gauge layers, modes of copy one are coupled to modes of copy 2. The projectors mix copies.
        For fermionic layers, the projectors don't mix copies to ensure the U(1) symmetry is obeyed.

        The sites are picked such that the left mode is right of the right modes,
        i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        This method must be overwritten in a subclass.

        Args:
            ncopy (int): number of copies of the ansatz
            g (fxnp.array): representation of group element
            coord (tuple): (x,y) coordinate on the lattice
            dir (lattice.Direction): direction of the link

        Returns:
            xnp.ndarray: Rotation matrix for gamma_in_neutral
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @abstractmethod
    def update_gauge_ind(self, link_ind, theta):
        """Update method that is called upon changing a gauge field.
        This method is central to the algorithm since it changes the gauged projectors
        and updates all incremental trackers of determinants and inverses.
        The re-calculation of determinants and inverses for the norm would be
        prohibitively expensive.

        This method overwrites an abstract method in System2DBase.

        Args:
            link_ind (int): Link index to be updated
            theta (xnp.array): New gauge field value
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    def update_gauge_full_system(self, gaugeconfig):
        """Replace all gauge fields on the links by the values given in gaugeconfig.

        Args:
            gaugeconfig (xnp.ndarray): Array of new values for the gauge field
        """
        for link_ind, gauge_val in enumerate(gaugeconfig):
            theta = xnp.asarray(gaugeconfig[link_ind])
            if not xnp.allclose(self._gaugefieldvec[link_ind], theta):
                # only actually do the update if it's a different gauge field
                self.update_gauge_ind(link_ind, gauge_val)

    def update_gauge_coord(self, coord, dir, theta):
        """Update a gauge field at a given coordinate and direction by a new value

        Args:
            coord (tuple): Coordinate of the vertex
            dir (Direction): Direction of the link
            theta (xnp.array): New value for the gauge field
        """
        ind = self.cfg.lattice.coord2ind_dir(coord, dir)
        self.update_gauge_ind(ind, theta)

    def calculate_update_gamma_in(self, offset, update_mat, gamma_in_sys):
        """Compute an update between the current gamma_in and the new gamma_in

        Args:
            offset (int): Offset in the matrix
            update_mat (xnp.ndarray): Array to replace the current content of gamma_in at offset
            gamma_in_sys (xnp.ndarray): gamma_in_sys. This is given as an argument so that different gamma_in_sys
            can be passed in when gamma_in_sys differs between layers.

        Returns:
            xnp.ndarray: Additional update to reach update_mat at gamma_in[offset:,offset:]
        """
        m_up, n_up = update_mat.shape
        gamma_in_old = backend.slice_matrix(gamma_in_sys, offset, offset + m_up, offset, offset + n_up)
        return -(update_mat - gamma_in_old)

    ################## Observables ######################
    @abstractmethod
    def _compute_mag_energy_op(self):
        """Compute the magnetic energy operator (w/o shift).
        This operator is diagonal in the gauge field (group element) basis and can thus
        be computed easily.

        This is an abstract method and has to be overwritten in a subclass.

        Args:
            use_trans_inv (bool, optional): Use the translationally invariant computation method. Defaults to True.

        Returns:
            float: magnetic energy w/o shift for a single plaquette
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    @abstractmethod
    def _compute_el_energy_op_vec(
        lognormvec_default,
        overall_factors,
        idxarrs,
        nlayer: int,
        covmat_out_virt_vec,
        norm_mod_vec,
        use_trans_inv: bool = True,
    ) -> xnp.ndarray:
        """Compute the electric energy.

        This is an abstract method and has to be overwritten in a subclass.

        Args:
            lognormvec_default: the usual norm without any modifications
            overall_factors: prefactors for building the required Pfaffians
            idxarrs: indices for building the required Pfaffians
            nlayer (int): total number of layers (pure gauge + fermionic)
            covmat_out_virt_vec:
            norm_mod_vec:
            use_trans_inv (bool, optional): Use the translationally invariant implementation. Defaults to True.

        Returns:
            list: list of electric energies for a single link
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    @abstractmethod
    def _compute_el_grad_vec(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        unitcell_size: int,
        nvirtmodes_link: int,
        nphysmodes_site: int,
        symbolvec: tuple,
        overall_factors,
        idxarr_vec,
        el_energy_vec,
        mat_b_mod_vec,
        gamma_in_sys_mod_vec,
        covmat_out_virt_vec,
        norm_mod_vec,
        lognorm_default_vec,
        wi_gamma_in_mod_inv_vec,
        wi_gamma_out_mod_inv_vec,
        mat_d_mod_inv_vec,
        gamma_maj_sys_deriv_layvec_ucvec_symbvec,
        grad_over_norm_vec,
        zeroed_params,
        use_trans_inv: bool = True,
    ):
        """Compute the electric energy gradients.
        We start by calculating the electric energies, since these are needed for evaluating the gradients.
        Since several operations needed for the computation of the gradient and the energy are similar,
        we can reuse many intermediate steps.

        This is an abstract method and has to be overwritten in a subclass.

        Args:
            use_trans_inv (bool, optional): Use the translationally invariant implementation. Defaults to True.

        Returns:
            list: list of gradients for the full system
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    @abstractmethod
    def _compute_mass_energy_op_vec(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        ferm_cov_vec: xnp.ndarray,
        use_trans_inv: bool = True,
    ):
        """Compute the mass term of the Hamiltonian (per layer).

        This is an abstract method and has to be overwritten in a subclass.

        Args:
            use_trans_inv (bool, optional): Use translationally invariant implementation. Defaults to True.

        Returns:
            array: mass energy as a vec over layers
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    @abstractmethod
    def _compute_mass_energy_grad(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        unitcell_size: int,
        symbolvec: tuple,
        d_gamma_out_symbolvec: xnp.ndarray,
        zeroed_params: tuple,
        use_trans_inv: bool = True,
    ):
        """Compute the mass energy gradient.

        This is an abstract method and has to be overwritten in a subclass.

        Args:
            use_trans_inv (bool, optional): Use translationally invariant implementation. Defaults to True.

        Returns:
            array: gradients of the mass energy
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    @abstractmethod
    def _compute_int_energy_op_vec(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        gaugefieldvec: xnp.ndarray,
        ferm_covmat_vec: xnp.ndarray,
        horizontal_neighbor_data: tuple,
        vertical_neighbor_data: tuple,
    ):
        """Compute the interaction energy (per layer) - the energy due to the interaction of the
        physical fermions with the gauge fields.

        This is an abstract method and has to be overwritten in a subclass.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    @abstractmethod
    def _compute_int_energy_grad(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        unitcell_size: int,
        nparams: int,
        gaugefieldvec: xnp.ndarray,
        d_gamma_out_symbolvec: xnp.ndarray,
        horizontal_neighbor_data: tuple,
        vertical_neighbor_data: tuple,
        zeroed_params: tuple,
    ):
        """Compute the interaction energy gradient.
        This is an abstract method and has to be overwritten in a subclass.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    @abstractmethod
    def _compute_chem_energy_op_vec(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        sublattice_factors: tuple,
        ferm_covmat_vec: xnp.ndarray,
    ) -> xnp.ndarray:
        """Compute the chemical potential energy (per layer).
        This is an abstract method and has to be overwritten in a subclass.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    @abstractmethod
    def _compute_chem_energy_grad(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        unitcell_size: int,
        symbolvec: tuple,
        sublattice_factors: tuple,
        zeroed_params: tuple,
        d_gamma_out_vec: xnp.ndarray,
    ) -> xnp.ndarray:
        """Compute the chemical potential energy gradient.
        This is an abstract method and has to be overwritten in a subclass.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @abstractmethod
    def _meson_string_vec(self, path):
        r"""Compute a layer resolved meson string for the given path.
        This is \psi^dagger (start) * String * \psi(end) before particle-hole,
        and assumes that start and end are on the same sublattice.

        This is an abstract method and has to be overwritten in a subclass.

        Args:
            path (list): List of tuples [(index,conj),....]. conj indicates whether the argument should be conjugated.

        Returns:
            array: meson_str_vec
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    ################## Energy Calculations ######################
    @property
    def energy(self) -> float:
        """Compute the total energy by adding all terms in the Hamiltonian
        This is a get function.

        Returns:
            float: Energy of the system
        """
        if self._energy is None:
            self._energy = 0.0
            if not utils.isclose(self.cfg.g_mag, 0):
                self._energy += self.mag_energy
            if not utils.isclose(self.cfg.g_el, 0):
                self._energy += self.el_energy
            if not utils.isclose(self.cfg.g_mass, 0):
                self._energy += self.mass_energy
            if not utils.isclose(self.cfg.g_int, 0):
                self._energy += self.int_energy
            if not np.allclose(self.cfg.g_chem, 0):
                self._energy += self.chem_energy
        return self._energy

    @property
    def mag_energy(self) -> float:
        """Compute magnetic energy with shift for the whole system
        This is a get function.

        Returns:
            float: magnetic energy
        """
        nplaq = self.cfg.lattice.nplaquettes
        mag_energy = self.cfg.g_mag * 2 * (nplaq - self.mag_energy_op)  # The 2 is for the hermitian conjugate
        return mag_energy

    # Functions that return a term of the energy in the Hamiltonian, including all
    # prefactors and energy from the entire lattice.
    @property
    def el_energy(self) -> float:
        """Compute electric energy with shift for the whole system
        This is a get function.

        Returns:
            float: electric energy
        """
        nlinks = self.cfg.lattice.nlinks
        el_energy = self.cfg.g_el * 2 * (nlinks - self.el_energy_op)
        return el_energy

    @property
    def mass_energy(self) -> float:
        """Compute mass energy for the whole system
        This is a get function.

        Returns:
            float: mass energy
        """
        mass_energy = self.cfg.g_mass * self.mass_energy_op
        return mass_energy

    @property
    def int_energy(self) -> float:
        """Compute interaction (of matter and gauge fields) energy for the whole system
        This is a get function.

        Returns:
            float: interaction energy
        """
        int_energy = self.cfg.g_int * self.int_energy_op
        return int_energy

    @property
    def chem_energy(self) -> float:
        """Compute chemical potential energy for the whole system
        This is a get function.

        Returns:
            float: chemical potential energy
        """
        chem_energy = 0.0
        for layer in range(self.cfg.num_pg_layer, self.cfg.nlayer):
            ind = layer - self.cfg.num_pg_layer
            chem_energy += self.cfg.g_chem[ind] * self.chem_energy_op_vec[layer]
        return chem_energy

    # Functions that return the energy for the operator part of a term in the Hamiltonian,
    # including the energy for the entire lattice, but not any shifts or prefactors.
    @property
    def mag_energy_op(self) -> float:
        """Compute the magnetic energy operator for the whole system without shift.
        This is a get function.

        Returns:
            float: Magnetic energy operator (w/o shift) for the whole system
        """
        if self._mag_energy_op is None:
            self._mag_energy_op = self._compute_mag_energy_op()
        return self._mag_energy_op

    @property
    def el_energy_op(self) -> float:
        """Compute electric energy (w/o shift) for the whole system.
        This is a get function.

        Returns:
            float: electric energy for the whole system w/o shift
        """
        if self._el_energy_op is None:
            # The different layers can be separated into separate PEPS and then multiplied together.
            nlinks = self.cfg.lattice.nlinks
            self._el_energy_op = nlinks * xnp.prod(self.el_energy_op_vec, dtype=float)
        return self._el_energy_op

    @property
    def mass_energy_op(self) -> float:
        """Compute the mass energy operator for the whole system without shift.
        This is a get function.

        Returns:
            float: Mass energy operator (w/o shift) for the whole system
        """
        if self._mass_energy_op is None:
            self._mass_energy_op = xnp.sum(self.mass_energy_op_vec)
            # don't multiply by the number of sites;
            # for the mass term this is assumed to happen lower down in the stack.
        return self._mass_energy_op

    @property
    def int_energy_op(self) -> float:
        """Compute the interaction energy operator for the whole system without shift.
        This is a get function.

        Returns:
            float: Interaction energy operator (w/o shift) for the whole system
        """
        if self._int_energy_op is None:
            self._int_energy_op = xnp.sum(self.int_energy_op_vec)
        return self._int_energy_op

    # Functions that return the layer-resolved energies of each energy operator
    @property
    def el_energy_op_vec(self) -> xnp.ndarray:
        """Compute electric energy operator w/o shift for all layers for the whole system.
        This is a get function.

        Returns:
            list: Layer-resolved electric energy w/o shift
        """
        if self._el_energy_op_vec is None:
            # This vector is the electric energy on a single link. Otherwise, we get a
            # power of nlinks in the product and the electric energy term (with prefactors) gets negative
            self._el_energy_op_vec = self._compute_el_energy_op_vec(
                self.lognorm_default_vec,
                self.cfg.el_overall_factors,
                self.cfg.idxarr_vec,
                self.cfg.nlayer,
                self.covmat_out_virt_vec(self.mod_link_ind),
                self.norm_mod_vec(self.mod_link_ind),
                use_trans_inv=True,
            )
        return self._el_energy_op_vec

    @property
    def mass_energy_op_vec(self) -> xnp.ndarray:
        """Compute mass energy operator w/o shift for all layers for the whole system.
        This is a get function.

        Returns:
            list: Layer-resolved mass energy w/o shift
        """
        if self._mass_energy_op_vec is None:
            self._mass_energy_op_vec = self._compute_mass_energy_op_vec(
                self.cfg.lattice.size,
                self.cfg.num_pg_layer,
                self.cfg.num_fermionic_layer,
                self.ferm_covmat_vec,
                use_trans_inv=True,
            )
        return self._mass_energy_op_vec

    @property
    def int_energy_op_vec(self) -> xnp.ndarray:
        """Compute interaction energy operator w/o shift for all layers for the whole system.
        This is a get function.

        Returns:
            list: Layer-resolved interaction energy w/o shift
        """
        if self._int_energy_op_vec is None:
            self._int_energy_op_vec = self._compute_int_energy_op_vec(
                self.cfg.lattice.size,
                self.cfg.num_pg_layer,
                self.cfg.num_fermionic_layer,
                self.gaugefieldvec,
                self.ferm_covmat_vec,
                self.cfg.lattice.horizontal_neighbor_data,
                self.cfg.lattice.vertical_neighbor_data,
            )
        return self._int_energy_op_vec

    @property
    def chem_energy_op_vec(self) -> xnp.ndarray:
        """Compute chemical potential energy operator w/o shift for all layers for the whole system.
        This is a get function.

        Returns:
            list: Layer-resolved interaction energy w/o shift
        """
        if self._chem_energy_op_vec is None:
            self._chem_energy_op_vec = self._compute_chem_energy_op_vec(
                self.cfg.lattice.size,
                self.cfg.num_pg_layer,
                self.cfg.num_fermionic_layer,
                self.cfg.lattice.sublattice_factors,
                self.ferm_covmat_vec,
            )
        return self._chem_energy_op_vec

    # Functions that return the layer-resolved gradients of each energy operator
    @property
    def el_energy_op_grad_vec(self) -> xnp.ndarray:
        """Compute the gradient of the electric operator (w/o shift) for all layers

        Returns:
            list: List of all electric energy gradients (w/o shift)
        """
        if self._el_energy_op_grad_vec is None:
            # In order to jit, we must pass arrays, not WoodburyInverter objects.
            wi_gamma_in_mod_inv_vec = xnp.asarray(
                [self.wi_gamma_in_mod_vec[lay].inv() for lay in range(self.cfg.nlayer)]
            )
            wi_gamma_out_mod_inv_vec = xnp.asarray(
                [self.wi_gamma_out_mod_vec[lay].inv() for lay in range(self.cfg.nlayer)]
            )

            self._el_energy_op_grad_vec = self._compute_el_grad_vec(
                self.cfg.lattice.size,
                self.cfg.num_pg_layer,
                self.cfg.num_fermionic_layer,
                self.cfg.unitcell_size,
                self.cfg.nvirtmodes_link,
                self.cfg.nphysmodes_site,
                tuple(self.cfg.symbolvec),
                self.cfg.el_overall_factors,
                self.cfg.idxarr_vec,
                self.el_energy_op_vec,
                self.mat_b_mod_vec(self.mod_link_ind),
                self.gamma_in_sys_mod_vec(self.mod_link_ind),
                self.covmat_out_virt_vec(self.mod_link_ind),
                self.norm_mod_vec(self.mod_link_ind),
                self.lognorm_default_vec,
                wi_gamma_in_mod_inv_vec,
                wi_gamma_out_mod_inv_vec,
                self.mat_d_mod_inv_vec(self.mod_link_ind),
                self.gamma_maj_sys_deriv_layvec_ucvec_symbvec,
                self.grad_over_norm_vec,
                self.cfg.zeroed_params,
                use_trans_inv=True,
            )
        return self._el_energy_op_grad_vec

    @property
    def mass_energy_op_grad_vec(self) -> xnp.ndarray:
        """Compute the gradient of the mass energy operator for the whole system without shift.
        This is a get function.

        Returns:
            float: gradient of the mass energy operator (w/o shift) for the whole system
        """
        if self._mass_energy_op_grad_vec is None:
            self._mass_energy_op_grad_vec = self._compute_mass_energy_grad(
                self.cfg.lattice.size,
                self.cfg.num_pg_layer,
                self.cfg.num_fermionic_layer,
                self.cfg.unitcell_size,
                tuple(self.cfg.symbolvec),
                self.d_gamma_out_symbolvec,
                self.cfg.zeroed_params,
                use_trans_inv=True,
            )
        return self._mass_energy_op_grad_vec

    @property
    def int_energy_op_grad_vec(self) -> xnp.ndarray:
        """Compute the gradient of the interaction energy operator for the whole system without shift.
        This is a get function.

        Returns:
            float: Gradient of the interaction energy operator (w/o shift) for the whole system
        """
        if self._int_energy_op_grad_vec is None:
            self._int_energy_op_grad_vec = self._compute_int_energy_grad(
                self.cfg.lattice.size,
                self.cfg.num_pg_layer,
                self.cfg.num_fermionic_layer,
                self.cfg.unitcell_size,
                len(self.cfg.symbolvec),
                self.gaugefieldvec,
                self.d_gamma_out_symbolvec,
                self.cfg.lattice.horizontal_neighbor_data,
                self.cfg.lattice.vertical_neighbor_data,
                self.cfg.zeroed_params,
            )
        return self._int_energy_op_grad_vec

    @property
    def chem_energy_op_grad_vec(self) -> xnp.ndarray:
        """Compute the gradient of the chemical potential energy operator for the whole system without shift.
        This is a get function.

        Returns:
            float: Gradient of the chemical potential energy operator (w/o shift) for the whole system
        """
        if self._chem_energy_op_grad_vec is None:
            self._chem_energy_op_grad_vec = self._compute_chem_energy_grad(
                self.cfg.lattice.size,
                self.cfg.num_pg_layer,
                self.cfg.num_fermionic_layer,
                self.cfg.unitcell_size,
                tuple(self.cfg.symbolvec),
                self.cfg.lattice.sublattice_factors,
                self.cfg.zeroed_params,
                self.d_gamma_out_symbolvec,
            )
        return self._chem_energy_op_grad_vec

    ##################  ######################

    @abstractmethod
    def occupation(self, lay: int, site: int, after_ph: bool = False) -> float:
        """Compute the occupation number for the given layer and site.
        # TODO: make this method also support specification of flavor, for use with non-Abelian gauge groups

        Args:
            lay (int): Layer index
            site (int): Site index
            after_ph (bool, optional): If True, compute the occupation number using the operators
                                       defined after the particle-hole transformation. Defaults to False.

        Returns:
            float: the occupation number for the given layer and site
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    def average_occupation(self, after_ph: bool = False) -> xnp.ndarray:
        """Compute the average occupation number for the system across all sites.

        Args:
            after_ph (bool, optional): If True, compute the occupation number using the
                operators as defined after the particle-hole transformation.
                Defaults to False.

        Returns:
            array: the average occupation number for the system across all sites, as a vector across layers.
        """
        total_occ = []
        for lay in range(self.cfg.num_pg_layer, self.cfg.nlayer):
            layer_val = 0.0
            for site in range(self.cfg.lattice.size):
                layer_val += self.occupation(lay, site, after_ph=after_ph)
            total_occ.append(layer_val / self.cfg.lattice.size)
        return xnp.array(total_occ)

    @property
    def all_occupations(self) -> xnp.ndarray:
        """Compute the occupation number for all layers and sites in the system.

        Returns:
            array: the occupation number for all layers and sites, as a 2D array
        """
        if self._all_occupations is None:

            # Initialize shape
            # this ensures that is has the proper shape even when there are no fermionic layers
            # (which is needed for transposes, etc. higher up in the stack)
            self._all_occupations = np.zeros((self.cfg.num_fermionic_layer, self.cfg.lattice.size))

            after_ph = False
            for lay in range(self.cfg.num_pg_layer, self.cfg.nlayer):
                for site in range(self.cfg.lattice.size):
                    self._all_occupations[lay - self.cfg.num_pg_layer, site] = self.occupation(
                        lay, site, after_ph=after_ph
                    )
        return self._all_occupations

    def meson_string(self, path) -> float:
        """Calculate the value of a meson string given a path.

        Args:
            path (list):

        Returns:
            float:
        """
        meson_val = xnp.sum(self._meson_string_vec(path))  # sum over layers/flavors
        return meson_val

    def compute_path(self, path):
        """Compute the observable corresponding the path given as an argument

        Args:
            path (list): List of tuples [(index,conj),....]. conj indicates whether the argument should be conjugated.
            This is the case if the link is traversed from right to left or from top to bottom.
        """
        path_product = self.cfg.gaugemgr.get_neutral_gauge_value()  # The identity matrix
        for ind, conj in path:
            if conj:
                path_product = path_product @ xnp.conjugate(xnp.transpose(self.gaugefieldvec[ind]))
            else:
                path_product = path_product @ self.gaugefieldvec[ind]
        return xnp.trace(path_product)

    @staticmethod
    @maybe_jit(static_argnames=["lattice_size", "nphysmodes_site", "nlayer"])
    def _compute_ferm_cov(
        lattice_size: int,
        nphysmodes_site: int,
        nlayer: int,
        wi_gamma_out_inv_vec: xnp.ndarray,
        mat_a_vec: xnp.ndarray,
        mat_b_vec: xnp.ndarray,
    ) -> xnp.ndarray:
        """Compute the covariance matrix of the fermions in the system for the given layer.
        We calculate it for all layers automatically, even though it is not needed for pure-gauge layers.
        TODO: investigate whether skipping the pure-gauge layers is worthwhile.

        Returns:
            array: array[lay] is the covmat in that layer
        """
        dim_gamma_out = 2 * lattice_size * nphysmodes_site
        shape = (nlayer, dim_gamma_out, dim_gamma_out)
        ferm_covmat_vec = xnp.full(shape, xnp.nan)

        for layer in range(nlayer):
            covmat = mat_a_vec[layer] + (
                mat_b_vec[layer] @ wi_gamma_out_inv_vec[layer] @ xnp.transpose(mat_b_vec[layer])
            )

            ferm_covmat_vec = backend.array_assign(ferm_covmat_vec, layer, covmat)
        return ferm_covmat_vec

    @property
    def ferm_covmat_vec(self) -> xnp.ndarray:
        """Compute the covariance matrix of the fermions in the system for the given layer.
        We calculate it for all layers automatically, even though it is not needed for pure-gauge layers."""
        if self._ferm_covmat_vec is None:
            wi_gamma_out_inv_vec = [self.wi_gamma_out_vec[layer].inv() for layer in range(self.cfg.nlayer)]
            self._ferm_covmat_vec = self._compute_ferm_cov(
                self.cfg.lattice.size,
                self.cfg.nphysmodes_site,
                self.cfg.nlayer,
                wi_gamma_out_inv_vec,
                self.mat_a_vec,
                self.mat_b_vec,
            )
        return self._ferm_covmat_vec

    ################## Mode Permutations ##################

    def get_link_based_mode_order(self) -> list:
        """Generate the link-based majorana mode order.

        This is a sketch of a 2x3 lattice.
        This lattice is what's used in testing this function, so check there for an explicit list of expected output.
            |         |
            "8"       "11"
            |         |
            4 --"4"-- 5 --"5"--
            |         |
            "7"       "10"
            |         |
            2 --"2"-- 3 --"3"--
            |         |
            "6"       "9"
            |         |
            0 --"0"-- 1 --"1"--

        Returns:
            list: List of strings of the form <mode_letter:majorana mode>_<copy>_<link_id>
        """

        lat = self.cfg.lattice
        num_copies = self.cfg.ncopy  # The ncopy property is defined the config of any child class of System2DBase
        mode_order = []
        num_colors = self.cfg.gaugemgr.rep_dim

        # Horizontal first
        for link in range(lat.nx * lat.ny):
            for color in range(1, num_colors + 1):
                for copy in range(1, num_copies + 1):
                    mode1 = ("l1", copy, color, link)  # majorana mode l1
                    mode2 = ("l2", copy, color, link)  # majorana mode l2
                    mode3 = ("r1", copy, color, link)
                    mode4 = ("r2", copy, color, link)
                    mode_order += [mode1, mode2, mode3, mode4]

        # Vertical
        for link in range(lat.nx * lat.ny):
            link_num = (
                link + lat.nx * lat.ny
            )  # vertical link numbers start at the number of horizontal links that there are
            for color in range(1, num_colors + 1):
                for copy in range(1, num_copies + 1):
                    mode1 = ("d1", copy, color, link_num)  # majorana mode d1
                    mode2 = ("d2", copy, color, link_num)  # majorana mode d2
                    mode3 = ("u1", copy, color, link_num)
                    mode4 = ("u2", copy, color, link_num)
                    mode_order += [mode1, mode2, mode3, mode4]

        # Convert to a list of strings
        # This was left as a tuple above in case there was ever any use for that format
        mode_order_str = []
        for mode in mode_order:
            mode_str = mode[0] + "_" + str(mode[1]) + "_" + str(mode[2]) + "_" + str(mode[3])
            mode_order_str.append(mode_str)

        return mode_order_str

    def get_site_based_mode_order(self) -> list:
        """Generate the site-based majorana mode order.

        This is a sketch of a 2x3 lattice.

            |         |
            "8"       "11"
            |         |
            4 --"4"-- 5 --"5"--
            |         |
            "7"       "10"
            |         |
            2 --"2"-- 3 --"3"--
            |         |
            "6"       "9"
            |         |
            0 --"0"-- 1 --"1"--

        On each site, the mode order is l1, l2, r1, r2, d1, d2, u1, u2 for the first copy,
        and then the same thing for the second/third/etc. copies (if they exist).

        Returns:
            list: List of strings of the form <mode_letter:majorana mode>_<copy>_<link_id>
        """

        lat = self.cfg.lattice
        num_copies = self.cfg.ncopy  # The ncopy property is defined the config of any child class of System2DBase
        num_colors = self.cfg.gaugemgr.rep_dim
        mode_order = []

        for site in range(lat.nx * lat.ny):
            for color in range(1, num_colors + 1):
                for copy in range(1, num_copies + 1):
                    x, y = lat.ind2coord(site)  # coordinates of the site

                    # Horizontal
                    mode1 = (
                        "l1",
                        copy,
                        color,
                        lat.coord2ind_dir((x - 1, y), Direction.X),
                    )
                    mode2 = (
                        "l2",
                        copy,
                        color,
                        lat.coord2ind_dir((x - 1, y), Direction.X),
                    )
                    mode3 = ("r1", copy, color, lat.coord2ind_dir((x, y), Direction.X))
                    mode4 = ("r2", copy, color, lat.coord2ind_dir((x, y), Direction.X))

                    # Vertical
                    mode5 = (
                        "d1",
                        copy,
                        color,
                        lat.coord2ind_dir((x, y - 1), Direction.Y),
                    )
                    mode6 = (
                        "d2",
                        copy,
                        color,
                        lat.coord2ind_dir((x, y - 1), Direction.Y),
                    )
                    mode7 = ("u1", copy, color, lat.coord2ind_dir((x, y), Direction.Y))
                    mode8 = ("u2", copy, color, lat.coord2ind_dir((x, y), Direction.Y))

                    mode_order += [
                        mode1,
                        mode2,
                        mode3,
                        mode4,
                        mode5,
                        mode6,
                        mode7,
                        mode8,
                    ]

        # Convert to a list of strings
        mode_order_str = []
        for mode in mode_order:
            mode_str = mode[0] + "_" + str(mode[1]) + "_" + str(mode[2]) + "_" + str(mode[3])
            mode_order_str.append(mode_str)

        return mode_order_str

    @staticmethod
    def get_single_link_majorana_mode_order(num_copies: int, num_colors: int) -> list:
        """Generate the link-based majorana mode order for a single link. We first order by color and then by copy.
        This is the actual order we use in system implementaion.

        Returns:
            list: List of strings of the form <mode_letter:majorana mode>_<copy>_<color>
        """

        mode_order = []
        # We demonstrate the order for a single horizontal link -
        for color in range(1, num_colors + 1):
            for copy in range(1, num_copies + 1):
                mode1 = ("l1", copy, color)  # majorana mode l1
                mode2 = ("l2", copy, color)  # majorana mode l2
                mode3 = ("r1", copy, color)
                mode4 = ("r2", copy, color)
                mode_order += [mode1, mode2, mode3, mode4]

        # Convert to a list of strings
        # This was left as a tuple above in case there was ever any use for that format
        mode_order_str = []
        for mode in mode_order:
            mode_str = mode[0] + "_" + str(mode[1]) + "_" + str(mode[2])
            mode_order_str.append(mode_str)

        return mode_order_str


def get_pfaffian_arrays(modes: list, coefficients: list) -> tuple:
    """Generate the arrays used for list comprehension to extract the required pfaffians, with the correct
    prefactors, used in the calculation of the electric energy and electric gradients.

    Each element in the returned list is of the form
        (k, (a_1 ... a_2p))
    where k in a prefactor, and (a_1 ... a_2p) is a tuple containing the indices to extract from the full
    covariance matrix to build a submatrix and compute the pfaffian.
    The electric energy will then be sum of these pfaffians (weighted by the prefactors), with some further
    normalization.

    Args:
        modes (list of lists of tuples of ints): _description_
        coefficients (list of lists of complex floats): _description_
        neg (float): _description_

    Returns:
        list: index array in the format required for the calculation of the electric energy (and electric gradients).
    """
    submatrices = [k for k in it.product(*modes)]
    indices = [sum(sub, ()) for sub in submatrices]

    factors = [np.asarray(k) for k in it.product(*coefficients)]
    prefactors = [np.prod(k) for k in factors]
    idxarr = [(p, i) for p, i in zip(prefactors, indices)]

    return tuple(idxarr)
