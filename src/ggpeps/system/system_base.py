import functools
from abc import ABC, abstractmethod
from typing import Optional

import logging

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
from ggpeps.system.config_base import Config2DBase, IdxGroup, CoeffsVec, ConstantsVec
from ggpeps.modearray import generate_permutation_matrix, generate_permutation
from ggpeps.system import overlap as ov

logger = logging.getLogger(ggpeps.LOGGER_NAME)


def maybe_jit(*jit_args, **jit_kwargs):
    """Apply jax.jit iff using the jax backend."""

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
        """Constructor of a 2D system, for any group, with any number of virtual or physical
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

        # Copy data from config so that it can be stored as a numpy/jax array (as appropriate)
        # TODO: this section does not need to be redone for every eval, just once during construction
        self.mask = xnp.asarray(self.cfg.mask)
        shape = (self.cfg.nlayer, len(self.cfg.mod_link_inds), self.cfg.unitcell_size, len(self.cfg.symbolvec))
        self.mod_mask_inds = xnp.nonzero(xnp.broadcast_to(self.mask[:, None, :, :], shape))  #

        # Parameter based matrices
        self._tmat_layervec_unitcellvec: Optional[list[list[xnp.ndarray]]] = None
        self._tmat_layervec_sitevec: Optional[xnp.ndarray] = None
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
        self._mat_a_mod_vec: Optional[xnp.ndarray] = None
        self._mat_b_mod_vec: Optional[xnp.ndarray] = None
        self._mat_d_mod_vec: Optional[xnp.ndarray] = None
        self._det_mat_d_mod_vec: Optional[xnp.ndarray] = None
        self._mat_d_mod_inv_vec: Optional[xnp.ndarray] = None
        self._deriv_mod_mats: Optional[tuple[xnp.ndarray, xnp.ndarray, xnp.ndarray]] = None
        self._dmatd_trace_sparse: Optional[tuple[xnp.ndarray, xnp.ndarray, xnp.ndarray]] = None
        # Block-structured norm-gradient helpers (see compute_grad_over_norm_vec).
        self._grad_norm_block_maps: Optional[tuple[xnp.ndarray, xnp.ndarray, xnp.ndarray]] = None
        self._grad_norm_site_deriv_blocks: Optional[xnp.ndarray] = None
        # Electric energy intermediate values - if we compute the electric energy,
        # we store intermediate values to be reused in the gradient calculation
        self._covmat_out_mod_vec: Optional[xnp.ndarray] = None
        self._norm_mod_vec: Optional[xnp.ndarray] = None
        self._el_pfaffians: Optional[xnp.ndarray] = None
        self._lognormvec_default: Optional[xnp.ndarray] = None

        # Management of the gauge fields
        # vec for layers (choices of projectors may be different for each layer), dirs for directions
        self._gamma_gauge_neutral_vec_dirs: Optional[xnp.ndarray] = None

        # In cases when different layers use the same projectors, all elements will point to the same gamma_in_sys:
        self._gamma_in_sys_vec: Optional[xnp.ndarray] = None
        # Cached open-link ("mod") extraction of gamma_in_sys.
        self._gamma_in_sys_mod_vec: Optional[xnp.ndarray] = None

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
        self._occupations_before_ph: Optional[xnp.ndarray] = None
        self._occupations_after_ph: Optional[xnp.ndarray] = None

        # Woodbury Update and Matrix Inversion
        self._wi_gamma_in_vec: Optional[xnp.ndarray] = None  # Tracks (D^-1 - gammain)^-1
        self._wi_gamma_out_vec: Optional[xnp.ndarray] = None  # Tracks (D + gammain)^-1
        self._incdet_vec: Optional[xnp.ndarray] = None  # Tracks det(D^-1 - gammain)

        self._wi_gamma_in_mod_vec: Optional[xnp.ndarray] = None  # Tracks (Dmod^-1 - gammain)^-1
        self._wi_gamma_out_mod_vec: Optional[xnp.ndarray] = None  # Tracks (Dmod + gammain)^-1
        self._incdet_mod_vec: Optional[xnp.ndarray] = None  # Tracks det(Dmod^-1 - gammain)

        # Drift control for the incremental (Woodbury/IncDet) trackers.
        # The closed and modified trackers are maintained incrementally for cheaper single-link
        # updates, but the per-step error accumulates and is amplified through near-singular updates.
        # To keep it bounded, the public update_gauge_ind wrapper recomputes BOTH families from
        # scratch (a) periodically every ``tracker_refresh_interval`` gauge steps and (b) whenever
        # the last incremental step left a near-singular tracked inverse (its largest entry exceeds
        # TRACKER_INV_MAG_THRESH).
        self._steps_since_refresh: int = 0
        self._last_step_max_inv_mag: float = 0.0
        self.tracker_refresh_interval: int = getattr(self.cfg, "tracker_refresh_interval", 0)

        # When True (set by the evaluator around the warmup loop), _update_gauge_ind skips maintaining the
        # open-link ("mod") family (gamma_in_sys_mod + wi_gamma_*_mod + incdet_mod), which is consumed only
        # by measurements.
        self.defer_mod_trackers: bool = False

        return

    # Magnitude guard: recompute immediately if the largest entry of a tracked inverse exceeds this
    # threshold. A large inverse <=> a near-singular tracked matrix, where the incremental Woodbury
    # update loses precision; the from-scratch recompute is backward-stable and matches even
    # where the inverse is genuinely large.
    TRACKER_INV_MAG_THRESH: float = 1e2

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
        self._occupations_before_ph = None
        self._occupations_after_ph = None

        self._el_energy_op_grad_vec = None
        self._mass_energy_op_grad_vec = None
        self._int_energy_op_grad_vec = None
        self._chem_energy_op_grad_vec = None
        self._grad_over_norm_vec = None

        self._covmat_out_mod_vec = None
        self._norm_mod_vec = None
        self._el_pfaffians = None
        self._lognormvec_default = None
        return

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

    def _eval_tmat_symb(self, paramvec: np.ndarray) -> xnp.ndarray:
        """Compute the numerical representation of the T matrix

        Args:
            paramvec (np.ndarray): array of parameter values (numerical)

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
    def tmat_layervec_sitevec(self) -> xnp.ndarray:
        """
        Generate the T-matrix vector (single virtual fermion on the link).
        Analytically, this mode order is not advantageous,
        but it makes the reshuffling of the modes easier for gamma_in and M_D in the covariance matrix.

        Returns:
            xnp.ndarray: an array of parameter matrices T, of shape (nlayer, nsites, dim_tmat, dim_tmat)
        """
        if self._tmat_layervec_sitevec is None:
            self.cfg.enforce_parameter_conditions(self.cfg.paramvec)
            tmat_layervec_sitevec = []
            for layer in range(self.cfg.nlayer):
                tmats = self.tmat_layervec_unitcellvec[layer]
                tmat_lay = [tmats[self.cfg.site_params_dict[site]] for site in range(self.cfg.lattice.size)]
                tmat_layervec_sitevec.append(tmat_lay)
            self._tmat_layervec_sitevec = xnp.array(tmat_layervec_sitevec)
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

                gamma_dirac_lay = [utils.tmat_to_covariance_matrix(tmat) for tmat in self.tmat_layervec_sitevec[lay]]
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
            # Note: since self.gamma_dirac_layervec_sitevec is already a vector over sites, here we are being
            #       slightly inneficent - we do the matrix multiplication for each entry, even though many of
            #       the gamma_dirac's are the same. However, this happens only once per evaluation,
            #       so it is ultimately negligible.
            self._gamma_maj_layervec_sitevec = xnp.real(smat @ self.gamma_dirac_layervec_sitevec @ xnp.transpose(smat))
        return self._gamma_maj_layervec_sitevec

    def _expand_gamma_maj_to_system(self, covmats_layervec_sitevec: xnp.ndarray) -> xnp.ndarray:
        """Expand the covariance matrix in Majorana modes to the full system.
        In order to obtain a structure that is convenient for further computations,
            (A    B)
            (-B^T D)
        we have to reorder the modes of the single-vertex matrix with respect to the full matrix.

        Args:
            covmats_layervec_sitevec (xnp.ndarray):
                array (per layer) of 2D covariance matrices of all sites; total shape (nlayer, nsites, nmodes, nmodes)

        Returns:
            xnp.ndarray: 2D covariance matrix of the full system
        """

        # Build permutation to convert modes from site order to link order;
        # be careful with the convention of the permutation vs its transpose; this way works with the code below.
        # We do this without explicitely building a permutation matrix, instead doing indexing, which is more efficient
        modes_link_order = self.get_link_based_mode_order()
        modes_site_order = self.get_site_based_mode_order()

        # total number of physical fermionic majorana modes on all the sites together
        num_phys = 2 * self.cfg.lattice.nx * self.cfg.lattice.ny * self.cfg.nphysmodes_site

        # Get permutation
        perm = generate_permutation(modes_link_order, modes_site_order)
        full_perm = xnp.concatenate([xnp.arange(num_phys), perm + num_phys])

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
            dest = mat_sys_unordered[full_perm[:, None], full_perm]
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
            "zeroed_params",
        ]
    )
    def _compute_d_gamma_out_symbolvec(
        lattice_size: int,
        nphysmodes_site: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        unitcell_size: int,
        nparams: int,
        zeroed_params: tuple,
        mat_b_vec: xnp.ndarray,
        gamma_out_inv_vec: xnp.ndarray,
        gamma_maj_sys_deriv_layvec_ucvec_symbvec: xnp.ndarray,
    ) -> xnp.ndarray:
        nlayer = num_pg_layer + num_fermionic_layer
        dim_gamma_out = 2 * lattice_size * nphysmodes_site
        shape = (nlayer, unitcell_size, nparams, dim_gamma_out, dim_gamma_out)
        d_gamma_out_symbolvec = xnp.zeros(shape)

        for layer in range(num_pg_layer, nlayer):
            mat_b = mat_b_vec[layer]
            diff_d_gamma_inv = gamma_out_inv_vec[layer]

            # Precompute terms to save time in the inner loops
            diff_times_b = diff_d_gamma_inv @ xnp.swapaxes(mat_b, -2, -1)
            b_times_diff = mat_b @ diff_d_gamma_inv

            for uc_ind in range(unitcell_size):
                for symbol_ind in range(nparams):
                    if (layer, uc_ind, symbol_ind) not in zeroed_params:

                        deriv_gamma_maj_sys = gamma_maj_sys_deriv_layvec_ucvec_symbvec[layer, uc_ind, symbol_ind]
                        d_mat_a, d_mat_b, d_mat_d = utils.extract_partial_covmats(deriv_gamma_maj_sys, dim_gamma_out)

                        d_gamma_out = (
                            d_mat_a
                            + d_mat_b @ diff_times_b
                            + b_times_diff @ xnp.swapaxes(d_mat_b, -2, -1)
                            - b_times_diff @ d_mat_d @ diff_times_b
                        )

                        d_gamma_out_symbolvec = backend.array_assign(
                            d_gamma_out_symbolvec, (layer, uc_ind, symbol_ind), d_gamma_out
                        )

        return d_gamma_out_symbolvec

    @property
    def d_gamma_out_symbolvec(self) -> xnp.ndarray:
        """Return a vector containing the derivatives of gamma_out for each symbol.

        Returns:
            array: d_gamma_out_symbolvec[layer, uc_ind, symbol] is the matrix of derivatives of gamma_out[lay]
                   wrt the parameter at that (lay, uc_ind, symbol)
        """
        if self._d_gamma_out_symbolvec is None:
            self._d_gamma_out_symbolvec = self._compute_d_gamma_out_symbolvec(
                self.cfg.lattice.size,
                self.cfg.nphysmodes_site,
                self.cfg.num_pg_layer,
                self.cfg.num_fermionic_layer,
                self.cfg.unitcell_size,
                len(self.cfg.symbolvec),
                self.cfg.zeroed_params,
                self.mat_b_vec,
                self.wi_gamma_out_vec,
                self.gamma_maj_sys_deriv_layvec_ucvec_symbvec,
            )
        return self._d_gamma_out_symbolvec

    @property
    def gamma_maj_sys_vec(self) -> xnp.ndarray:
        """Return the covariance matrix of the full system in Majorana modes.
        The mode order is changed to fit the mode order of gamma_in.
        See documentation of gamma_in for details.

        This is a get function.

        Returns:
            xnp.ndarray: Covariance matrix of the full system
        """
        if self._gamma_maj_sys_vec is None:
            self._gamma_maj_sys_vec = self._expand_gamma_maj_to_system(self.gamma_maj_layervec_sitevec)
        return self._gamma_maj_sys_vec

    @property
    def mat_a_vec(self) -> xnp.ndarray:
        """Extract the matrix for physical-physical correlations.
        The mode ordering of this matrix is (p_1(0,0),p_2(0,0),p_1(1,0),p_2(1,0)....).
        It is formulated in terms of Majorana modes.
        The mode ordering of the sites is identical to the site convention defined in the lattice class.
        This is a get function.

        This is an array of A matrices - the output shape is (nlayers, dim, dim), where
        dim = 2 (for majorana) * [ nsites * nphysmodespersite (# phys modes) ]

        Returns:
            xnp.ndarray: Correlations of the physcial modes for the full system.
        """
        if self._mat_a_vec is None:
            offset = 2 * self.cfg.lattice.size * self.cfg.nphysmodes_site
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = utils.extract_partial_covmats(
                self.gamma_maj_sys_vec, offset
            )
        return self._mat_a_vec

    @property
    def mat_b_vec(self) -> xnp.ndarray:
        """Extract the matrix for physical-virtual correlations.
        This is a get function.

        This is an array of B matrices - the output shape is (nlayers, dim1, dim2), where
        dim1 = 2 (for majorana) * [ nsites * nphysmodespersite (# phys modes) ]
        dim2 = 2 (for majorana) * [ nlinks * 2 * ncopy (virt modes/link) ]

        Returns:
            xnp.ndarray: Correlations of the physcial modes with the virtual modes for the full system.
        """
        if self._mat_b_vec is None:
            offset = 2 * self.cfg.lattice.size * self.cfg.nphysmodes_site
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = utils.extract_partial_covmats(
                self.gamma_maj_sys_vec, offset
            )
        return self._mat_b_vec

    @property
    def mat_d_vec(self) -> xnp.ndarray:
        """Extract the matrix for virtual-virtual correlations (aka D).
        This is a get function.

        This is an array of D matrices - the output shape is (nlayers, dim, dim), where
        dim = 2 (for majorana) * [ nlinks * 2 * ncopy (virt modes/link) ]

        Returns:
            xnp.ndarray: Correlations of the virtual modes for the full system.
        """
        if self._mat_d_vec is None:
            offset = 2 * self.cfg.lattice.size * self.cfg.nphysmodes_site
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = utils.extract_partial_covmats(
                self.gamma_maj_sys_vec, offset
            )
        return self._mat_d_vec

    @property
    def det_mat_d_vec(self) -> xnp.ndarray:
        """Compute the determinant of the virtual-virtual correlation matrix.
        This is a get function.

        This is an array of D matrix logdets - the output shape is (nlayers,).

        Returns:
            xnp.ndarray: array of log-determinants
        """
        if self._det_mat_d_vec is None:
            _, self._det_mat_d_vec = xnp.linalg.slogdet(self.mat_d_vec)
        return self._det_mat_d_vec

    @property
    def mat_d_inv_vec(self) -> xnp.ndarray:
        """Compute the vector of the inverses of the D matrix.
        The D matrix is the correlation matrix of virtual-virtual correlations.
        This is a get function.

        This is a vector of D matrix inverses - the output shape is (nlayers, dim, dim), where
        dim = 2 (for majorana) * [ nlinks * 2 * ncopy (virt modes/link) ]

        Returns:
            xnp.ndarray: array of matrix inverses of the D matrix
        """
        if self._mat_d_inv_vec is None:
            self._mat_d_inv_vec = xnp.linalg.inv(self.mat_d_vec)
        return self._mat_d_inv_vec

    @property
    def mat_a_mod_vec(self) -> xnp.ndarray:
        """Extract the matrix for physical-physical correlations and one virtual mode.
        This shifted matrix is used for the computation of the electric energy.
        The mode ordering of this matrix is (p_1(0,0),p_2(0,0),p_1(1,0),p_2(1,0)....).
        It is formulated in terms of Majorana modes.
        The mode ordering of the sites is identical to the site convention defined in the lattice class.
        "Physical" modes here includes the virtual modes on the link on which the electric energy is computed.
        This is a get function.

        This is a vector of A matrices - the output shape is (nlayers, num mod links, dim, dim), where
        dim = 2 (for majorana) * [ nsites * nphysmodespersite (# phys modes) + 2 * ncopy (virt modes/link) ]

        Returns:
            [xnp.ndarray]: Correlations of the physical modes for the full system.
        """
        if self._mat_a_mod_vec is None:
            self._mat_a_mod_vec, self._mat_b_mod_vec, self._mat_d_mod_vec = utils.extract_mod_covmats(
                self.gamma_maj_sys_vec,
                self.cfg.mod_link_inds,
                self.cfg.lattice.size,
                self.cfg.nphysmodes_site,
                self.cfg.nvirtmodes_link,
            )
        return self._mat_a_mod_vec

    @property
    def mat_b_mod_vec(self) -> xnp.ndarray:
        """Extract the matrix for physical-virtual correlations.
        This matrix contains one link less than the original matrix (used for the electric energy computation).
        "Physical" modes here includes the virtual modes on the link on which the electric energy is computed.
        This is a get function.

        This is a vector of B matrices - the output shape is (nlayers, num mod links, dim1, dim2), where
        dim1 = 2 (for majorana) * [ nsites * nphysmodespersite (# phys modes) + 2 * ncopy (virt modes/link) ]
        dim2 = 2 (for majorana) * [ (nlinks - 1) * 2 * ncopy (virt modes/link) ]

        Returns:
            [xnp.ndarray]: Correlations of the physical modes with the virtual modes for the full system.
        """
        if self._mat_b_mod_vec is None:
            self._mat_a_mod_vec, self._mat_b_mod_vec, self._mat_d_mod_vec = utils.extract_mod_covmats(
                self.gamma_maj_sys_vec,
                self.cfg.mod_link_inds,
                self.cfg.lattice.size,
                self.cfg.nphysmodes_site,
                self.cfg.nvirtmodes_link,
            )
        return self._mat_b_mod_vec

    @property
    def mat_d_mod_vec(self) -> xnp.ndarray:
        """Extract the matrix for virtual-virtual correlations.
        This matrix contains one link less than the original matrix (used for the electric energy computation).
        This is a get function.

        This is a vector of D matrices - the output shape is (nlayers, num mod links, dim, dim), where
        dim = 2 (for majorana) * [ (nlinks - 1) * 2 * ncopy (virt modes/link) ]

        Returns:
            [xnp.ndarray]: Correlations of the virtual modes for the full system.
        """
        if self._mat_d_mod_vec is None:
            self._mat_a_mod_vec, self._mat_b_mod_vec, self._mat_d_mod_vec = utils.extract_mod_covmats(
                self.gamma_maj_sys_vec,
                self.cfg.mod_link_inds,
                self.cfg.lattice.size,
                self.cfg.nphysmodes_site,
                self.cfg.nvirtmodes_link,
            )
        return self._mat_d_mod_vec

    @property
    def det_mat_d_mod_vec(self) -> xnp.ndarray:
        """
        Compute the determinant of the virtual-virtual correlation matrix for the modified matrix.
        There is a vector of determinants of D matrices - the output shape is (nlayers, num mod links).
        This is a get function.

        Returns:
            list: array of log-determinants
        """
        if self._det_mat_d_mod_vec is None:
            _, self._det_mat_d_mod_vec = xnp.linalg.slogdet(self.mat_d_mod_vec)
        return self._det_mat_d_mod_vec

    @property
    def mat_d_mod_inv_vec(self) -> xnp.ndarray:
        """Compute the inverse of modified D matrices.
        There is a vector of inverses of D matrices - the output shape is (nlayers, num mod links, dim, dim), where
        dim = 2 (for majorana) * [ (nlinks - 1) * 2 * ncopy (virt modes/link) ]
        This is a get function.

        Returns:
            xnp.ndarray: array of inverses of modified D matrices
        """
        if self._mat_d_mod_inv_vec is None:
            self._mat_d_mod_inv_vec = xnp.linalg.inv(self.mat_d_mod_vec)
        return self._mat_d_mod_inv_vec

    @property
    def covmat_out_mod_vec(self) -> xnp.ndarray:
        """Compute the convariance matrix of the state, including physical fermions and
        the virtual fermions on the link on which the electric energy is computed.
        This function returns a vector over layers of these covariance matrices.
        This is a get function.

        Returns:
            xnp.ndarray: a vector of covariance matrices
        """

        if self._covmat_out_mod_vec is None:
            covmat_out_mod_vec = []

            # Number of fermions = # of sites
            # Since we have 2 copies, we get 8 virtual fermions per site
            single_link_offset = 2 * self.cfg.nvirtmodes_link

            # TODO: vectorize!
            for layerind in range(self.cfg.nlayer):
                covmat_out_linkvec = []

                for ind, link_ind in enumerate(self.cfg.mod_link_inds):
                    group_element = self._gaugefieldvec[link_ind]
                    coord, dir = self.cfg.lattice.ind2coord_dir(link_ind)
                    R = self.generate_rotmat(self.cfg.ncopy, group_element, coord, dir)
                    R_T = xnp.transpose(R)
                    # Get the modified matrices, which include the virtual modes of the given link among the physical
                    mat_a = self.mat_a_mod_vec[layerind, ind]
                    mat_b = self.mat_b_mod_vec[layerind, ind]

                    mat_a_mod_block = mat_a[
                        -single_link_offset:, -single_link_offset:
                    ]  # modified-modified part of the A matrix
                    mat_b_mod_block = mat_b[-single_link_offset:, :]
                    # modified-virtual part of the B matrix

                    diff_d_gamma_inv = self.wi_gamma_out_mod_vec[layerind][ind]

                    # Compute covmat
                    covmat_out_virt_ungauged = mat_a_mod_block + mat_b_mod_block @ diff_d_gamma_inv @ xnp.transpose(
                        mat_b_mod_block
                    )

                    covmat_out_virt = R_T @ covmat_out_virt_ungauged @ R

                    # pfapack (used in the el energy) is rather picky about the anti-symmetrization (to 1e-14)
                    covmat_out_virt = utils.anti_symmetrize(covmat_out_virt)
                    covmat_out_linkvec.append(covmat_out_virt)
                covmat_out_mod_vec.append(covmat_out_linkvec)

            self._covmat_out_mod_vec = xnp.asarray(covmat_out_mod_vec)
        return self._covmat_out_mod_vec

    @property
    def el_pfaffians(self) -> xnp.ndarray:
        """Compute the pfaffians of the modified covariance matrices used for the electric energy and its derivative.
        This function returns a vector over layers and links of these pfaffians.
        This is a get function.

        Returns:
            xnp.array: a vector of pfaffians
        """
        if self._el_pfaffians is None:
            self._el_pfaffians = self._compute_el_pfaffians(
                self.cfg.nlayer,
                self.cfg.mod_link_inds,
                self.covmat_out_mod_vec,
                self.cfg.uniq_idx_vec,
            )
        return self._el_pfaffians

    @staticmethod
    @maybe_jit(static_argnames=["nlayer", "idxarr_vec", "mod_link_inds"])
    def _compute_el_pfaffians(
        nlayer: int,
        mod_link_inds: tuple[int, ...],
        covmat_out_virt_vec: xnp.ndarray,
        idxarr_vec: IdxGroup,
    ) -> xnp.ndarray:
        """Compute the pfaffians of the modified covariance matrices used for the electric energy and its derivative.
        This function returns a vector over layers and links of these pfaffians.
        The pfaffians are not multiplied by any prefactors.

        idxarr_vec is the unique index structure (cfg.uniq_idx_vec): the Pfaffian of a stored
        monomial depends only on (layer, link, index tuple), so each unique tuple is computed once
        and the per-group-element coefficients are applied by the el_energy and el_energy_grad functions
        (via cfg.uniq_coeffs_vec).

        Returns:
            xnp.ndarray: pfaffians, shape (nlayer, num_el_links, num_size_classes, num_terms_max)
        """

        num_el_links = len(mod_link_inds)  # number of links to calculate the electric energy on.
        num_terms_per_size = max(len(size) for layer in idxarr_vec for link in layer for size in link)
        # Max terms per size class; size classes may vary in term count.
        num_lens = max(len(link) for layer in idxarr_vec for link in layer)
        el_pfaffians = xnp.zeros((nlayer, num_el_links, num_lens, num_terms_per_size))

        for layerind in range(nlayer):
            layer_idxs = idxarr_vec[layerind]  # tuple of tuples of indices

            covmat_out_virt_linkvec = covmat_out_virt_vec[layerind]

            # Iterate over the links
            for link_pos, covmat_out_virt in enumerate(covmat_out_virt_linkvec):
                link_idxs = layer_idxs[link_pos]

                # Go over the different lengthed tuples of tuples of indices
                for size_idxs_ind, size_idxs in enumerate(link_idxs):
                    inds_batch = xnp.asarray(size_idxs)
                    rows = inds_batch[:, :, None]
                    cols = inds_batch[:, None, :]
                    submatrices = covmat_out_virt[rows, cols]  # shape (num_terms, size, size)

                    pfavals = backend.pfaffian_vectorized(submatrices)

                    el_pfaffians = backend.array_assign(
                        el_pfaffians,
                        (layerind, link_pos, size_idxs_ind, slice(0, len(pfavals))),
                        pfavals,
                    )

        return el_pfaffians

    @property
    def norm_mod_vec(self) -> xnp.ndarray:
        """Compute the normalization factor for the modified covariance matrix.
        This is used to compute the electric energy.
        This function returns a vector over layers of these normalization factors.
        This is a get function.

        Returns:
            list[float]: a vector of modified normalization factors
        """
        if self._norm_mod_vec is None:
            norm_mod_layervec_linkvec = []
            lognormvec_default = self.lognormvec_default

            for layerind in range(self.cfg.nlayer):

                norm_mod_linkvec = []

                # TODO: this is not yet actually vectorized
                for ind, link_ind in enumerate(self.cfg.mod_link_inds):

                    # For the modified norm, we still have to take into account the other contributions
                    # from the unmodified parts
                    norm_mod_vec = self._calculate_lognormvec_inc(
                        xnp.expand_dims(self.incdet_mod_vec[layerind, ind], axis=0),
                        xnp.expand_dims(self.det_mat_d_mod_vec[layerind, ind], axis=0),
                        self.gamma_in_sys_mod_vec[layerind, ind].shape[0],
                        all_factors=True,
                    )
                    norm_mod = xnp.sum(norm_mod_vec)

                    norm_mod += utils.add_except(lognormvec_default, layerind)
                    norm_mod_linkvec.append(norm_mod)
                norm_mod_layervec_linkvec.append(norm_mod_linkvec)

            self._norm_mod_vec = xnp.asarray(norm_mod_layervec_linkvec)
        return self._norm_mod_vec

    @property
    def lognormvec_default(self) -> xnp.ndarray:
        """Compute the log of the norm of the state, with no modifications (e.g. those of the electric energy),
        including all factors.
        This function returns a vector over layers of these norms.
        This is a get function.

        Returns:
            array: a vector of norms
        """
        if self._lognormvec_default is None:
            self._lognormvec_default = self.calculate_lognormvec(incremental=True, all_factors=True)
        return self._lognormvec_default

    def initialize_gamma_in_and_trackers(self) -> tuple:
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

        # Initialize gamma_in_sys for the full system (and trackers)
        size = self.cfg.lattice.size  # number of sites
        id = xnp.eye(size)

        # TODO: vectorize!
        gamma_in_sys_listvec = []
        for layer in range(self.cfg.nlayer):
            neutral_gauge_X = xnp.kron(id, self.gamma_gauge_neutral_vec[layer][Direction.X])
            neutral_gauge_Y = xnp.kron(id, self.gamma_gauge_neutral_vec[layer][Direction.Y])
            gamma_in_sys = xscipy.linalg.block_diag(neutral_gauge_X, neutral_gauge_Y)
            gamma_in_sys_listvec.append(gamma_in_sys)

        gamma_in_sys_vec = xnp.array(gamma_in_sys_listvec)
        # Set gamma_in_sys_vec now so the from-scratch tracker helpers below can read it via the
        # property without re-triggering initialization.
        self._gamma_in_sys_vec = gamma_in_sys_vec

        # Build both tracker families from scratch and reset the drift-control counter
        closed_trackers = self._compute_closed_trackers()
        mod_trackers = self._compute_mod_trackers()
        self._steps_since_refresh = 0

        return (gamma_in_sys_vec, closed_trackers, mod_trackers)

    def _compute_closed_trackers(self) -> tuple:
        """Recompute the closed (full-system) Woodbury inverses and incremental determinant from
        scratch from the CURRENT ``gamma_in_sys_vec``.

        Returns:
            tuple: ``(wi_gamma_in_vec, wi_gamma_out_vec, incdet_vec)`` -- the closed Woodbury
                inverses of ``mat_d_inv - gamma_in`` and ``mat_d + gamma_in`` and the log-abs
                determinant of ``mat_d_inv - gamma_in``, all for the current gauge field.
        """
        gamma_in_sys_vec = self.gamma_in_sys_vec
        wi_gamma_in_vec = xnp.linalg.inv(self.mat_d_inv_vec - gamma_in_sys_vec)
        wi_gamma_out_vec = xnp.linalg.inv(self.mat_d_vec + gamma_in_sys_vec)
        _, incdet_vec = xnp.linalg.slogdet(self.mat_d_inv_vec - gamma_in_sys_vec)
        return wi_gamma_in_vec, wi_gamma_out_vec, incdet_vec

    @property
    def gamma_in_sys_vec(self) -> xnp.ndarray:
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
    def incdet_vec(self) -> xnp.ndarray:
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
    def wi_gamma_in_vec(self) -> xnp.ndarray:
        """Return the vector of Woodbury inverters for (D^-1 - gammain)^-1 for the different layers.
        The length of the first axis is equal to the number of layers.
        This is a get function.

        Returns:
            xnp.ndarray: Array of Woodbury inverters
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
    def wi_gamma_out_vec(self) -> xnp.ndarray:
        """Return the vector of Woodbury inverters for (D + gammain)^-1 for the different layers.
        The length of the first axis is equal to the number of layers.
        This is a get function.

        Returns:
            xnp.ndarray: Array of Woodbury inverters
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

    @property
    def gamma_in_sys_mod_vec(self) -> xnp.ndarray:
        """Get function to return the gauged gamma_in_sys_vec with a single link modification
        (to compute the electric energy), the covariance matrix of the links for the whole system.

        Returns:
            xnp.ndarray: Gauged, modified covariance matrices of the system for each layer
        """
        if self._gamma_in_sys_mod_vec is None:
            self._gamma_in_sys_mod_vec = self._extract_gamma_in_sys_mod_vec(
                self.cfg.mod_link_inds, self.gamma_in_sys_vec
            )
        return self._gamma_in_sys_mod_vec

    def _extract_gamma_in_sys_mod_vec(self, link_inds: tuple[int, ...], gamma_in_sys: xnp.ndarray) -> xnp.ndarray:
        """Get function to return the gauged gamma_in_sys_vec with a single link modification
        (to compute the electric energy), the covariance matrix of the links for the whole system.
        This function can accept a 2D matrix, or a stack of 2D matrices (i.e. a 3D array).

        Args:
            link_ind (tuple[int, ...]): indices of the links to exclude
                                        (returned array will have a second axis of this length)
            gamma_in_sys (xnp.ndarray): Covariance matrix of the links for the whole system, or stack of such matrices.

        Returns:
            xnp.ndarray: Gauged, modified covariance matrices of the system for each layer and link
        """

        res = utils.extract_mod_covmats(
            gamma_in_sys,
            link_inds=link_inds,
            lattice_size=self.cfg.lattice.size,
            nphysmodes_site=0,  # gamma_in_sys does not know about physical modes, so they should not be accounted for
            nvirtmodes_link=self.cfg.nvirtmodes_link,
        )
        gamma_in_sys_mod_linkvec_layervec = res[2]  # extract the "virtual-virtual" part
        return gamma_in_sys_mod_linkvec_layervec

    def _compute_mod_trackers(self) -> tuple:
        """Recompute the modified (open-link) Woodbury inverses and incremental determinant
        from scratch from the CURRENT ``gamma_in_sys_vec``.

        This is the from-scratch recomputation point for the modified trackers. The modified objects
        ARE tracked incrementally (see the subclasses' ``_update_gauge_ind``) for a cheaper
        single-link update, but their incremental Woodbury/IncDet update can drift
        during an eval. To bound that drift ``refresh_trackers`` recomputes from scratch
        here -- periodically and on the magnitude guard (see ``update_gauge_ind``).
        ``gamma_in_sys_vec`` itself is maintained exactly, so recomputing the modified objects from
        it is correct and numerically stable.

        Returns:
            tuple: ``(wi_gamma_in_mod, wi_gamma_out_mod, incdet_mod)`` -- the open-link Woodbury
                inverses of ``mat_d_mod_inv - gamma_in_mod`` and ``mat_d_mod + gamma_in_mod`` and the
                log-abs determinant of ``mat_d_mod_inv - gamma_in_mod``, stacked over (layer, link).
        """
        gamma_in_sys_mod = self.gamma_in_sys_mod_vec
        diff_in = self.mat_d_mod_inv_vec - gamma_in_sys_mod
        wi_gamma_in_mod = xnp.linalg.inv(diff_in)
        wi_gamma_out_mod = xnp.linalg.inv(self.mat_d_mod_vec + gamma_in_sys_mod)
        _, incdet_mod = xnp.linalg.slogdet(diff_in)
        return wi_gamma_in_mod, wi_gamma_out_mod, incdet_mod

    # ---------------- Incremental tracker update + drift control ----------------

    def refresh_trackers(self) -> None:
        """Recompute BOTH the closed and modified incremental trackers from scratch.

        The closed (`wi_gamma_in/out_vec`, `incdet_vec`) and modified (`wi_gamma_*_mod_vec`,
        `incdet_mod_vec`) trackers are maintained incrementally by ``_update_gauge_ind`` for the
        cheaper single-link update, but the per-step error accumulates along a long update chain. This
        function recomputes both families from the CURRENT ``gamma_in_sys_vec`` (which is maintained exactly),
        bounding that drift. Called from the public
        ``update_gauge_ind`` periodically and on the magnitude guard; resets the step counter.

        Returns:
            None
        """
        self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = self._compute_closed_trackers()
        self.weight = 0.5 * np.sum(self._incdet_vec)
        # During warmup the mod family is deferred (see defer_mod_trackers). Don't recompute it here
        # either - it is rebuilt once when warmup ends (recompute_mod_trackers).
        if not self.defer_mod_trackers:
            (
                self._wi_gamma_in_mod_vec,
                self._wi_gamma_out_mod_vec,
                self._incdet_mod_vec,
            ) = self._compute_mod_trackers()
        self._steps_since_refresh = 0

    def recompute_mod_trackers(self) -> None:
        """Recompute the open-link ("mod") family from scratch from the CURRENT gamma_in_sys.

        Called by the evaluator when it clears ``defer_mod_trackers`` at the end of warmup: during warmup
        the mod trackers were not maintained, so rebuild them (and re-extract gamma_in_sys_mod) fresh from
        the exactly-maintained gamma_in_sys before the first measurement.
        """
        self._gamma_in_sys_mod_vec = None  # force a fresh extract from the current gamma_in_sys
        (
            self._wi_gamma_in_mod_vec,
            self._wi_gamma_out_mod_vec,
            self._incdet_mod_vec,
        ) = self._compute_mod_trackers()

    @property
    def incdet_mod_vec(self) -> xnp.ndarray:
        """Return the vector of incremental determinants for the modified matrices for the different layers.
        This is a get function.

        Returns:
            xnp.ndarray: Array of incremental determinant trackers, over layers and links
        """
        if self._incdet_mod_vec is None:
            (
                self._wi_gamma_in_mod_vec,
                self._wi_gamma_out_mod_vec,
                self._incdet_mod_vec,
            ) = self._compute_mod_trackers()
        return self._incdet_mod_vec

    @property
    def wi_gamma_in_mod_vec(self) -> xnp.ndarray:
        """Return the vector of Woodbury inverters for (D^-1 - gammain_mod)^-1 for the different layers.
        This is a get function.

        Returns:
            xnp.ndarray: Array of Woodbury inverters, over layers and links
        """
        if self._wi_gamma_in_mod_vec is None:
            (
                self._wi_gamma_in_mod_vec,
                self._wi_gamma_out_mod_vec,
                self._incdet_mod_vec,
            ) = self._compute_mod_trackers()
        return self._wi_gamma_in_mod_vec

    @property
    def wi_gamma_out_mod_vec(self) -> xnp.ndarray:
        """Return the vector of Woodbury inverters for (D + gammain_mod)^-1 for the different layers.
        This is a get function.

        Returns:
            xnp.ndarray: Array of Woodbury inverters, over layers and links
        """
        if self._wi_gamma_out_mod_vec is None:
            (
                self._wi_gamma_in_mod_vec,
                self._wi_gamma_out_mod_vec,
                self._incdet_mod_vec,
            ) = self._compute_mod_trackers()
        return self._wi_gamma_out_mod_vec

    ################## Computation of derivatives ######################

    def compute_gamma_dirac_deriv(self, symb: sympy.Symbol, layerind: int, uc_ind: int) -> xnp.ndarray:
        """Return the numerical derivative of the gamma_dirac, the Dirac covariance matrix of one fiducial state.

        Args:
            symb (sympy.Symbol): Symbol with respect to which we derive
            layerind (int): index of the layer
            uc_ind (int): index of the unit cell

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

    def compute_gamma_maj_deriv(self, symb: sympy.Symbol, layerind: int, uc_ind: int) -> xnp.ndarray:
        """Return the numerical derivative of the gamma_maj, the Majorana covariance matrix of one fiducial state.

        Args:
            symb (sympy.Symbol): Symbol with respect to which we derive
            layerind (int): index of the layer
            uc_ind (int): index of the unit cell

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

                    # _expand_gamma_maj_to_system() expects a vector over layers, so we include an extra initial axis
                    # and then extract the first element of the output
                    gamma_maj_derivs_sitevec = xnp.zeros((1, self.cfg.lattice.size, *gamma_maj_deriv.shape))
                    for site in range(self.cfg.lattice.size):
                        if self.cfg.site_params_dict[site] == uc_ind:
                            gamma_maj_derivs_sitevec = backend.array_assign(
                                gamma_maj_derivs_sitevec, (0, site), gamma_maj_deriv
                            )
                    gamma_maj_sys_derivs = self._expand_gamma_maj_to_system(gamma_maj_derivs_sitevec)[0]

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

    @property
    def grad_norm_block_maps(self) -> tuple[xnp.ndarray, xnp.ndarray, xnp.ndarray]:
        """Index/mask helpers for the block-structured norm gradient (geometry-only, gauge/param-independent).

        Returns ``(virt_inds, uc_mask, nonzero_mask)``:
          - ``virt_inds`` (nsites, modes_per_site) int: for each site, the row/column indices of that site's
            virtual modes inside the system covariance matrices, ordered to match the site-local mode
            ordering of ``compute_gamma_maj_deriv`` -- so ``prod[virt_inds[s]][:, virt_inds[s]]`` is site s's
            block, aligned entry-for-entry with the site-level derivative block. Derivation below.
          - ``uc_mask`` (unitcell_size, nsites) float: 1 where a site belongs to a unit-cell group, else 0.
          - ``nonzero_mask`` (nlayer, unitcell_size, nparams) float: 0 for symmetry-zeroed parameters, else 1.

        Deriving ``virt_inds``: the system covariance orders virtual modes by link, whereas the per-site
        blocks are naturally contiguous in "site order" (all of site 0's modes, then all of site 1's, ...).
        ``site_to_link_perm`` is the permutation between the two orderings, so a mode ``i`` in link order sits
        at site-order position ``site_to_link_perm.argmax(axis=0)[i]``. In site order that position equals
        ``site * modes_per_site + local_mode``, so ``// modes_per_site`` gives the site and ``% modes_per_site``
        the local mode; grouping indices by site and sorting each group by local mode yields ``virt_inds``.
        """
        if self._grad_norm_block_maps is None:
            nsites = self.cfg.lattice.size
            # Permutation from site-order (contiguous per site) to link-order (used by the system matrices).
            site_to_link_perm = np.asarray(
                generate_permutation_matrix(self.get_site_based_mode_order(), self.get_link_based_mode_order())
            )
            n_virt_modes = site_to_link_perm.shape[0]
            modes_per_site = n_virt_modes // nsites  # virtual majorana modes on one site (32 for 2-copy D6)

            # site_to_link_perm has a single 1 per column; argmax over rows gives, for each mode in the
            # system (link) ordering, the position it occupies in the contiguous per-site (site) ordering.
            site_order_of_mode = site_to_link_perm.argmax(axis=0)
            # Split the contiguous site-order index into (which site, which local mode 0..modes_per_site-1).
            site_of_mode = site_order_of_mode // modes_per_site  # which site each system-order mode belongs to
            local_mode_of_mode = site_order_of_mode % modes_per_site  # its mode index within that site

            # virt_inds[s] = system-order indices of site s's virtual modes, in site-local order (so the
            # per-site block sliced from prod aligns entry-for-entry with the site-level derivative block).
            virt_inds = np.zeros((nsites, modes_per_site), dtype=np.int64)
            for s in range(nsites):
                modes_on_site = np.where(site_of_mode == s)[0]  # system indices belonging to site s
                virt_inds[s] = modes_on_site[np.argsort(local_mode_of_mode[modes_on_site])]  # order by local mode

            # uc_mask[u, s] = 1 iff site s is in unit-cell group u (used to sum per-site blocks within a group).
            uc_mask = np.zeros((self.cfg.unitcell_size, nsites))
            for s in range(nsites):
                uc_mask[self.cfg.site_params_dict[s], s] = 1.0

            # nonzero_mask = 0 for parameters frozen to zero by symmetry (their gradient is unused), else 1.
            nonzero_mask = np.ones((self.cfg.nlayer, self.cfg.unitcell_size, len(self.symbolvec)))
            for lay, uc_ind, symbol_ind in self.cfg.zeroed_params:
                nonzero_mask[lay, uc_ind, symbol_ind] = 0.0
            self._grad_norm_block_maps = (
                xnp.asarray(virt_inds),
                xnp.asarray(uc_mask),
                xnp.asarray(nonzero_mask),
            )
        return self._grad_norm_block_maps

    @property
    def grad_norm_site_deriv_blocks(self) -> xnp.ndarray:
        """Site-level parameter derivatives, virtual block only, shape
        (nlayer, unitcell_size, nparams, modes_per_site, modes_per_site).

        Param-dependent (rebuilt per ``initialize``), gauge-independent. This is the small site-level
        counterpart of the dense system-sized derivative ``gamma_maj_sys_deriv_layvec_ucvec_symbvec``, used
        by ``compute_grad_over_norm_vec``: because the system derivative is block-diagonal per site with an
        identical block on every site (translation invariance), one per-site block per parameter carries all
        the information, so this stores that block instead of the full system-sized matrix.
        """
        if self._grad_norm_site_deriv_blocks is None:
            offset_site = 2 * self.cfg.nphysmodes_site  # physical majorana modes per site
            lay_blocks = []
            for lay in range(self.cfg.nlayer):
                uc_blocks = []
                for uc_ind in range(self.cfg.unitcell_size):
                    param_blocks = [
                        self.compute_gamma_maj_deriv(symb, lay, uc_ind)[offset_site:, offset_site:]
                        for symb in self.symbolvec
                    ]
                    uc_blocks.append(xnp.stack(param_blocks))
                lay_blocks.append(xnp.stack(uc_blocks))
            self._grad_norm_site_deriv_blocks = xnp.stack(lay_blocks)
        return self._grad_norm_site_deriv_blocks

    @staticmethod
    @maybe_jit(
        static_argnames=[
            "nlayer",
            "unitcell_size",
            "nparams",
        ]
    )
    def compute_grad_over_norm_vec(
        nlayer: int,
        unitcell_size: int,
        nparams: int,
        nonzero_mask: xnp.ndarray,
        wi_gamma_out_vec: xnp.ndarray,
        site_deriv_blocks: xnp.ndarray,
        virt_inds: xnp.ndarray,
        uc_mask: xnp.ndarray,
    ) -> xnp.ndarray:
        r"""Compute the gradient of the norm divided by the norm for all layers with respect to all parameters.

        NOTE: this is a performance-optimized (block-structured) rewrite. A simpler, easier-to-follow dense
        reference implementation of this function lives at commit 3679276.

        The gradient is ``grad_a = -0.5 * Tr(deriv_d_a @ prod)`` with
        ``prod = mat_d_inv @ wi_gamma_in @ gamma_in_sys`` (per layer) and ``deriv_d_a`` the
        virtual-virtual block of ``d gamma_maj_sys / d theta_a``.

        ``prod`` is not computed here: since ``gamma_in_sys`` is a pure-state covariance
        (:math:`\Gamma^2 = -1`, each link block is a pure gauged projector state),
        ``mat_d_inv @ (mat_d_inv - gamma_in_sys)^{-1} @ gamma_in_sys = (1 - \Gamma D)^{-1}\Gamma
        = -(D + \Gamma)^{-1} = -wi_gamma_out``, which is already tracked incrementally.

        Block-structured evaluation (exploited structure): ``deriv_d_a`` is block-diagonal per site, and
        every site in a unit-cell class shares one *identical* diagonal block -- the site-level derivative
        block ``site_deriv_blocks[layer, uc, a]`` (from ``compute_gamma_maj_deriv``). (For a translation-
        invariant ansatz there is a single class of all sites; for unitcell_size > 1 each class is summed
        separately via ``uc_mask``.) Therefore only the same-site diagonal blocks of ``prod`` are ever needed:
        ``grad_a = -0.5 * Tr(site_deriv_block_a @ prod_sum)`` where
        ``prod_sum = sum over sites s in the unit-cell group of prod[virt_inds[s], virt_inds[s]]``.
        This replaces the 20 full-system-sized traces of the previous dense implementation with one small
        per-site trace per parameter.

        Args:
            nlayer (int): Number of layers
            unitcell_size (int): Size of the unit cell
            nparams (int): Number of parameters per (layer, unit-cell) site
            nonzero_mask (xnp.ndarray): (nlayer, unitcell_size, nparams) float mask; 0 for parameters forced
                to zero by the ansatz (their gradient is not used), 1 otherwise.
            wi_gamma_out_vec (xnp.ndarray): the system's wi_gamma_out_vec, i.e. (mat_d + gamma_in_sys)^-1,
                per layer (equals ``-prod``, see above).
            site_deriv_blocks (xnp.ndarray): (nlayer, unitcell_size, nparams, modes_per_site, modes_per_site)
                virtual-block site-level parameter derivatives.
            virt_inds (xnp.ndarray): (nsites, modes_per_site) int; per-site virtual-mode indices, ordered
                to match the site-local mode ordering of ``site_deriv_blocks``.
            uc_mask (xnp.ndarray): (unitcell_size, nsites) float; uc-group membership of each site.

        Returns:
            xnp.ndarray: Vector of gradients of the norm divided by the norm wrt all parameters.
        """
        dest_grad = xnp.zeros((nlayer, unitcell_size, nparams))

        for layerind in range(nlayer):
            # prod = mat_d_inv @ wi_gamma_in @ gamma_in_sys = -(D + gamma_in)^-1 (pure-state identity).
            prod = -wi_gamma_out_vec[layerind]

            # Diagonal per-site blocks of prod: (nsites, modes_per_site, modes_per_site)
            site_blocks = prod[virt_inds[:, :, None], virt_inds[:, None, :]]
            # Sum those blocks over the sites of each unit-cell group: (unitcell_size, modes_per_site, ...)
            prod_sum = xnp.einsum("us,smn->umn", uc_mask, site_blocks)
            # grad[uc, param] = -0.5 * trace(site_deriv_block[uc, param] @ prod_sum[uc]);
            # the trace is real for these antisymmetric factors -- make the cast explicit.
            grad = xnp.real(-0.5 * xnp.einsum("uaij,uji->ua", site_deriv_blocks[layerind], prod_sum))
            grad = grad * nonzero_mask[layerind]  # drop parameters frozen to zero

            dest_grad = backend.array_assign(dest_grad, layerind, grad)

        return dest_grad

    ################## Weight management ######################

    @property
    def weight(self) -> float:
        """Return the Monte Carlo weight of the current configuration.
        This function is a get function.
        If the value does not exist, it will be calculated.
        Each subsequent Monte Carlo step will update the weight.
        A full computation is only necessary in the beginning of the procedure.

        Returns:
            float: Weight of the MC configuration
        """
        if self._weight is None:
            self._weight = self.calculate_lognorm(incremental=True)
        return self._weight

    @weight.setter
    def weight(self, val: float) -> None:
        """Setter of the weight"""
        self._weight = val

    def calculate_weight_attempt(self, link_ind: int, theta: np.ndarray, all_factors=False) -> float:
        """
        Compute the weight of an update attempt in which the link index link_ind is substituted for theta
        The inclusion of all constant pre-factors can be switched on and off.


        Args:
            link_ind (int): Link index
            theta (array): New gauge field value
            all_factors (bool, optional): Include all constant factors. Defaults to False.

        Returns:
            float: Logarithm of the weight of the proposed configuration
        """
        theta = xnp.asarray(theta)
        coord, dir = self.cfg.lattice.ind2coord_dir(link_ind)
        return self._calculate_weight_attempt(
            self.gamma_in_sys_vec,
            self.gamma_gauge_neutral_vec,
            theta,
            coord,
            link_ind,
            self.incdet_vec,
            self.wi_gamma_in_vec,
            self.mat_d_vec,
            ncopy=self.cfg.ncopy,
            nvirtmodes_link=self.cfg.nvirtmodes_link,
            dir=dir,
            all_factors=all_factors,
        )

    @classmethod
    @maybe_jit(static_argnames=["cls", "ncopy", "nvirtmodes_link", "dir", "all_factors"])
    def _calculate_weight_attempt(
        cls,
        gamma_in_sys_vec: xnp.ndarray,
        gamma_gauge_neutral_vec: xnp.ndarray,
        theta: xnp.ndarray,
        coord: tuple,
        link_ind: int,
        incdet_vec: xnp.ndarray,
        wi_gamma_in_vec: xnp.ndarray,
        mat_d_vec: xnp.ndarray,
        ncopy: int,
        nvirtmodes_link: int,
        dir: Direction,
        all_factors: bool,
    ) -> float:
        """Pure weight of a single-link update attempt, without touching any tracker.

        link_ind, theta, coord and all arrays may be traced; the static arguments select one
        compilation per (direction, all_factors).

        Returns:
            float: Logarithm of the weight of the proposed configuration
        """
        # Each virtual mode is two Majorana modes -> 2*nvirtmodes_link Majoranas per link
        ind_mat = 2 * nvirtmodes_link * link_ind
        rotmat = cls.generate_rotmat(ncopy, theta, coord, dir)
        gamma_in_subst_layers = rotmat @ gamma_gauge_neutral_vec[:, dir] @ xnp.transpose(rotmat)
        updates = cls.calculate_update_gamma_in(ind_mat, gamma_in_subst_layers, gamma_in_sys_vec)
        return cls.update_lognorm_inc(
            incdet_vec, wi_gamma_in_vec, gamma_in_sys_vec, mat_d_vec, ind_mat, updates, all_factors
        )

    def calculate_lognorm(self, incremental: bool, all_factors: bool = False) -> float:
        """Compute the logarithm of the norm.

        Args:
            incremental (bool): Whether to use the incremental trackers.
            all_factors (bool, optional): Include all pre-factors in the computation. Defaults to False.

        Returns:
            float: Logarithm of the norm
        """
        lognormvec = self.calculate_lognormvec(incremental=incremental, all_factors=all_factors)
        return xnp.sum(lognormvec)

    def calculate_lognormvec(self, incremental: bool, all_factors: bool = False) -> xnp.ndarray:
        """Compute the logarithm of the norm for each layer.

        Args:
            incremental (bool): Whether to use the incremental trackers.
            all_factors (bool, optional): Include all constant prefactors. Defaults to False.

        Returns:
            float: Logarithm of the norm
        """
        if incremental:
            res = self._calculate_lognormvec_inc(
                self.incdet_vec,
                self.det_mat_d_vec,
                self.gamma_in_sys_vec[0].shape[0],
                all_factors=all_factors,
            )
        else:
            res = self._calculate_lognormvec(self.gamma_in_sys_vec, self.mat_d_vec, all_factors=all_factors)
        return res

    @staticmethod
    @maybe_jit(static_argnames=["all_factors"])
    def _calculate_lognormvec(
        gamma_in_sys_vec: xnp.ndarray,
        mat_d_vec: xnp.ndarray,
        all_factors: bool = False,
    ) -> xnp.ndarray:
        """Calculate the lognormvec from scratch (not incrementally)."""

        # This is still the plain formula, without any update mechanism
        nlayer = mat_d_vec.shape[0]
        n = mat_d_vec[0].shape[0]

        prod_vec = gamma_in_sys_vec @ mat_d_vec
        eye_vec = xnp.broadcast_to(xnp.eye(n), (nlayer, n, n))

        sign_vec, logval_vec = xnp.linalg.slogdet((eye_vec - prod_vec))
        if all_factors:
            logval_vec -= n * xnp.log(2)
        else:
            # We are skipping a global factor of 2**(-n) here, to get a reasonable size of the norm
            pass

        # The factor 1/2 is the square-root
        return logval_vec / 2

    @staticmethod
    @maybe_jit(static_argnames=["n", "all_factors"])
    def _calculate_lognormvec_inc(
        det_vec: xnp.ndarray, det_mat_d_vec: xnp.ndarray, n: int, all_factors: bool = False
    ) -> xnp.ndarray:
        """Calculate the lognormvec incrementally, i.e. by incrementally updating the previous value
        (using IncDet and Woodbury)."""

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

    @staticmethod
    @maybe_jit(static_argnames=["all_factors"])
    def update_lognorm_inc(
        incdet_vec: xnp.ndarray,
        d_inv_minus_gammain_inv_vec: xnp.ndarray,
        gamma_in_sys_vec: xnp.ndarray,
        mat_d_vec: xnp.ndarray,
        offset: int,
        updates: xnp.ndarray,
        all_factors: bool = False,
    ) -> float:
        """Update the logarithm of the norm incrementally with the given update.

        Args:
            incdet_vec (xnp.ndarray): Vector of incremental determinants of (D^-1 - gammain)^-1
            d_inv_minus_gammain_inv_vec (xnp.ndarray): Vector of (D^-1 - gammain)^-1
            gamma_in_sys_vec (xnp.ndarray): gamma_in for the whole system
            mat_d_vec (xnp.ndarray): Vector of D matrices
            offset (int): Offset into the matrix.
            updates (xnp.ndarray): Update matrix to replace the current sub-matrix
            all_factors (bool, optional): Include all constant pre-factors. Defaults to False.

        Returns:
            float: Updated logarithmic value of the norm
        """
        cumval = 0
        detval_vec = utils.IncLogAbsDeterminant.update_index(
            incdet_vec, d_inv_minus_gammain_inv_vec, updates, offset, offset
        )
        if all_factors:
            detval_vec -= gamma_in_sys_vec[0].shape[0] * xnp.log(2)
            detval_vec += xnp.linalg.slogdet(mat_d_vec)[1]
        # The factor 0.5 is the sqrt of the formula. We are storing the logarithm of the norm.
        # The addition of the cumval is the multiplication of the indpendent PEPS
        cumval += 0.5 * xnp.sum(detval_vec)
        return cumval

    @property
    def grad_over_norm_vec(self) -> xnp.ndarray:
        """Compute the gradient of the norm divided by the norm over all parameters for all layers and unit cells.

        Returns:
            xnp.ndarray: Vector of gradients of the norm divided by the norm with respect to all parameters
        """
        if self._grad_over_norm_vec is None:
            virt_inds, uc_mask, nonzero_mask = self.grad_norm_block_maps
            self._grad_over_norm_vec = self.compute_grad_over_norm_vec(
                self.cfg.nlayer,
                self.cfg.unitcell_size,
                len(self.symbolvec),
                nonzero_mask,
                self.wi_gamma_out_vec,
                self.grad_norm_site_deriv_blocks,
                virt_inds,
                uc_mask,
            )
        return self._grad_over_norm_vec

    ################## Local Gauge ######################

    @property
    def gaugefieldvec(self) -> xnp.ndarray:
        """Return the vector of gauge fields.

        Returns:
            xnp.ndarray: Gauge fields of the simulation
        """
        return self._gaugefieldvec

    @gaugefieldvec.setter
    def gaugefieldvec(self, val):
        logger.error("Do not set the gaugefieldvec explicitly. Use 'update_gauge_ind'.")

    @property
    def gamma_gauge_neutral_vec(self) -> xnp.ndarray:
        if self._gamma_gauge_neutral_vec_dirs is None:
            self._gamma_gauge_neutral_vec_dirs = self._generate_gamma_gauge_neutral_dict()
        return self._gamma_gauge_neutral_vec_dirs

    def _generate_gamma_gauge_neutral_dict(self) -> xnp.ndarray:
        """Define the ungauged covariance matrix of a single link.
        The substitution method must ensure a consistent order of the modes.
        The direction parameter controls which covariance matrix is retrieved,
        since these can differ between directions.
        """
        gamma_gauge_neutral_vec_dirs = self.cfg.generate_gamma_gauge_neutral_dict()
        return xnp.array(gamma_gauge_neutral_vec_dirs)

    @classmethod
    @maybe_jit(static_argnames=["cls", "ncopy"])
    def generate_rotmat(cls, ncopy: int, group_element: xnp.ndarray, coord: tuple, dir: Direction) -> xnp.ndarray:
        """Generate the matrix to rotate gamma_in_neutral according to a given gauge field value.
        We work in the convention where for majorana c_i: U_g c_i U_g^dagger = sum_{j} rotmat_{i,j} c_j
        In this convention gamma_in_neutral, is gauged with gamma_in = rotmat @ gamma_neutral rotnat^T.
        Where gamma_in is the covariance matrix of the state U_g^dagger w |Omega>.

        The mode order is the same as for gamma_in_neutral:
        we order first by link, then by copy, then by color.

        The sites are picked such that the left mode is right of the right modes,
        i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        Args:
            ncopy (int): number of copies of the ansatz
            group_element (xnp.array): representation of group element
            coord (tuple): (x,y) coordinate on the lattice
            dir (lattice.Direction): direction of the link from (x,y)

        Returns:
            xnp.ndarray: Rotation matrix for gamma_in_neutral
        """
        g = group_element
        # We are only rotating the right modes.
        # Thus, we leave an identity matrix for the left modes.
        real_g = xnp.real(g)
        imag_g = xnp.imag(g)
        # Gauging is different for different sublattices: even sites use g, odd sites conjugate(g).
        # Note that this gauging is true only for b modes and c virtual modes
        # (in the conventions of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.110.054511).
        # TODO: Generalize this to fermionic layers as well.
        sign = (-1) ** xnp.sum(xnp.asarray(coord))
        rot_right = xnp.block(
            [
                [real_g, -sign * imag_g],
                [sign * imag_g, real_g],
            ],
        )  # This is the rot_right for the mode order of {r_1_1, r_1_2,r_2_1,r_2_2}

        # We have dim(representation) left mode => 2*dim(representation) Majorana modes
        dim_rep = len(g)  # dimension of the representation
        rot_left = xnp.eye(2 * dim_rep)

        # The mode order is lr (horizontally) or du (vertically).
        # We rotate the different copies in the SAME way.
        dest = xscipy.linalg.block_diag(
            rot_left, rot_right
        )  # This is the rot for the mode order of {l_1_1, l_1_2,l_2_1,l_2_2, r_1_1, r_1_2,r_2_1,r_2_2}

        rotmat = xnp.kron(xnp.eye(ncopy), dest)

        # TODO: we should rather just order correctly from the start
        perm_mat = cls._rotmat_perm_mat(ncopy, dim_rep)
        rotmat = xnp.transpose(perm_mat) @ rotmat @ perm_mat

        return rotmat

    @classmethod
    @functools.lru_cache
    def _rotmat_perm_mat(cls, ncopy: int, rep_dim: int) -> np.ndarray:
        """Constant permutation taking generate_rotmat's copy-then-color block order to the
        color-then-copy order used everywhere else. Depends only on ncopy and the representation
        dimension, so it is cached. Plain numpy on purpose: caching an array created inside a
        jit trace would leak a tracer."""
        copy_then_color_order = cls.get_single_link_majorana_mode_order_first_copy_then_color(ncopy, rep_dim)
        color_then_copy_order = cls.get_single_link_majorana_mode_order(ncopy, rep_dim)
        return np.array(generate_permutation_matrix(copy_then_color_order, color_then_copy_order))

    @classmethod
    @maybe_jit(static_argnames=["cls", "mod_link_inds", "nvirtmodes_link", "dir", "defer_mod"])
    def _update_gauge_ind(
        cls,
        gamma_in_sys_vec: xnp.ndarray,
        gamma_gauge_neutral_vec: xnp.ndarray,
        rotmat: xnp.ndarray,
        link_ind: int,
        wi_gamma_in_vec: xnp.ndarray,
        wi_gamma_out_vec: xnp.ndarray,
        incdet_vec: xnp.ndarray,
        gamma_in_sys_mod_vec: Optional[xnp.ndarray],
        wi_gamma_in_mod_vec: Optional[xnp.ndarray],
        wi_gamma_out_mod_vec: Optional[xnp.ndarray],
        incdet_mod_vec: Optional[xnp.ndarray],
        mod_link_inds: tuple[int, ...],
        nvirtmodes_link: int,
        dir: Direction,
        defer_mod: bool,
    ) -> tuple:
        """Pure single-link update of gamma_in_sys and all incremental trackers.

        link_ind and all arrays may be traced; the static arguments select one compilation per
        (direction, warmup/measurement). With defer_mod=True (warmup) the mod family arguments are
        ignored and returned unchanged.

        Returns:
            tuple: (gamma_in_sys_vec, wi_gamma_in_vec, wi_gamma_out_vec, incdet_vec, weight,
                gamma_in_sys_mod_vec, wi_gamma_in_mod_vec, wi_gamma_out_mod_vec, incdet_mod_vec,
                max_inv_mag)
        """
        # Each virtual mode is two Majorana modes -> 2*nvirtmodes_link Majoranas per link
        ind_mat = 2 * nvirtmodes_link * link_ind

        gamma_neutral = gamma_gauge_neutral_vec[:, dir]  # (nlayer, k, k)
        gamma_in_subst = rotmat @ gamma_neutral @ xnp.transpose(rotmat)  # broadcasts over layers
        update_arr = cls.calculate_update_gamma_in(ind_mat, gamma_in_subst, gamma_in_sys=gamma_in_sys_vec)

        # Substitute in the array, all layers at once
        gamma_in_sys_vec = backend.dynamic_update_slice(gamma_in_sys_vec, (0, ind_mat, ind_mat), gamma_in_subst)

        # --- Incrementally update the closed (full-system) trackers via Woodbury / IncDet.
        # gamma_out now inverts (mat_d + gamma_in), so a +Delta change in gamma_in enters its
        # inverted matrix with the opposite sign -> the out-tracker Woodbury step uses -update.
        incdet_vec = utils.IncLogAbsDeterminant.update_index(incdet_vec, wi_gamma_in_vec, update_arr, ind_mat, ind_mat)
        weight = 0.5 * xnp.sum(incdet_vec)
        wi_gamma_in_vec = utils.WoodburyInverter.update_index(wi_gamma_in_vec, update_arr, ind_mat, ind_mat)
        wi_gamma_out_vec = utils.WoodburyInverter.update_index(wi_gamma_out_vec, -update_arr, ind_mat, ind_mat)
        # Largest entry of any updated inverse, for the refresh magnitude guard.
        inv_mags = [xnp.max(xnp.abs(wi_gamma_in_vec)), xnp.max(xnp.abs(wi_gamma_out_vec))]

        # --- Incrementally update the modified (open-link) family - gamma_in_sys_mod and its
        # trackers -- batched over layers. The local update is shifted by the carved-out link when
        # it sits below the changed link. When the changed link IS the measured link its modes are
        # not in that entry: the update is masked to zero, so Woodbury, the determinant lemma
        # and the block add all leave that entry exactly unchanged (link_ind may be traced).

        # Skipped entirely during warmup (defer_mod_trackers): these are measurement-only and are
        # recomputed from scratch (recompute_mod_trackers) when warmup ends.
        if not defer_mod:
            assert gamma_in_sys_mod_vec is not None  # for mypy
            assert wi_gamma_in_mod_vec is not None
            assert wi_gamma_out_mod_vec is not None
            assert incdet_mod_vec is not None
            k = 2 * nvirtmodes_link
            maxpos = wi_gamma_in_mod_vec.shape[-1] - k
            for ind, mod_link_ind in enumerate(mod_link_inds):
                # If the changed link is the measured link itself, there is nothing to update here:
                # update is set to zero. Then, if it is also the last link, pos falls past the matrix
                # end and the clip brings it back inside.
                pos = xnp.clip(ind_mat - k * (link_ind > mod_link_ind), 0, maxpos)
                update = xnp.where(link_ind == mod_link_ind, 0.0, update_arr)

                # update = cur - new block, so cur - update is the new block.
                cur = backend.take_block(gamma_in_sys_mod_vec[:, ind], pos, k, axis=-2)
                cur = backend.take_block(cur, pos, k, axis=-1)
                gamma_in_sys_mod_vec = backend.dynamic_update_slice(
                    gamma_in_sys_mod_vec, (0, ind, pos, pos), (cur - update)[:, None]
                )

                new_det = utils.IncLogAbsDeterminant.update_index(
                    incdet_mod_vec[:, ind], wi_gamma_in_mod_vec[:, ind], update, pos, pos
                )
                incdet_mod_vec = backend.array_assign(incdet_mod_vec, (slice(None), ind), new_det)
                new_in = utils.WoodburyInverter.update_index(wi_gamma_in_mod_vec[:, ind], update, pos, pos)
                wi_gamma_in_mod_vec = backend.array_assign(wi_gamma_in_mod_vec, (slice(None), ind), new_in)
                new_out = utils.WoodburyInverter.update_index(wi_gamma_out_mod_vec[:, ind], -update, pos, pos)
                wi_gamma_out_mod_vec = backend.array_assign(wi_gamma_out_mod_vec, (slice(None), ind), new_out)
                inv_mags.append(xnp.max(xnp.abs(new_in)))
                inv_mags.append(xnp.max(xnp.abs(new_out)))

        max_inv_mag = xnp.max(xnp.asarray(inv_mags))
        return (
            gamma_in_sys_vec,
            wi_gamma_in_vec,
            wi_gamma_out_vec,
            incdet_vec,
            weight,
            gamma_in_sys_mod_vec,
            wi_gamma_in_mod_vec,
            wi_gamma_out_mod_vec,
            incdet_mod_vec,
            max_inv_mag,
        )

    def update_gauge_ind(self, link_ind: int, gauge_val: np.ndarray) -> None:
        """Update a gauge field at a given link index by a new value.
        This function can be called from outside the system, and so accepts a gauge field value as np.ndarray.
        We convert here to xnp.ndarray. The heavy per-step work runs in the jitted classmethod
        _update_gauge_ind. After the incremental update it recomputes the trackers from scratch
        when the periodic interval elapsed or the magnitude guard fired.

        Args:
            link_ind (int): Link index of the gauge field to be updated
            gauge_val (np.ndarray): New value for the gauge field

        Returns:
            None
        """
        theta = xnp.asarray(gauge_val)
        if not xnp.allclose(self.gaugefieldvec[link_ind], theta):
            # only actually do the update if it's a different gauge field

            # Update the gaugefield
            self._gaugefieldvec = backend.array_assign(self._gaugefieldvec, link_ind, theta)

            coord, dir = self.cfg.lattice.ind2coord_dir(link_ind)
            rotmat = self.generate_rotmat(self.cfg.ncopy, theta, coord, dir)

            defer_mod = self.defer_mod_trackers
            res = self._update_gauge_ind(
                self.gamma_in_sys_vec,
                self.gamma_gauge_neutral_vec,
                rotmat,
                link_ind,
                self.wi_gamma_in_vec,
                self.wi_gamma_out_vec,
                self.incdet_vec,
                None if defer_mod else self.gamma_in_sys_mod_vec,
                None if defer_mod else self.wi_gamma_in_mod_vec,
                None if defer_mod else self.wi_gamma_out_mod_vec,
                None if defer_mod else self.incdet_mod_vec,
                mod_link_inds=self.cfg.mod_link_inds,
                nvirtmodes_link=self.cfg.nvirtmodes_link,
                dir=dir,
                defer_mod=defer_mod,
            )
            self._gamma_in_sys_vec, self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec, self.weight = res[
                :5
            ]
            if not defer_mod:
                # modified trackers family
                (
                    self._gamma_in_sys_mod_vec,
                    self._wi_gamma_in_mod_vec,
                    self._wi_gamma_out_mod_vec,
                    self._incdet_mod_vec,
                ) = res[5:9]
            # float() = sync point: pull the tracker max from GPU for the refresh guard
            self._last_step_max_inv_mag = float(res[9])

            # Invalidate gauge dependent quantities
            self.invalidate_gauge_update()

            self._maybe_refresh_trackers()

    def _maybe_refresh_trackers(self) -> None:
        """Recompute both tracker families from scratch to bound the incremental Woodbury/IncDet
        drift. Three modes via tracker_refresh_interval (set per run from EvaluatorManager /
        --tracker_refresh_interval):
          > 0 : periodic every `interval` steps AND the magnitude guard (default; 1 == every step)
          == 0: magnitude guard only, no periodic recomputation ("no_periodic")
          <  0: no refresh at all -- pure incremental, even if a tracked inverse blows up
        """
        interval = self.tracker_refresh_interval
        if interval >= 0:
            periodic_due = False
            if interval > 0:
                self._steps_since_refresh += 1
                periodic_due = self._steps_since_refresh >= interval
            if periodic_due or self._last_step_max_inv_mag > self.TRACKER_INV_MAG_THRESH:
                self.refresh_trackers()

    def update_gauge_full_system(self, gaugeconfig: list[np.ndarray]) -> None:
        """Replace all gauge fields on the links by the values given in gaugeconfig.

        Args:
            gaugeconfig (list[np.ndarray]): Array of new values for the gauge field
        """
        for link_ind, gauge_val in enumerate(gaugeconfig):
            self.update_gauge_ind(link_ind, gauge_val)

    def update_gauge_coord(self, coord: tuple[int, int], dir: Direction, gauge_val: np.ndarray) -> None:
        """Update a gauge field at a given coordinate and direction by a new value

        Args:
            coord (tuple): Coordinate of the vertex
            dir (Direction): Direction of the link
            theta (np.array): New value for the gauge field
        """
        ind = self.cfg.lattice.coord2ind_dir(coord, dir)
        self.update_gauge_ind(ind, gauge_val)

    @staticmethod
    def calculate_update_gamma_in(offset: int, update_mat: xnp.ndarray, gamma_in_sys: xnp.ndarray) -> xnp.ndarray:
        """Compute an update between the current gamma_in and the new gamma_in.

        Args:
            offset (int): Offset in the matrix
            update_mat (xnp.ndarray): Array to replace the current content of gamma_in at offset
            gamma_in_sys (xnp.ndarray): gamma_in_sys. This is given as an argument so that different gamma_in_sys
                can be passed in when gamma_in_sys differs between layers.

        Returns:
            xnp.ndarray: Additional update to reach update_mat at gamma_in[offset:,offset:]
        """
        m_up, n_up = update_mat.shape[-2:]
        gamma_in_old = backend.take_block(gamma_in_sys, offset, m_up, axis=-2)
        gamma_in_old = backend.take_block(gamma_in_old, offset, n_up, axis=-1)
        return -(update_mat - gamma_in_old)

    ################## Observables ######################
    def _compute_mag_energy_op(self, use_trans_inv: bool = False) -> float:
        """Compute the magnetic energy operator (w/o shift).
        This operator is diagonal in the gauge field (group element) basis and can thus
        be computed easily.

        Args:
            use_trans_inv (bool, optional): Evaluate a single plaquette and multiply by the number
                of plaquettes. Only valid for a translation invariant ansatz. Defaults to False.

        Returns:
            float: magnetic energy w/o shift for the whole system
        """
        if use_trans_inv:
            if self.cfg.unitcell_size > 1:
                raise ValueError("Cannot rely on translation invariance if unitcell size is >1.")

            # Evaluate one plaquette and multiply by number of plaquettes
            wilson_plaquette = self.cfg.lattice.generate_wilson_loop((0, 0), (1, 1))
            nplaq = self.cfg.lattice.nplaquettes
            mag_energy_bare = nplaq * xnp.real(self.compute_path(wilson_plaquette))
        else:
            # Evaluate every plaquette of the system
            mag_energy_bare = 0
            for x in range(self.cfg.lattice.nx):
                for y in range(self.cfg.lattice.ny):
                    wilson_plaquette = self.cfg.lattice.generate_wilson_loop((x, y), (1, 1))
                    mag_energy_bare += xnp.real(self.compute_path(wilson_plaquette))
        return mag_energy_bare

    @staticmethod
    @maybe_jit(static_argnames=["mod_link_inds", "nlayer", "coeffs_vec", "constants_vec"])
    def _compute_el_energy_op_vec(
        lognormvec_default: xnp.ndarray,
        mod_link_inds: tuple[int, ...],
        nlayer: int,
        el_pfaffians: xnp.ndarray,
        norm_mod_vec: xnp.ndarray,
        group_elements_for_el_energy: tuple[xnp.ndarray, ...],
        coeffs_vec: CoeffsVec,
        constants_vec: ConstantsVec,
    ) -> xnp.ndarray:
        """Compute the electric energy.

        Args:
            TODO: describe arguments

        Returns:
            array: electric energies for the links specified in self.cfg.mod_link_inds for all layers
                   with shape: (num_group_elements, nlayer, len(self.cfg.mod_link_inds))
        """
        lognorm_default = xnp.sum(lognormvec_default)

        num_el_links = len(mod_link_inds)  # number of links on which the electric energy is computed
        num_group_elements = len(group_elements_for_el_energy)
        dest = xnp.zeros((num_group_elements, nlayer, num_el_links), dtype=complex)

        # TODO: vectorize!
        for group_element_idx in range(num_group_elements):
            # coeffs for the specific group element, in the unique-index basis (cfg.uniq_coeffs_vec):
            # the pfaffians are computed once per unique index tuple (no group-element axis) and
            # each group element applies its own coefficient dot product. For Z2 there is only 1
            # stored group element anyway.
            coeffs_group_element = coeffs_vec[group_element_idx]
            for layerind in range(nlayer):
                layer_coeffs = coeffs_group_element[layerind]  # tuple of tuples of coeffs
                norm_mod_linkvec = norm_mod_vec[layerind]

                # Iterate over the links
                for link_pos, norm_mod in enumerate(norm_mod_linkvec):
                    ###################### Calculation of sum f_j |jmn><jmn| ########################
                    link_coeffs = layer_coeffs[link_pos]
                    pf_tot: complex = constants_vec[group_element_idx][layerind][link_pos]
                    # this is the constant term in the sum, which does not come with a Pfaffian,
                    # for Z2 it should be 0
                    for size_ind, size_term in enumerate(link_coeffs):
                        # TODO: We should convert this to array outside the loop
                        array_size_term = xnp.asarray(size_term)
                        current_pfaffians = el_pfaffians[layerind, link_pos, size_ind, : len(size_term)]
                        pf_tot += xnp.dot(array_size_term, current_pfaffians)

                    # Keep el_energy_link COMPLEX. The real part is taken only after the product over
                    # layers and the sum over group elements (prod(Re) != Re(prod)).
                    el_energy_link = pf_tot * xnp.exp(norm_mod - lognorm_default)

                    dest = backend.array_assign(dest, (group_element_idx, layerind, link_pos), el_energy_link)
        return dest

    @staticmethod
    @maybe_jit(
        static_argnames=[
            "lattice_size",
            "num_pg_layer",
            "num_fermionic_layer",
            "unitcell_size",
            "nvirtmodes_link",
            "nphysmodes_site",
            "mod_link_inds",
            "symbolvec",
            "idxarr_vec",
            "coeffs_vec",
        ]
    )
    def _compute_el_grad_vec(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        unitcell_size: int,
        nvirtmodes_link: int,
        nphysmodes_site: int,
        mod_link_inds: tuple[int, ...],
        symbolvec: tuple,
        el_energy_vec: xnp.ndarray,
        mat_b_mod_vec: xnp.ndarray,
        covmat_out_mod_vec: xnp.ndarray,
        el_pfaffians: xnp.ndarray,
        norm_mod_vec: xnp.ndarray,
        lognorm_default_vec: xnp.ndarray,
        gamma_out_mod_inv_vec: xnp.ndarray,
        d_mat_a_vec: xnp.ndarray,
        d_mat_b_vec: xnp.ndarray,
        d_mat_d_vec: xnp.ndarray,
        grad_over_norm_vec: xnp.ndarray,
        inds: tuple,
        group_elements_for_el_energy: tuple[xnp.ndarray, ...],
        idxarr_vec: IdxGroup,
        coeffs_vec: CoeffsVec,
        rotmat_vec: xnp.ndarray,
        sp_ii: xnp.ndarray,
        sp_jj: xnp.ndarray,
        sp_vals: xnp.ndarray,
    ) -> xnp.ndarray:
        """Compute the electric energy gradients.

        In early 2026, this function was significantly optimized.
        This was done after it was generalized in various ways over the previous months:
            compute on multiple links, horizontal and vertical links, on different sublattices, for non-Abelian groups.
        As a result, it is somewhat harder to read.
        It may be easier to read the (slower and less general) versions at (in reverse chronological order)
            commit 6cbabbd: before significant vectorization
            commit 1d63a6b: after generalization to multiple hor/vert links, but before many optimizations,
        or even earlier versions.
        """
        num_group_elements = len(group_elements_for_el_energy)

        nlayer = num_pg_layer + num_fermionic_layer
        grad_shape = (num_group_elements, nlayer, len(mod_link_inds), unitcell_size, len(symbolvec))
        # real part is taken only after the product over layers (and sum over group elements).
        dest_grad = xnp.zeros(grad_shape, dtype=complex)

        nlinks = 2 * lattice_size  # valid for 2D with periodic boundary conditions
        k = 2 * nvirtmodes_link  # single link offset
        lognorm_default = xnp.sum(lognorm_default_vec)

        # Calculate the derivatives (wrt all non-zero parameters) of the modified covmat_out
        shape = (nlayer, len(mod_link_inds), unitcell_size, len(symbolvec), k, k)
        d_covmat_out_virt_vec = xnp.zeros(shape)

        l, m, u, s = inds
        # (nlayer, nmodlinks, mod_virt_dim, mod_virt_dim)
        # prod = mat_d_mod_inv @ wi_gamma_in_mod @ gamma_in_sys_mod = -(Dmod + gamma_in_mod)^-1:
        # gamma_in_sys_mod is a pure-state covariance (Gamma^2 = -1), so
        # (1 - Gamma D)^-1 Gamma = -(D + Gamma)^-1, which is the tracked wi_gamma_out_mod.
        prod_mod_norm_vec = -gamma_out_mod_inv_vec
        # (nlayer, nmodlinks, mod_virt_dim, link_dim), take only the last k columns
        diff_times_b_vec = gamma_out_mod_inv_vec @ xnp.swapaxes(mat_b_mod_vec, -1, -2)[:, :, :, -k:]
        # (nlayer, nmodlinks, link_dim, mod_virt_dim), take only the last k rows
        b_times_diff_vec = mat_b_mod_vec[:, :, -k:, :] @ gamma_out_mod_inv_vec

        shape = (nlayer, len(mod_link_inds), unitcell_size, len(symbolvec), k, k)
        d_covmat_out_virt_vec = xnp.zeros(shape)

        l, m, u, s = inds

        R_active = rotmat_vec[m]
        R_active_T = xnp.swapaxes(R_active, -1, -2)

        diffB = diff_times_b_vec[l, m]
        Bdiff = b_times_diff_vec[l, m]

        vals = (
            R_active_T
            @ (
                d_mat_a_vec[..., -k:, -k:]
                + d_mat_b_vec[..., -k:, :] @ diffB
                + Bdiff @ xnp.swapaxes(d_mat_b_vec, -1, -2)[..., :, -k:]
                - Bdiff @ d_mat_d_vec @ diffB
            )
            @ R_active
        )

        d_covmat_out_virt_vec = backend.array_assign(d_covmat_out_virt_vec, (l, m, u, s), vals)

        # Calculate the modified norms: trace(d_mat_d_a @ prod_mod_norm_a) for every active param a.
        # d_mat_d is a parameter-derivative -> config-independent and ~99% zero, so instead of the
        # dense form utils.trace_of_product((d_mat_d_vec, prod_mod_norm_vec[l, m])) - whose fancy-index
        # copy + full O(D^2) einsum dominated this function - we sum only over the
        # nonzero entries of d_mat_d: Tr(dD_a @ P_a) = sum_k vals[a,k] * P_a[jj[a,k], ii[a,k]], with
        # P_a = prod_mod_norm_vec[l[a], m[a]]. The (ii, jj, vals) are precomputed once per eval
        # (dmatd_trace_sparse). This gathers only the nonzeros.
        norm_shape = (nlayer, len(mod_link_inds), unitcell_size, len(symbolvec))
        prod_vec = xnp.zeros(norm_shape)
        # (num_active, kmax): prod at the nonzero (row=ii, col=jj) positions; trace uses P[jj, ii].
        p_gather = prod_mod_norm_vec[l[:, None], m[:, None], sp_jj, sp_ii]
        vals = xnp.sum(sp_vals * p_gather, axis=-1)
        prod_vec = backend.array_assign(prod_vec, (l, m, u, s), vals)

        # The pfaffian derivatives depend only on (layer, link, index tuple) -- NOT on the group
        # element: the pfaffians live in the unique-index basis (cfg.uniq_idx_vec), so each
        # derivative is computed once and the per-group-element coefficients (cfg.uniq_coeffs_vec,
        # same unique basis and length for every group element) are applied as a stacked dot below.
        # For Z2 there is only 1 stored group element anyway.
        deriv_pf_tot_vec_vec = xnp.zeros(
            (num_group_elements, nlayer, len(mod_link_inds), unitcell_size, len(symbolvec)), dtype=complex
        )

        for layerind in range(nlayer):

            for link_pos, _ in enumerate(mod_link_inds):

                for lens_ind in range(len(idxarr_vec[layerind][link_pos])):
                    # (# pfafs, pfaf submat dim)
                    inds_arr = xnp.asarray(idxarr_vec[layerind][link_pos][lens_ind])
                    # (num_group_elements, num_pfafs) coefficients in the unique basis
                    prefactors = xnp.asarray(
                        [coeffs_vec[ge][layerind][link_pos][lens_ind] for ge in range(num_group_elements)]
                    )

                    # We slice the last dimension because the el_pfaffians array is padded with zeros.
                    pfafs = el_pfaffians[layerind, link_pos, lens_ind, : len(inds_arr)]

                    virts = covmat_out_mod_vec[layerind][link_pos][
                        None, None, inds_arr[:, :, None], inds_arr[:, None, :]
                    ]
                    d_virts = d_covmat_out_virt_vec[layerind, link_pos][
                        :, :, inds_arr[:, :, None], inds_arr[:, None, :]
                    ]

                    # (unitcell_size, len(symbolvec), num_pfafs)
                    deriv_pf_tot_vectorized = utils.derivative_pfaffian_vectorized(virts, d_virts, pfafs)
                    deriv_pf_tot_vec_vec = backend.array_add(
                        deriv_pf_tot_vec_vec,
                        (slice(None), layerind, link_pos),
                        xnp.einsum("usn,gn->gus", deriv_pf_tot_vectorized, prefactors),
                    )

        for group_element_idx in range(num_group_elements):
            for layerind in range(nlayer):
                for link_pos, _ in enumerate(mod_link_inds):

                    d_el_energy_vec = deriv_pf_tot_vec_vec[group_element_idx, layerind, link_pos] * xnp.exp(
                        norm_mod_vec[layerind][link_pos] - lognorm_default
                    )

                    # Summand with derivative of norms
                    trace_def = grad_over_norm_vec[layerind]

                    # Instead of computing the modified grad over the norm as:
                    # compute_grad_over_norm(gamma_in_sys_mod, d_mat_d, mat_d_mod_inv, diff_d_inv_gamma_inv)
                    #    = -0.5 * trace(gamma_in_sys @ deriv_d @ mat_d_inv @ diff)
                    # we have saved the product of several mats above
                    # (since they don't change in inner loops), and use it here
                    trace_mod = -0.5 * prod_vec[layerind, link_pos]

                    # This is the second contribution of the elctric energy gradient F_{el} (\tilde(v) - v)
                    d_el_energy_vec += el_energy_vec[group_element_idx][layerind][link_pos] * (trace_mod - trace_def)

                    dest_grad = backend.array_add(dest_grad, (group_element_idx, layerind, link_pos), d_el_energy_vec)

        # scale to system size - currently only valid when all links should be weighed equally
        dest_grad *= nlinks / len(mod_link_inds)

        # We have to weigh the different layers with the electric energy operator expectation of the other layers.
        # They act as a prefactor in the derivative.
        # This must be done separately over all links.
        if nlayer > 1:
            for group_idx in range(num_group_elements):
                for lay in range(nlayer):
                    for linkind in range(len(mod_link_inds)):
                        prod_other_layers = utils.multiply_except(el_energy_vec[group_idx, :, linkind], lay)
                        dest_grad = backend.array_mult(dest_grad, (group_idx, lay, linkind), prod_other_layers)
        dest_grad = xnp.sum(dest_grad, axis=0)  # sum over group elements
        dest_grad = xnp.sum(dest_grad, axis=1)  # sum over the links

        # Take the real part only now, after the layer product and the group-element sum.
        return xnp.real(dest_grad)

    @staticmethod
    @abstractmethod
    def _compute_mass_energy_op_vec(
        occupations_after_ph: xnp.ndarray,
        use_trans_inv: bool = True,
    ) -> xnp.ndarray:
        """Compute the mass term of the Hamiltonian (per layer).

        This is an abstract method and has to be overwritten in a subclass.

        Args:
            use_trans_inv (bool, optional): Use translationally invariant implementation. Defaults to False.

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
    ) -> xnp.ndarray:
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
    ) -> xnp.ndarray:
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
    ) -> xnp.ndarray:
        """Compute the interaction energy gradient.
        This is an abstract method and has to be overwritten in a subclass.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    @abstractmethod
    def _compute_chem_energy_op_vec(
        occupations_before_ph: xnp.ndarray,
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
    def _meson_string_vec(self, path: tuple[tuple[int, bool], ...]) -> xnp.ndarray:
        r"""Compute a layer resolved meson string for the given path.
        This is \psi^dagger (start) * String * \psi(end) before particle-hole,
        and assumes that start and end are on the same sublattice.

        This is an abstract method and has to be overwritten in a subclass.

        Args:
            path (tuple): Tuple of tuples [(index,conj),....]. conj indicates whether the argument should be conjugated.

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
        mag_energy = (
            self.cfg.g_mag * 2 * (nplaq * self.cfg.gaugemgr.mag_offset - self.mag_energy_op)
        )  # The 2 is for the hermitian conjugate
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
        el_energy = self.cfg.g_el * (
            self.cfg.gaugemgr.el_offset * nlinks - self.cfg.gaugemgr.el_mult_factor * self.el_energy_op
        )
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
        # chem_energy = 0.0
        # for layer in range(self.cfg.num_pg_layer, self.cfg.nlayer):
        #    ind = layer - self.cfg.num_pg_layer
        #    chem_energy += self.cfg.g_chem[ind] * self.chem_energy_op_vec[layer]
        chem_energy = xnp.sum(self.cfg.g_chem * self.chem_energy_op_vec)
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
            nlinks = self.cfg.lattice.nlinks
            el_energy_link_vec = xnp.prod(self.el_energy_op_vec, axis=1)  # complex product over layers
            el_energy_link_vec = xnp.sum(el_energy_link_vec, axis=0)  # sum over group elements (complex)

            self._el_energy_op = xnp.real((nlinks / len(self.cfg.mod_link_inds)) * xnp.sum(el_energy_link_vec))
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

    def _overlap_chi_per_layer(self, gamma_in_vec: xnp.ndarray) -> xnp.ndarray:
        """chi_lay = Pf(M1+M2) Pf(Delta+M0) for each layer, given a full-system gauged link
        covariance `gamma_in_vec` (shape (nlayer, dim, dim)). Returns complex (nlayer,)."""

        mat_d = self.mat_d_vec
        dim = gamma_in_vec.shape[-1]
        m0 = xnp.asarray(ov.vacuum_covmat(dim))
        chis = []
        for lay in range(self.cfg.nlayer):
            # Convert our covariance matrices to Bravyi's convention: M = -S Gamma S with
            # S = diag(1,-1,...) flipping every second Majorana (our gamma^(2) = -c_Bravyi).
            # See ov.to_bravyi_covmat. M0 is already supplied in Bravyi's convention
            # (which is identical to ours in this case).
            m1 = ov.to_bravyi_covmat(gamma_in_vec[lay])
            m2 = ov.to_bravyi_covmat(mat_d[lay])
            chis.append(ov.gaussian_overlap_chi(m1, m2, m0))
        return xnp.array(chis)

    def _overlap_gamma_in_with_modified_link(self, link_ind: int, gauge_val: xnp.ndarray) -> xnp.ndarray:
        """Fresh copy of gamma_in_sys_vec with link `link_ind`'s block rebuilt for `gauge_val`.
        Mirrors _update_gauge_ind's block construction but touches no trackers. No Woodbury."""
        ind_mat = 2 * self.cfg.nvirtmodes_link * link_ind
        coord, dir = self.cfg.lattice.ind2coord_dir(link_ind)
        rotmat = self.generate_rotmat(self.cfg.ncopy, gauge_val, coord, dir)
        gamma = xnp.array(self.gamma_in_sys_vec)  # copy (nlayer, dim, dim)
        blockdim = rotmat.shape[0]
        for layer in range(self.cfg.nlayer):
            gamma_neutral = self.gamma_gauge_neutral_vec[layer][dir]
            block = rotmat @ gamma_neutral @ xnp.transpose(rotmat)
            inds = (layer, slice(ind_mat, ind_mat + blockdim), slice(ind_mat, ind_mat + blockdim))
            gamma = backend.array_assign(gamma, inds, block)
        return gamma

    def _overlap_el_op_vec_for_elements(self, elements: tuple) -> xnp.ndarray:
        """Per-(element, layer, measured-link) cross-overlap ratio conj(chi_lay(G')/chi_lay(G)),
        where G' is the current config with link q's gauge value right-multiplied by `element`.
        Shape (len(elements), nlayer, n_el_links). Drop-in for el_energy_op_vec."""
        chi_G = self._overlap_chi_per_layer(self.gamma_in_sys_vec)  # (nlayer,)
        out = []
        for h in elements:
            per_link = []
            for q in self.cfg.mod_link_inds:
                g_q = self.gaugefieldvec[q]
                g_prime = xnp.asarray(g_q) @ xnp.asarray(h)  # g_q h
                gamma_prime = self._overlap_gamma_in_with_modified_link(q, g_prime)
                chi_Gp = self._overlap_chi_per_layer(gamma_prime)  # (nlayer,)
                per_link.append(xnp.conj(chi_Gp / chi_G))  # (nlayer,)
            out.append(xnp.transpose(xnp.array(per_link), (1, 0)))  # (nlayer, n_el_links)
        return xnp.array(out)  # (n_h, nlayer, n_el_links)

    def _compute_el_energy_op_vec_overlap(self) -> xnp.ndarray:
        """Overlap drop-in for el_energy_op_vec, summed over the standard electric-energy group elements."""
        # Pure gauge only.
        if self.cfg.num_fermionic_layer != 0:
            raise NotImplementedError("The overlap electric-energy path supports pure gauge only.")
        return self._overlap_el_op_vec_for_elements(self.cfg.gaugemgr.group_elements_for_el_energy)

    # Functions that return the layer-resolved energies of each energy operator
    @property
    def el_energy_op_vec(self) -> xnp.ndarray:
        """Compute electric energy operator w/o shift for all layers for the whole system.
        This is a get function.

        Returns:
            array: Layer-resolved electric energy w/o shift of shape (nlayer, len(self.cfg.mod_link_inds))
        """
        if self._el_energy_op_vec is None:
            if getattr(self.cfg, "el_method", "pfaffian") == "overlap":
                self._el_energy_op_vec = self._compute_el_energy_op_vec_overlap()
            else:
                self._el_energy_op_vec = self._compute_el_energy_op_vec(
                    self.lognormvec_default,
                    self.cfg.mod_link_inds,
                    self.cfg.nlayer,
                    self.el_pfaffians,
                    self.norm_mod_vec,
                    self.cfg.gaugemgr.group_elements_for_el_energy,
                    self.cfg.uniq_coeffs_vec,
                    self.cfg.constants_vec,
                )
        return self._el_energy_op_vec

    @property
    def mass_energy_op_vec(self) -> xnp.ndarray:
        """Compute mass energy operator w/o shift for all layers for the whole system.
        This is a get function.

        Returns:
            array: Layer-resolved mass energy w/o shift
        """
        if self._mass_energy_op_vec is None:
            self._mass_energy_op_vec = self._compute_mass_energy_op_vec(
                self.occupations_after_ph,
                use_trans_inv=False,
            )
        return self._mass_energy_op_vec

    @property
    def int_energy_op_vec(self) -> xnp.ndarray:
        """Compute interaction energy operator w/o shift for all layers for the whole system.
        This is a get function.

        Returns:
            array: Layer-resolved interaction energy w/o shift
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
            array: Layer-resolved interaction energy w/o shift
        """
        if self._chem_energy_op_vec is None:
            self._chem_energy_op_vec = self._compute_chem_energy_op_vec(
                self.occupations_before_ph,
            )
        return self._chem_energy_op_vec

    def deriv_mod_mats(self, inds: tuple) -> tuple[xnp.ndarray, xnp.ndarray, xnp.ndarray]:
        """Extract the derivatives of the modified covmats.

        Args:
            inds (tuple): indices to use to mask out the zeroed params.
                The mask is 1 where the parameter is not zeroed, and 0 where it is zeroed.

        Returns:
            tuple: Tuple of the derivatives of the modified covmats (d_mat_a_mod, d_mat_b_mod, d_mat_d_mod)
                These would have shape (nlayer, nmodlinks, unitcell_size, n_symbols, dim1, dim2) if not masked by inds.
                However, the masking changes the shape to (num_active, dim1, dim2), where num_active comes from
                collapsing the leading dimensions.
        """

        if self._deriv_mod_mats is None:
            # each of shape: (nlayer, nmodlinks, unitcell_size, n_symbols, dim1, dim2)
            # where dim1, dim2 are the (effective) physical and virtual dimensions as appropriate
            d_mat_a_mod_vec, d_mat_b_mod_vec, d_mat_d_mod_vec = utils.extract_mod_covmats(
                self.gamma_maj_sys_deriv_layvec_ucvec_symbvec,
                self.cfg.mod_link_inds,
                self.cfg.lattice.size,
                self.cfg.nphysmodes_site,
                self.cfg.nvirtmodes_link,
                ax=1,
            )

            # Keep only the indices corresponding to the non-zeroed params
            # (It might be better to do this before extract_mod_covmats(), since that will speed up the extraction,
            # but since this only runs once per evaluation on a given set of params, it doesn't matter much)
            l, m, u, s = inds  # layer, mod_link, unitcell, symbol
            dA = d_mat_a_mod_vec[l, m, u, s]
            dB = d_mat_b_mod_vec[l, m, u, s]
            dD = d_mat_d_mod_vec[l, m, u, s]
            self._deriv_mod_mats = (dA, dB, dD)
        return self._deriv_mod_mats

    def dmatd_trace_sparse(self, inds: tuple) -> tuple[xnp.ndarray, xnp.ndarray, xnp.ndarray]:
        """Precompute the sparse (nonzero) structure of d_mat_d for the electric-gradient trace.

        d_mat_d is a parameter-derivative, so it is INDEPENDENT of the gauge configuration and very
        sparse (~1-3% nonzero for D6). The electric-gradient trace Tr(d_mat_d_a @ prod_a) therefore
        only needs prod_a at the nonzero positions of d_mat_d_a. We precompute those positions and
        values ONCE per evaluation (paramvec is fixed during an MC run) and reuse them every step.

        Returns padded rectangular arrays (num_active, kmax): row indices, col indices, values.
        Padding entries have value 0 (index 0), so they contribute nothing to the trace.

        Args:
            inds (tuple): the (l, m, u, s) mask indices of the non-zeroed params (as in deriv_mod_mats).
        """
        if self._dmatd_trace_sparse is None:
            dD = np.asarray(self.deriv_mod_mats(inds)[2])  # (num_active, D, D), config-independent
            na = dD.shape[0]
            nzs = [np.argwhere(np.abs(dD[a]) > 1e-14) for a in range(na)]
            kmax = max((nz.shape[0] for nz in nzs), default=1)
            ii = np.zeros((na, kmax), dtype=np.int64)
            jj = np.zeros((na, kmax), dtype=np.int64)
            vals = np.zeros((na, kmax))
            for a, nz in enumerate(nzs):
                kk = nz.shape[0]
                ii[a, :kk], jj[a, :kk] = nz[:, 0], nz[:, 1]
                vals[a, :kk] = dD[a, nz[:, 0], nz[:, 1]]
            self._dmatd_trace_sparse = (xnp.asarray(ii), xnp.asarray(jj), xnp.asarray(vals))
        return self._dmatd_trace_sparse

    # Functions that return the layer-resolved gradients of each energy operator
    @property
    def el_energy_op_grad_vec(self) -> xnp.ndarray:
        """Compute the gradient of the electric operator (w/o shift) for all layers

        Returns:
            array: List of all electric energy gradients (w/o shift)
        """
        if self._el_energy_op_grad_vec is None:

            l, m, u, s = self.mod_mask_inds  # layer, mod_link, unitcell, symbol

            # each of shape: (nlayer, nmodlinks, unitcell_size, n_symbols, dim1, dim2)
            d_mat_a_vec_vec, d_mat_b_vec_vec, d_mat_d_vec_vec = self.deriv_mod_mats((l, m, u, s))
            # Sparse structure of d_mat_d for the electric-gradient trace (config-independent, cached).
            sp_ii, sp_jj, sp_vals = self.dmatd_trace_sparse((l, m, u, s))

            rotmat_vec = xnp.stack(
                [
                    self.generate_rotmat(
                        self.cfg.ncopy, self._gaugefieldvec[link_ind], *self.cfg.lattice.ind2coord_dir(link_ind)
                    )
                    for link_ind in self.cfg.mod_link_inds
                ]
            )

            self._el_energy_op_grad_vec = self._compute_el_grad_vec(
                self.cfg.lattice.size,
                self.cfg.num_pg_layer,
                self.cfg.num_fermionic_layer,
                self.cfg.unitcell_size,
                self.cfg.nvirtmodes_link,
                self.cfg.nphysmodes_site,
                self.cfg.mod_link_inds,
                tuple(self.cfg.symbolvec),
                self.el_energy_op_vec,
                self.mat_b_mod_vec,
                self.covmat_out_mod_vec,
                self.el_pfaffians,
                self.norm_mod_vec,
                self.lognormvec_default,
                self.wi_gamma_out_mod_vec,
                d_mat_a_vec_vec,
                d_mat_b_vec_vec,
                d_mat_d_vec_vec,
                self.grad_over_norm_vec,
                (l, m, u, s),
                self.cfg.gaugemgr.group_elements_for_el_energy,
                self.cfg.uniq_idx_vec,
                self.cfg.uniq_coeffs_vec,
                rotmat_vec,
                sp_ii,
                sp_jj,
                sp_vals,
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

    @staticmethod
    @abstractmethod
    @maybe_jit(static_argnames=["after_ph"])
    def occupation(covmat: xnp.ndarray, site: int, site_coord: tuple[int, int], after_ph: bool = False) -> float:
        """Compute the occupation number for the given layer and site.
        # TODO: make this method also support specification of flavor, for use with non-Abelian gauge groups

        Args:
            covmat (ndarray): The covariance matrix for the layer
            site (int): Site index
            site_coord (tuple): Coordinate of the site. Provided to remove reliance on lattice inside jitted function.
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
                site_coord = self.cfg.lattice.ind2coord(site)
                layer_val += self.occupation(self.ferm_covmat_vec[lay], site, site_coord, after_ph=after_ph)
            total_occ.append(layer_val / self.cfg.lattice.size)
        return xnp.array(total_occ)

    @property
    def occupations_before_ph(self) -> xnp.ndarray:
        """Compute the occupation number before the particle-hole transformation
        for all layers and sites in the system.

        Returns:
            array: the occupation number for all layers and sites, as a 2D array
        """
        if self._occupations_before_ph is None:

            # Initialize shape
            # this ensures that is has the proper shape even when there are no fermionic layers
            # (which is needed for transposes, etc. higher up in the stack)
            self._occupations_before_ph = xnp.zeros((self.cfg.num_fermionic_layer, self.cfg.lattice.size))

            after_ph = False
            for lay in range(self.cfg.num_pg_layer, self.cfg.nlayer):
                for site in range(self.cfg.lattice.size):
                    site_coord = self.cfg.lattice.ind2coord(site)
                    self._occupations_before_ph = backend.array_assign(
                        self._occupations_before_ph,
                        (lay - self.cfg.num_pg_layer, site),
                        self.occupation(self.ferm_covmat_vec[lay], site, site_coord, after_ph=after_ph),
                    )
        return self._occupations_before_ph

    @property
    def occupations_after_ph(self) -> xnp.ndarray:
        """Compute the occupation number after the particle-hole transformation
        for all layers and sites in the system.

        Returns:
            array: the occupation number for all layers and sites, as a 2D array
        """
        if self._occupations_after_ph is None:

            # Initialize shape
            # this ensures that is has the proper shape even when there are no fermionic layers
            # (which is needed for transposes, etc. higher up in the stack)
            self._occupations_after_ph = xnp.zeros((self.cfg.num_fermionic_layer, self.cfg.lattice.size))

            after_ph = True
            for lay in range(self.cfg.num_pg_layer, self.cfg.nlayer):
                for site in range(self.cfg.lattice.size):
                    site_coord = self.cfg.lattice.ind2coord(site)
                    self._occupations_after_ph = backend.array_assign(
                        self._occupations_after_ph,
                        (lay - self.cfg.num_pg_layer, site),
                        self.occupation(self.ferm_covmat_vec[lay], site, site_coord, after_ph=after_ph),
                    )
        return self._occupations_after_ph

    def meson_string(self, path: tuple[tuple[int, bool], ...]) -> float:
        """Calculate the value of a meson string given a path.

        Args:
            path (tuple):

        Returns:
            float:
        """
        meson_val = xnp.sum(self._meson_string_vec(path))  # sum over layers/flavors
        return meson_val

    def compute_path(self, path: tuple[tuple[int, bool], ...]) -> float:
        """Compute the trace of group elements along the path.

        Args:
            path (tuple): Tuple of tuples [(index,conj),....]. conj indicates whether the argument should be conjugated.
            This is the case if the link is traversed from right to left or from top to bottom.
        """
        val = self._compute_path(self.cfg.gaugemgr.get_neutral_gauge_value(), self.gaugefieldvec, path)
        return val

    @staticmethod
    @maybe_jit(static_argnames=["path"])
    def _compute_path(
        neutral_gauge_value: xnp.ndarray, gaugefieldvec: xnp.ndarray, path: tuple[tuple[int, bool], ...]
    ) -> float:
        path_product = neutral_gauge_value
        for ind, conj in path:
            if conj:
                path_product = path_product @ xnp.conjugate(xnp.transpose(gaugefieldvec[ind]))
            else:
                path_product = path_product @ gaugefieldvec[ind]
        return xnp.trace(path_product)

    @staticmethod
    @maybe_jit(static_argnames=["lattice_size", "nphysmodes_site", "nlayer"])
    def _compute_ferm_cov(
        lattice_size: int,
        nphysmodes_site: int,
        nlayer: int,
        gamma_out_inv_vec: xnp.ndarray,
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
            covmat = mat_a_vec[layer] + (mat_b_vec[layer] @ gamma_out_inv_vec[layer] @ xnp.transpose(mat_b_vec[layer]))

            ferm_covmat_vec = backend.array_assign(ferm_covmat_vec, layer, covmat)
        return ferm_covmat_vec

    @property
    def ferm_covmat_vec(self) -> xnp.ndarray:
        """Compute the covariance matrix of the fermions in the system for the given layer.
        We calculate it for all layers automatically, even though it is not needed for pure-gauge layers."""
        if self._ferm_covmat_vec is None:
            self._ferm_covmat_vec = self._compute_ferm_cov(
                self.cfg.lattice.size,
                self.cfg.nphysmodes_site,
                self.cfg.nlayer,
                self.wi_gamma_out_vec,
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

    @staticmethod
    def get_single_link_majorana_mode_order_first_copy_then_color(num_copies: int, num_colors: int) -> list:
        """Generate the link-based majorana mode order for a single link. We first order by copy and then by color.
        This is not the order we use in the code. This is just to change the generate_rotmat ordering.

        Returns:
            list: List of strings of the form <mode_letter:majorana mode>_<copy>_<color>
        """

        mode_order = []
        # We demonstrate the order for a single horizontal link -
        for copy in range(1, num_copies + 1):
            for color in range(1, num_colors + 1):
                mode1 = ("l1", copy, color)  # majorana mode l1
                mode_order += [mode1]
            for color in range(1, num_colors + 1):
                mode2 = ("l2", copy, color)  # majorana mode l2
                mode_order += [mode2]
            for color in range(1, num_colors + 1):
                mode1 = ("r1", copy, color)
                mode_order += [mode1]
            for color in range(1, num_colors + 1):
                mode2 = ("r2", copy, color)
                mode_order += [mode2]

        # Convert to a list of strings
        # This was left as a tuple above in case there was ever any use for that format
        mode_order_str = []
        for mode in mode_order:
            mode_str = mode[0] + "_" + str(mode[1]) + "_" + str(mode[2])
            mode_order_str.append(mode_str)

        return mode_order_str
