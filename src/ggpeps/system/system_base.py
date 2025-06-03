from abc import ABC, abstractmethod
from typing import (
    Union,
    Optional,
    List,
)  # used in type hints; this approach might be deprecated in later python versions
from dataclasses import dataclass, field

import sys
import logging
import itertools as it

import sympy
from scipy.linalg import block_diag

import numpy as np
from ggpeps import xnp as xnp

import ggpeps
from ggpeps import gauge, utils
from ggpeps.lattice import Direction, Lattice2D, Lattice3D
from ggpeps.system.global_funcs import *
from ggpeps.modearray import generate_permutation_matrix

logger = logging.getLogger(ggpeps.LOGGER_NAME)

################## Utility Functions ######################


def calculate_lognorm(
    gamma_in_sys_vec: List[xnp.ndarray],
    mat_d_vec: List[xnp.ndarray],
    all_factors: bool = False,
) -> float:
    # This is still the plain formula, without any update mechanism
    normvec = calculate_lognormvec(gamma_in_sys_vec, mat_d_vec, all_factors=all_factors)
    return xnp.sum(normvec)


def calculate_lognormvec_inc(incdet_vec, det_mat_d_vec, n, all_factors: bool = False):
    dest = []
    for ind in range(len(incdet_vec)):
        detval = incdet_vec[ind].det()
        if all_factors:
            detval -= n * xnp.log(2)
            detval += det_mat_d_vec[ind]
        # The factor 0.5 is the sqrt of the formula. We are storing the logarithm of the norm.
        # The addition of the cumval is the multiplication of the indpendent PEPS
        dest.append(0.5 * detval)
    return xnp.array(dest)


def calculate_lognorm_inc(incdet_vec, det_mat_d_vec, n, all_factors: bool = False):
    lognormvec = calculate_lognormvec_inc(
        incdet_vec, det_mat_d_vec, n, all_factors=all_factors
    )
    return xnp.sum(lognormvec)


@dataclass  # if we upgrade to python 3.10 or higher, add in 'slots=True'
class ElectricEnergyIntermediateVals:
    """Class for keeping track of intermediate calculations of the electric energy,
    for re-use with the gradient calculation"""

    # TODO: make into numpy/jax arrays
    covmat_out_virt_vec: List[xnp.array] = field(
        default_factory=list
    )  # this is the pythonic way to use lists in dataclasses
    norm_mod_vec: List[float] = field(default_factory=list)
    lognorm_default_vec: List[float] = field(default_factory=list)
    pfaffian_vec: List[float] = field(default_factory=list)


################## Config2DBase ######################
class Config2DBase(ABC):
    """Configuration for a system in two dimensions

    This class inherits from the abstract base class to enable abstract methods that have to be overwritten in a child class.
    This class cannot be instantiated directly.
    """

    # Ansatz settings
    # This will be overwritten by the specifications of each ansatz
    _nparams: int = None  # number of params per site per layer
    ncopy: int = None

    def __init__(
        self,
        lattice: Union[Lattice2D, Lattice3D],
        g_el: float,
        g_mag: float,
        g_int: float,
        g_mass: float,
        g_chem: Optional[np.array],
        num_pg_layer: int = 1,
        num_fermionic_layer: int = 0,
    ):
        """Constructor.

        Args:
            lattice (Union[Lattice2D, Lattice3D]): lattice.
            g_el (float): Hamiltonian prefactor for electric energy
            g_mag (float): prefactor for magnetic energy
            g_int (float): prefactor for gauge-matter coupling
            g_mass (float): mass of physical fermions (i.e. prefactor on the mass term).
            g_chem (Optional[np.array]): chemical potential for the fermions. If None, all are set to zero.
            num_pg_layer (int, optional): number of pure gauge layers. Defaults to 1.
            num_fermionic_layer (int, optional): number of fermionic layers. Defaults to 0.
        """
        # The parameters have the following order: [[t1,y1,z1],[t2,y2,z2],....]

        self.lattice = lattice
        self.num_pg_layer = num_pg_layer
        self.num_fermionic_layer = num_fermionic_layer
        self.nlayer = self.num_pg_layer + self.num_fermionic_layer

        self._paramvec: Optional[np.ndarray] = None

        self.zeroed_params: List[int] = (
            []
        )  # will store a list of the parameters forced to be zero by the ansatz
        # currently this is set in self.enforce_parameter_conditions
        # (this only happens for the fermionic ansatz's)

        # Symbolvec
        self._symbolvec: Optional[List[sympy.Symbol]] = (
            None  # the list is just all the symbols, which are the same for each layer (even if for some layers some are forced to zero)
        )

        # Parameters of the Hamiltonian
        self.g_el = g_el
        self.g_mag = g_mag
        self.g_int = g_int
        self.g_mass = g_mass
        self.g_chem = g_chem
        if self.g_chem is None:
            self.g_chem = np.zeros(self.nlayer)
        elif len(self.g_chem) != self.nlayer:
            raise ValueError(
                "The number of chemical potentials must match the number of layers."
            )

    def __str__(self):
        # define a string method that can be used, e.g., in filenaming
        # note that this string doesn't include the number of copies
        return f"L_{self.lattice.nx:02d}x{self.lattice.ny:02d}_gel_{self.g_el}_gmag_{self.g_mag}_gint_{self.g_int}_gmass_{self.g_mass}_nlayer_{self.nlayer}"

    @property
    def paramvec(self) -> np.ndarray:
        return self._paramvec

    @paramvec.setter
    def paramvec(self, val):
        if not isinstance(val, np.ndarray):
            val = np.array(val)
        if self.trans_inv and val.ndim == 2:
            # if the system is translation invariant, we add an extra dimension corresponding to the site index
            val = np.expand_dims(val, axis=1)
        if self.check_params(val):
            self._paramvec = val
            self.nlayer = len(val)
        else:
            logger.error("The set of parameters is not consistent.")
            sys.exit(1)

    def check_params(self, params: np.ndarray) -> bool:
        """Check the consistency of the input parameters.
        All arrays must have the same length.

        Args:
            params (list or np.ndarray): two dimensional array of input parameters
        """
        shape = params.shape
        target_shape = self.param_shape()
        return shape == target_shape

    @property
    def nparams_per_layer(self):
        return self._nparams * self.unitcell_size

    def nvarparams(self):
        return self._nparams * self.unitcell_size * self.nlayer

    def param_shape(self):
        """Return the shape required for valid parameters."""
        shape = (self.nlayer, self.unitcell_size, self._nparams)
        return shape

    def parse_params(self, paramvec, layer, site):
        """Process the parameters and return the parameters for the given layer and site.

        Args:
            paramvec (array): parameters
            layer (int): the layer for which the parameters are needed
            site (int): the site for which the parameters are needed

        Returns:
            array: parameters for the given site and layer (this will be a subarray of paramvec)
        """
        shape = self.param_shape()
        if len(shape) == 2:
            res = paramvec[layer]
        else:
            ind = 0  # TODO: modify this to account for not every site being independent
            res = paramvec[layer][ind]
        return res

    def print_parametervec(self, symbolvec):
        """Printing of the parametervec

        Args:
            symbolvec (list): List of the symbolvecs
        """
        for ind in range(self.nlayer):
            for symb, val in zip(symbolvec, self._paramvec[ind]):
                print(str(symb), val)

    @property
    def trans_inv(self) -> bool:
        """Flag to indicate whether the system is translationally invariant.

        Returns:
            bool: True is ansatz is translationally invariant, False otherwise.
        """
        return self.unitcell_size == 1

    @abstractmethod
    def make_pure_gauge(self):
        """Ensure that the system is pure gauge, i.e. no physical fermions.
        This abstract method must be overwritten by a subclass.
        """
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )

    def enforce_parameter_conditions(self, mat):
        """In some cases, there are extra conditions we wish to impose on the parameters."""
        return

    @abstractmethod
    def _create_symbolvec(self) -> List[sympy.Symbol]:
        """
        Function to define the list of parameters as sympy variables.
        We need these symbols to analytically derive T automatically.
        This function has to be overwritten in the child-class.
        """
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )

    @property
    def symbolvec(self) -> List[sympy.Symbol]:
        """Return the symbolvec.
        This is a get function. It computes the symbolvec only if it does not exist yet.
        If it exists, then it will be returned directly. If not, it will be created and then stored in _symbolvec.

        Returns:
            list: Vector of analytic symbols
        """
        if self._symbolvec is None:
            self._symbolvec = self._create_symbolvec()
        return self._symbolvec

    @property
    @abstractmethod
    def tmat_symb(self):
        """Create the symbolic version of the T matrix.
        This is an abstract function that has to be overwritten by the child class.
        """
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )

    @abstractmethod
    def generate_gamma_gauge_neutral_dict(self):
        """Abstract method to define the ungauged covariance matrix of a single link.
        The substitution method must ensure a consistent order of the modes.
        The direction parameter controls which covariance matrix is retrieved, since these can differ between directions.
        This method must be overwritten in a subclass.
        """
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )


################## System2DBase ######################


class System2DBase(ABC):
    """Base class for two dimensional systems.

    This class inherits from the abstract base class to enable abstract methods that must be overwritten in a child class.
    This class cannot be instantiated directly.
    """

    def __init__(self, cfg: Config2DBase):
        self.cfg: Config2DBase = cfg

        # All variables that contain _vec are arrays of length nlayer in the first dimension.
        # Other types of vec are indicated by layervec, sitevec, etc.

        # Parameter based matrices
        self._tmat_layervec_unitcellvec: Optional[List[List[xnp.ndarray]]] = None
        self._tmat_layervec_sitevec: Optional[List[List[xnp.ndarray]]] = None
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
        self._ferm_covmat_vec: Optional[List[xnp.ndarray]] = None

        # Parameter dependent quantities for the electric energy
        self._mat_a_mod_vec: Optional[xnp.ndarray] = None
        self._mat_b_mod_vec: Optional[xnp.ndarray] = None
        self._mat_d_mod_vec: Optional[xnp.ndarray] = None
        self._det_mat_d_mod_vec: Optional[xnp.ndarray] = None
        self._mat_d_mod_inv_vec: Optional[xnp.ndarray] = None
        self._electric_energy_intermediate_vals = ElectricEnergyIntermediateVals()

        # Management of the gauge fields
        self._gamma_gauge_neutral_vec_dirs: Optional[xnp.ndarray] = (
            None  # vec for layers (choices of projectors may be different for each layer), dirs for directions
        )
        self._gamma_in_sys_vec: Optional[xnp.ndarray] = (
            None  # in cases when different layers use the same projectors, all elements will point to the same gamma_in_sys
        )
        neutral_gauge = self.cfg.gaugemgr.get_neutral_gauge_value()
        self._gaugefieldvec: xnp.ndarray = xnp.array(
            [neutral_gauge] * self.cfg.lattice.nlinks
        )

        # Weight
        self._weight: Optional[float] = None

        # Gradients
        self._gamma_maj_sys_deriv_dict: Optional[
            dict[sympy.Symbol, List[xnp.ndarray]]
        ] = None  # the list is for layers
        self._el_energy_op_grad_vec: Optional[xnp.ndarray] = (
            None  # first index is layer, second index is symbol
        )
        self._mass_energy_op_grad_vec: Optional[xnp.ndarray] = None
        self._int_energy_op_grad_vec: Optional[xnp.ndarray] = None
        self._chem_energy_op_grad_vec = None
        self._d_gamma_out_symbolvec: Optional[List[List[List[xnp.ndarray]]]] = (
            None  # gradients of gamma_out for all symbols: first index is layer, second index uc_ind, third is symbol
        )
        self._grad_over_norm_dict: Optional[
            dict[tuple[int, int, sympy.Symbol], float]
        ] = {
            (lay, uc_ind, symb): None
            for lay, uc_ind, symb in it.product(
                range(self.cfg.nlayer),
                range(self.cfg.unitcell_size),
                self.symbolvec,
            )
        }

        # Observables
        self._energy: Optional[float] = None
        self._el_energy_op: Optional[float] = None
        self._el_energy_op_vec: Optional[List[float]] = None
        self._mag_energy_op: Optional[float] = None
        self._mass_energy_op: Optional[float] = None
        self._mass_energy_op_vec: Optional[List[float]] = None
        self._int_energy_op: Optional[float] = None
        self._int_energy_op_vec: Optional[List[float]] = None
        self._chem_energy_op_vec = None

        # Woodbury Update and Matrix Inversion
        self._wi_gamma_in_vec: Optional[List[utils.WoodburyInverter]] = (
            None  # Tracks (D^-1 - gammain)^-1
        )
        self._wi_gamma_out_vec: Optional[List[utils.WoodburyInverter]] = (
            None  # Tracks (D - gammain)^-1
        )
        self._incdet_vec: Optional[List[utils.IncLogAbsDeterminant]] = (
            None  # Tracks det(D^-1 - gammain)
        )

        self._wi_gamma_in_mod_vec: Optional[List[utils.WoodburyInverter]] = (
            None  # Tracks (Dmod^-1 - gammain)^-1
        )
        self._wi_gamma_out_mod_vec: Optional[List[utils.WoodburyInverter]] = (
            None  # Tracks (Dmod - gammain)^-1
        )
        self._incdet_mod_vec: Optional[List[utils.IncLogAbsDeterminant]] = (
            None  # Tracks det(Dmod^-1 - gammain)
        )

    def initialize(self):
        """Initialization function.
        This is a good spot to copy essential data from the configuration.
        """
        return None

    def invalidate_gauge_update(self):
        """Reset the values of computed quantitities to avoid spillover from previous computations.
        We do not need to reset quantities that are not dependent on the gauge fields, such as _gamma_maj_sys_vec, _mat_a_vec, etc.
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

        self._el_energy_op_grad_vec = None
        self._mass_energy_op_grad_vec = None
        self._int_energy_op_grad_vec = None
        self._chem_energy_op_grad_vec = None
        self._grad_over_norm_dict = {
            (lay, uc_ind, symb): None
            for lay, uc_ind, symb in it.product(
                range(self.cfg.nlayer),
                range(self.cfg.unitcell_size),
                self.symbolvec,
            )
        }
        self._electric_energy_intermediate_vals = ElectricEnergyIntermediateVals()
        return

    def _extract_partial_covmatvec(self, offset: int):
        # We are assuming one physical mode per site

        mat_a_vec = self.gamma_maj_sys_vec[:, :offset, :offset]
        mat_b_vec = self.gamma_maj_sys_vec[:, :offset, offset:]
        mat_d_vec = self.gamma_maj_sys_vec[:, offset:, offset:]
        return mat_a_vec, mat_b_vec, mat_d_vec

    @property
    def symbolvec(self) -> List[sympy.Symbol]:
        """Return the symbolvec.

        Returns:
            list: Vector of analytic symbols
        """
        return self.cfg.symbolvec

    @property
    def tmat_symb(self):
        """Return the symbolic version of the T matrix.
        This is stored in the config for each ansatz, because it is part of the definition of the ansatz,
        and is not specific to a particular state (i.e. particular parameters).
        """
        return self.cfg.tmat_symb

    def compute_tmat_deriv(self, symb: sympy.Symbol):
        """Return the derivative of the T matrix with respect to the symbol

        Args:
            symb (sympy.Symbol): Symbol to be derived with respect to

        Returns:
            xnp.ndarray: Array of symbols
        """
        tmat_symb = self.cfg.tmat_symb
        return xnp.asarray(
            np.asarray(sympy.diff(tmat_symb, symb)).astype(complex)
        )  # convert to numpy array, then to xnp (jax cannot convert from sympy directly)

    def _eval_tmat_symb(self, paramvec):
        """Compute the numerical representation of the T matrix

        Args:
            paramvec (list): List of parameter values (numerical)

        Returns:
            xnp.ndarray: T matrix with numerical values
        """
        tmat_eval = self.cfg.tmat_symb.evalf(
            subs={self.symbolvec[i]: paramvec[i] for i in range(len(paramvec))}
        )
        return xnp.asarray(
            np.asarray(tmat_eval).astype(complex)
        )  # convert to numpy array, then to xnp (jax cannot convert from sympy directly)

    @property
    def tmat_layervec_unitcellvec(self) -> List[List[xnp.ndarray]]:
        if self._tmat_layervec_unitcellvec is None:
            self.cfg.enforce_parameter_conditions(self.cfg.paramvec)
            self._tmat_layervec_unitcellvec = []
            for layer in range(self.cfg.nlayer):
                tmats = [
                    self._eval_tmat_symb(self.cfg.paramvec[layer][ind])
                    for ind in range(self.cfg.unitcell_size)
                ]
                self._tmat_layervec_unitcellvec.append(tmats)
        return self._tmat_layervec_unitcellvec

    @property
    def tmat_layervec_sitevec(self) -> List[List[xnp.ndarray]]:
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
                tmat_lay = [
                    tmats[self.cfg.site_params_dict[site]]
                    for site in range(self.cfg.lattice.size)
                ]
                self._tmat_layervec_sitevec.append(tmat_lay)
            # self._tmat_layervec_sitevec = xnp.array(self._tmat_layervec_sitevec)
        return self._tmat_layervec_sitevec

    @property
    def gamma_dirac_layervec_sitevec(self) -> xnp.ndarray:
        """Return the vector of covariance matrices in dirac modes.

        Returns:
            xnp.ndarray: Vector of covariance matrices in Dirac modes
        """
        if self._gamma_dirac_layervec_sitevec is None:

            self._gamma_dirac_layervec_sitevec = []
            for lay in range(self.cfg.nlayer):

                gamma_dirac_lay = [
                    xnp.array(utils.tmat_to_covariance_matrix(tmat))
                    for tmat in self.tmat_layervec_sitevec[lay]
                ]
                self._gamma_dirac_layervec_sitevec.append(gamma_dirac_lay)

            self._gamma_dirac_layervec_sitevec = xnp.array(
                self._gamma_dirac_layervec_sitevec
            )
        return self._gamma_dirac_layervec_sitevec

    @property
    def gamma_maj_layervec_sitevec(self):
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
            self._gamma_maj_layervec_sitevec = xnp.real(
                smat @ self.gamma_dirac_layervec_sitevec @ xnp.transpose(smat)
            )
        return self._gamma_maj_layervec_sitevec

    def _expand_gamma_maj_to_system(self, covmats_layervec_sitevec):
        """Expand the covariance matrix in Majorana modes to the full system.
        In order to obtain a structure that is convenient for further computations,
            (A    B)
            (-B^T D)
        we have to reorder the modes of the single-vertex matrix with respect to the full matrix.

        This method is overwritten for the U1 system.

        Args:
            covmats_layervec_sitevec (List[List[xnp.ndarray]]): list (per layer) of 2D covariance matrices of all sites; total shape (nlayer, nsites, nmodes, nmodes)

        Returns:
            xnp.ndarray: 2D covariance matrix of the full system
        """

        # Preliminaries
        nsites = self.cfg.lattice.size
        id = xnp.eye(nsites)

        # Build permutation matrix to convert modes from site order to link order
        modes_link_order = self.get_link_based_mode_order()
        modes_site_order = self.get_site_based_mode_order()
        mat_perm_links = generate_permutation_matrix(
            modes_site_order, modes_link_order
        )  # be careful with the convention of the permutation matrix vs its transpose; this way works with the code below.
        sites_perm = xnp.eye(
            2 * self.cfg.lattice.nx * self.cfg.lattice.ny * self.cfg.nphysmodes_site
        )  # total number of physical fermionic majorana modes on all the sites together
        mat_perm = block_diag(sites_perm, mat_perm_links)

        # TODO: properly vectorize!
        gamma_maj_sys_vec = []
        for covmats_sitevec in covmats_layervec_sitevec:
            covmats = covmats_sitevec

            # Extract the parts of the covariance matrix
            offset = (
                2 * self.cfg.nphysmodes_site
            )  # number of physical (majarona) modes per site
            amats = [
                covmats[site][:offset, :offset] for site in range(self.cfg.lattice.size)
            ]
            bmats = [
                covmats[site][:offset, offset:] for site in range(self.cfg.lattice.size)
            ]
            dmats = [
                covmats[site][offset:, offset:] for site in range(self.cfg.lattice.size)
            ]
            # Expand them
            amat_sys = block_diag(*amats)
            bmat_sys = block_diag(*bmats)
            dmat_sys = block_diag(*dmats)
            # Reassemble them in the correct order
            mat_sys_unordered = xnp.block(
                [[amat_sys, bmat_sys], [-xnp.transpose(bmat_sys), dmat_sys]]
            )
            dest = xnp.transpose(mat_perm) @ mat_sys_unordered @ mat_perm
            gamma_maj_sys_vec.append(dest)
        return xnp.array(gamma_maj_sys_vec)

    ## MOVE TO GLOBAL
    def d_gamma_out_symbolvec(self, layer: int, uc_ind: int):
        """Return a vector containing the derivatives of gamma_out (for the given layer) for each symbol.

        Returns:
            [List]: List of xnp.ndarrays, with length equal to the number of symbols.
        """
        if self._d_gamma_out_symbolvec is None:
            self._d_gamma_out_symbolvec = [None] * self.cfg.nlayer
        if self._d_gamma_out_symbolvec[layer] is None:
            self._d_gamma_out_symbolvec[layer] = [None] * self.cfg.unitcell_size
        if self._d_gamma_out_symbolvec[layer][uc_ind] is None:
            self._d_gamma_out_symbolvec[layer][uc_ind] = []
            offset = 2 * self.cfg.lattice.size * self.cfg.nphysmodes_site

            for symbol in self.symbolvec:
                mat_b = self.mat_b_vec[layer]
                deriv_gamma_maj_sys = self.gamma_maj_sys_deriv_vec(symbol)[
                    layer, uc_ind
                ]
                d_mat_a, d_mat_b, d_mat_d = extract_partial_covmats(
                    deriv_gamma_maj_sys, offset
                )
                diff_d_gamma_inv = self.wi_gamma_out_vec[layer].inv()
                d_gamma_out = (
                    d_mat_a
                    + d_mat_b @ diff_d_gamma_inv @ xnp.transpose(mat_b)
                    + mat_b @ diff_d_gamma_inv @ xnp.transpose(d_mat_b)
                    - mat_b
                    @ diff_d_gamma_inv
                    @ d_mat_d
                    @ diff_d_gamma_inv
                    @ xnp.transpose(mat_b)
                )
                self._d_gamma_out_symbolvec[layer][uc_ind].append(d_gamma_out)

        return self._d_gamma_out_symbolvec[layer][uc_ind]

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
            self._gamma_maj_sys_vec = self._expand_gamma_maj_to_system(
                self.gamma_maj_layervec_sitevec
            )
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
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = (
                self._extract_partial_covmatvec(offset)
            )
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
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = (
                self._extract_partial_covmatvec(offset)
            )
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
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = (
                self._extract_partial_covmatvec(offset)
            )
        return self._mat_d_vec

    @property
    def det_mat_d_vec(self):
        """Compute the determinant of the virtual-virtual correlation matrix.
        There is a vector of D matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            xnp.ndarray: List of log-determinants
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

    @property
    def mat_a_mod_vec(self):
        """Extract the matrix for physical-physical correlations and one virtual mode.
        This shifted matrix is used for the computation of the electric energy.
        The mode ordering of this matrix is (p_1(0,0),p_2(0,0),p_1(1,0),p_2(1,0)....).
        It is formulated in terms of Majorana modes.
        The mode ordering of the sites is identical to the site convention defined in the lattice class.
        There is a vector of A matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            [xnp.ndarray]: Correlations of the physcial modes for the full system.
        """
        if self._mat_a_mod_vec is None:
            offset = (
                2 * self.cfg.lattice.size * self.cfg.nphysmodes_site
            )  # offset for physical modes
            offset += (
                2 * self.cfg.nvirtmodes_link
            )  # offset for virtual modes of the link
            self._mat_a_mod_vec, self._mat_b_mod_vec, self._mat_d_mod_vec = (
                self._extract_partial_covmatvec(offset)
            )
        return self._mat_a_mod_vec

    @property
    def mat_b_mod_vec(self):
        """Extract the matrix for physical-virtual correlations.
        This matrix contains one link less than the original matrix (used for the electric energy computation)
        There is a vector of B matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            [xnp.ndarray]: Correlations of the physcial modes with the virtual modes for the full system.
        """
        if self._mat_b_mod_vec is None:
            offset = (
                2 * self.cfg.lattice.size * self.cfg.nphysmodes_site
            )  # offset for physical modes
            offset += (
                2 * self.cfg.nvirtmodes_link
            )  # offset for virtual modes of the link
            self._mat_a_mod_vec, self._mat_b_mod_vec, self._mat_d_mod_vec = (
                self._extract_partial_covmatvec(offset)
            )
        return self._mat_b_mod_vec

    @property
    def mat_d_mod_vec(self):
        """Extract the matrix for virtual-virtual correlations.
        This matrix contains one link less than the original matrix (used for the electric energy computation)
        There is a vector of D matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            [xnp.ndarray]: Correlations of the virtual modes for the full system.
        """
        if self._mat_d_mod_vec is None:
            offset = (
                2 * self.cfg.lattice.size * self.cfg.nphysmodes_site
            )  # offset for physical modes
            offset += (
                2 * self.cfg.nvirtmodes_link
            )  # offset for virtual modes of the link
            self._mat_a_mod_vec, self._mat_b_mod_vec, self._mat_d_mod_vec = (
                self._extract_partial_covmatvec(offset)
            )
        return self._mat_d_mod_vec

    @property
    def det_mat_d_mod_vec(self):
        """
        Compute the determinant of the virtual-virtual correlation matrix for the modified matrix.
        There is a vector of determinants of D matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            list: list of log-determinants
        """
        if self._det_mat_d_mod_vec is None:
            _, self._det_mat_d_mod_vec = xnp.linalg.slogdet(self.mat_d_mod_vec)
        return self._det_mat_d_mod_vec

    @property
    def mat_d_mod_inv_vec(self):
        """Compute the inverse of modified D matrices.
        There is a vector of inverses of D matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            xnp.ndarray: List of inverses of modified D matrices
        """
        if self._mat_d_mod_inv_vec is None:
            self._mat_d_mod_inv_vec = xnp.linalg.inv(self.mat_d_mod_vec)
        return self._mat_d_mod_inv_vec

    @abstractmethod
    def initialize_gamma_in_sys(self):
        """Abstract function to initialize gamma_in (the covariance matrix of the projectors) in a child class;
        this function has to be overwritten in a child class.

        This function returns gamma_in_sys_vec even for cases where gamma_in_sys does not vary between layers.
        In that case, each element of gamma_in_sys_vec points to the same gamma_in_sys.
        """
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )

    @property
    def gamma_in_sys(self):
        """Get function to return the gauged gamma_in_sys, the covariance matrix of the links for the whole system.
        This is required to maintain compatibility with early development, in which gamma_in did not vary between layers.
        Possibly the code should be modified to use gamma_in_sys_vec everywhere; this can be done without significant memory cost.

        Returns:
            xnp.ndarray: Gauged covariance matrix of the system
        """
        if self._gamma_in_sys_vec is None:
            self._gamma_in_sys_vec, full_tuple, mod_tuple = (
                self.initialize_gamma_in_sys()
            )
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            (
                self._wi_gamma_in_mod_vec,
                self._wi_gamma_out_mod_vec,
                self._incdet_mod_vec,
            ) = mod_tuple
        return self._gamma_in_sys_vec[0]

    @property
    def gamma_in_sys_vec(self):
        """Get function to return the gauged gamma_in_sys_vec, the covariance matrices of the links for the whole system for each layer.
        This function is required to allow for gamma_in to vary between layers.

        Returns:
            xnp.ndarray: vector of gauged covariance matrices of the system
        """
        if self._gamma_in_sys_vec is None:
            self._gamma_in_sys_vec, full_tuple, mod_tuple = (
                self.initialize_gamma_in_sys()
            )
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
            self._gamma_in_sys_vec, full_tuple, mod_tuple = (
                self.initialize_gamma_in_sys()
            )
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
            self._gamma_in_sys_vec, full_tuple, mod_tuple = (
                self.initialize_gamma_in_sys()
            )
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
            self._gamma_in_sys_vec, full_tuple, mod_tuple = (
                self.initialize_gamma_in_sys()
            )
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            (
                self._wi_gamma_in_mod_vec,
                self._wi_gamma_out_mod_vec,
                self._incdet_mod_vec,
            ) = mod_tuple
        return self._wi_gamma_out_vec

    @property
    def gamma_in_sys_mod(self):
        """Get function to return the gauged gamma_in_sys with a single link modification (to compute the electric energy),
        the covariance matrix of the links for the whole system.

        Returns:
            xnp.ndarray: Gauged, modified covariance matrix of the system
        """
        single_link_offset = 2 * self.cfg.nvirtmodes_link
        # return self.gamma_in_sys[single_link_offset:, single_link_offset:] # TODO: fix for JAX - DONE
        return gamma_in_sys_mod(self.gamma_in_sys, single_link_offset)

    @property
    def gamma_in_sys_mod_vec(self):
        """Get function to return the gauged gamma_in_sys_vec with a single link modification (to compute the electric energy),
        the covariance matrix of the links for the whole system.

        Returns:
            xnp.ndarray: Gauged, modified covariance matrices of the system for each layer
        """
        gamma_in_sys_mod_vec = []
        single_link_offset = (
            2 * self.cfg.nvirtmodes_link
        )  # we can use the same offset for all layers, since all dimensions, mode ordering, etc. are the same
        for layer in range(self.cfg.nlayer):
            gamma_in_sys_mod_vec.append(
                self.gamma_in_sys_vec[layer][single_link_offset:, single_link_offset:]
            )
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
            self._gamma_in_sys_vec, full_tuple, mod_tuple = (
                self.initialize_gamma_in_sys()
            )
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
            self._gamma_in_sys_vec, full_tuple, mod_tuple = (
                self.initialize_gamma_in_sys()
            )
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
            self._gamma_in_sys_vec, full_tuple, mod_tuple = (
                self.initialize_gamma_in_sys()
            )
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
        d_lt = (
            idttinv_minus @ d_idtt_minus @ idttinv_minus @ tmat
            - idttinv_minus @ deriv_t
        )
        d_rt = (
            0.5 * idttinv_minus @ d_idtt_plus @ idttinv_minus @ idtt_plus
            + 0.5 * idttinv_minus @ d_idtt_plus
        )
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

    def _generate_gamma_maj_sys_deriv_dict(self):
        """Internal function to generate a dictionary of all possible derivatives of gamma_maj_sys, the system-wide
        covariance matrix of the fiducial state.
        The key to the dictionary is the symbol with respect to which we derived.
        Each entry contains a list with len(list) = nlayer.

        Returns:
            dict: Dictionary with all derivatives
        """
        # TODO: should we save the computations here in private variables (as done elsewhere)?
        dest = {}
        for symb in self.symbolvec:
            arr = []
            for lay in range(self.cfg.nlayer):
                uc_vec = []
                for uc_ind in range(self.cfg.unitcell_size):
                    gamma_maj_deriv = self.compute_gamma_maj_deriv(symb, lay, uc_ind)

                    gamma_maj_derivs_sitevec = []
                    for site in range(self.cfg.lattice.size):
                        if self.cfg.site_params_dict[site] == uc_ind:
                            gamma_maj_derivs_sitevec.append(gamma_maj_deriv)
                        else:
                            gamma_maj_derivs_sitevec.append(
                                xnp.zeros_like(gamma_maj_deriv)
                            )

                    gamma_maj_sys_derivs = self._expand_gamma_maj_to_system(
                        [gamma_maj_derivs_sitevec]
                    )[0]
                    uc_vec.append(gamma_maj_sys_derivs)
                arr.append(uc_vec)

            dest[symb] = xnp.array(arr)
        return dest

    def gamma_maj_sys_deriv_vec(self, symb: sympy.Symbol) -> xnp.ndarray:
        """Return a list of derivatives of all layers of gamma_maj_sys with respect to a given symbol.
        This is a get function.

        Args:
            symb (sympy.Symbol): Symbol with repsect to which we derive

        Returns:
            xnp.ndarray: List of derived gamma_maj_sys
        """
        if symb in self.symbolvec:
            if self._gamma_maj_sys_deriv_dict is None:
                self._gamma_maj_sys_deriv_dict = (
                    self._generate_gamma_maj_sys_deriv_dict()
                )
            return self._gamma_maj_sys_deriv_dict[symb]
        else:
            logger.error("gamma_maj_sys_deriv: Invalid variable name.")
        return None

    ## MOVE TO GLOBAL
    def compute_grad_norm_vec(self) -> xnp.ndarray:
        """Compute the gradient of the norm for all layers with respect to all parameters.
        The parameter order is [[dt1, dy1, dz1...],[dt2,dy2,dz2...]...]

        Returns:
            xnp.ndarray: Vector of gradients of the norm with respect to all parameters
        """
        dest = []
        for layerind in range(self.cfg.nlayer):
            layer_grad = []
            for uc_ind in range(self.cfg.unitcell_size):
                layer_grad.append(self.compute_grad_norm(layerind, uc_ind))
            dest.append(layer_grad)
        dest = xnp.asarray(dest)

        # Enforce ansatz conditions on the gradients
        self.cfg.enforce_parameter_conditions(dest)
        return dest

    ## MOVE TO GLOBAL
    def compute_grad_norm(self, layerind: int, uc_ind: int) -> xnp.ndarray:
        """Compute the gradient of the norm for a given layer wrt to all parameters.
        The parameter order is the same as in the symbolvec

        Args:
            layerind (int): layer index
            uc_ind (int): unit cell index

        Returns:
            xnp.ndarray: Vector of gradients for the norm
        """

        dest_grad = xnp.zeros(len(self.symbolvec), dtype=xnp.float64)
        for symbol_ind, symbol in enumerate(self.symbolvec):
            if (layerind, uc_ind, symbol_ind) not in self.cfg.zeroed_params:
                # the derivative calculation is computationally expensive
                # we can skip it for parameters that are forced by the ansatz to be zero

                # Compute gradient
                if ggpeps.PREFERRED_BACKEND == "jax":
                    dest_grad = dest_grad.at[symbol_ind].set(
                        self.compute_grad_over_norm(symbol, layerind, uc_ind)
                    )
                else:
                    dest_grad[symbol_ind] = self.compute_grad_over_norm(
                        symbol, layerind, uc_ind
                    )
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

    ## MOVE TO GLOBAL
    def calculate_weight_attempt(self, link_ind: int, theta: float, all_factors=False):
        """
        For D2n gauge groups, we overwrite this function in system implementation.

        Compute the weight of an update attempt in which the link index link_ind is substituted for theta
        The inclusion of all constant pre-factors can be switched on and off.

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
        rotmat = self.generate_rotmat(theta, coord, dir)
        gamma_neutral_gauge_vec = self.gamma_gauge_neutral_vec
        gamma_in_subst_layers = [
            rotmat @ gamma_neutral_gauge[dir] @ xnp.transpose(rotmat)
            for gamma_neutral_gauge in gamma_neutral_gauge_vec
        ]
        updates = [
            self.calculate_update_gamma_in(ind_mat, gamma_in_subst, gamma_in_sys)
            for gamma_in_subst, gamma_in_sys in zip(
                gamma_in_subst_layers, self.gamma_in_sys_vec
            )
        ]
        return self.update_lognorm_inc(ind_mat, updates, all_factors)

    def calculate_lognorm(self, all_factors=False):
        """Compute the logarithm of the norm

        Args:
            all_factors (bool, optional): Include all constant prefactors. Defaults to False.

        Returns:
            float: Logarithm of the norm
        """
        return calculate_lognorm(
            self.gamma_in_sys_vec, self.mat_d_vec, all_factors=all_factors
        )

    def calculate_lognormvec(self, all_factors=False):
        """Compute the logarithm of the norm for each layer

        Args:
            all_factors (bool, optional): Include all constant prefactors. Defaults to False.

        Returns:
            float: Logarithm of the norm
        """
        return calculate_lognormvec(
            self.gamma_in_sys_vec, self.mat_d_vec, all_factors=all_factors
        )

    def calculate_lognormvec_inc(self, all_factors=False):
        """Compute the logarithm of the norm for all layers by incrementally updating the previous value (using IncDet and Woodbury)

        Args:
            all_factors (bool, optional): Include all pre-factors in the computation. Defaults to False.

        Returns:
            list: Vector of the incrementally updated norms for all layers
        """
        return calculate_lognormvec_inc(
            self.incdet_vec,
            self.det_mat_d_vec,
            self.gamma_in_sys.shape[0],
            all_factors=all_factors,
        )

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
                detval -= self.gamma_in_sys.shape[0] * xnp.log(2)
                detval += xnp.linalg.slogdet(self.mat_d_vec[ind])[1]
            # The factor 0.5 is the sqrt of the formula. We are storing the logarithm of the norm.
            # The addition of the cumval is the multiplication of the indpendent PEPS
            cumval += 0.5 * detval
        return cumval

    ## MOVE TO GLOBAL
    def compute_grad_over_norm(
        self, var: sympy.Symbol, layerind: int, uc_ind: int
    ) -> float:
        """Compute the quotient of derivative of the norm over the norm itself.
        We can avoid a lot of factors by computing the quotient directly.

        Args:
            var (sympy.Symbol): Name of the variable
            layerind (int): Index of the layer

        Returns:
            float: Value of the gradient divided by the norm of the state
        """
        if self._grad_over_norm_dict[(layerind, uc_ind, var)] is None:
            diff = self.wi_gamma_in_vec[layerind].inv()
            # 2 phys. Majorana modes per vertex, this is indepent of the number of copies or layers
            offset = 2 * self.cfg.lattice.size * self.cfg.nphysmodes_site
            # Extract only the part of the virtual-virtual correlations
            # deriv_d = self.gamma_maj_sys_deriv_vec(var)[layerind][offset:, offset:] # TODO: fix for JAX - DONE
            _, _, deriv_d = extract_partial_covmats(
                self.gamma_maj_sys_deriv_vec(var)[layerind, uc_ind], offset
            )
            mat_d_inv = self.mat_d_inv_vec[layerind]

            # TODO: We might save one matrix-matrix multiplication here
            # The derivd and mat_d_inv are constant
            self._grad_over_norm_dict[(layerind, uc_ind, var)] = compute_grad_over_norm(
                self.gamma_in_sys_vec[layerind], diff, deriv_d, mat_d_inv
            )
        return self._grad_over_norm_dict[(layerind, uc_ind, var)]

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
            self._gamma_gauge_neutral_vec_dirs = (
                self._generate_gamma_gauge_neutral_dict()
            )
        return self._gamma_gauge_neutral_vec_dirs

    def _generate_gamma_gauge_neutral_dict(self):
        """Define the ungauged covariance matrix of a single link.
        The substitution method must ensure a consistent order of the modes.
        The direction parameter controls which covariance matrix is retrieved, since these can differ between directions.
        """
        gamma_gauge_neutral_vec_dirs = self.cfg.generate_gamma_gauge_neutral_dict()
        return xnp.array(gamma_gauge_neutral_vec_dirs)

    @abstractmethod
    def generate_rotmat(self, theta, coord, dir):
        """Abstract method to define the rotation matrix of a single link.
        The substitution method must ensure a consistent order of the modes.
        This method must be overwritten in a subclass.
        """
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )

    @abstractmethod
    def update_gauge_ind(self, link_ind, theta):
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )

    def update_gauge_full_system(self, gaugeconfig):
        """Replace all gauge fields on the links by the values given in gaugeconfig.

        Args:
            gaugeconfig (xnp.ndarray): Array of new values for the gauge field
        """
        for link_ind, gauge in enumerate(gaugeconfig):
            theta = gaugeconfig[link_ind]
            if not xnp.allclose(self._gaugefieldvec[link_ind], theta):
                # only actually do the update if it's a different gauge field
                self.update_gauge_ind(link_ind, gauge)

    def update_gauge_coord(self, coord, dir, theta):
        """Update a gauge field at a given coordinate and direction by a new value

        Args:
            coord (tuple): Coordinate of the vertex
            dir (Direction): Direction of the link
            theta (xnp.array): New value for the gauge field
        """
        ind = self.cfg.lattice.coord2ind_dir(coord, dir)
        self.update_gauge_ind(ind, theta)

    def calculate_update_gamma_in(self, offset, update_mat, gamma_in_sys=None):
        """Compute an update between the current gamma_in and the new gamma_in

        Args:
            offset (int): Offset in the matrix
            update_mat (xnp.ndarray): Array to replace the current content of gamma_in at offset
            gamma_in_sys (xnp.ndarray): gamma_in_sys. This is given as an argument so that different gamma_in_sys can be passed in when gamma_in_sys differs between layers.

        Returns:
            xnp.ndarray: Additional update to reach update_mat at gamma_in[offset:,offset:]
        """
        if gamma_in_sys is None:
            gamma_in_sys = (
                self.gamma_in_sys
            )  # take the first element, which is shared between all the layers
        m_up, n_up = update_mat.shape
        gamma_in_old = slice_matrix(
            gamma_in_sys, offset, offset + m_up, offset, offset + n_up
        )
        # gamma_in_sys[offset:offset + m_up, offset:offset + n_up] # TODO: fix for JAX - DONE
        return -(update_mat - gamma_in_old)

    ################## Observables ######################
    @abstractmethod
    def _compute_mag_energy_op(self):
        """Compute the bare operator (without shift) of the magnetic energy.
        This is an abstract method and has to be overwritten in a subclass.
        """
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )

    @abstractmethod
    def _compute_mass_energy_op_vec_and_grad(self):
        """Compute the mass energy and the gradient (per layer).
        This is an abstract method and has to be overwritten in a subclass.
        """
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )

    @abstractmethod
    def _compute_int_energy_op_vec_and_grad(self):
        """Compute the interaction energy and the gradient (for a single layer).
        This is an abstract method and has to be overwritten in a subclass.
        """
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )

    @abstractmethod
    def _compute_chem_energy_op_vec_and_grad(self):
        """Compute the chemical potential energy and the gradient (per layer).
        This is an abstract method and has to be overwritten in a subclass.
        """
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )

    def _compute_el_energy_op_vec(self, use_trans_inv: bool = True):
        """Compute the electric energy.
        This is an abstract method and has to be overwritten in a subclass.
        """
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )

    def _compute_el_grad_vec(self, use_trans_inv: bool = True):
        """Compute the electric energy gradients.
        This is an abstract method and has to be overwritten in a subclass.
        """
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )

    def _meson_string_vec(self, path):
        """Compute a meson string.
        This is an abstract method and has to be overwritten in a subclass.
        """
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )

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
            if not utils.isclose(self.cfg.g_el, 0):
                self._energy += self.el_energy
            if not utils.isclose(self.cfg.g_mag, 0):
                self._energy += self.mag_energy
            if not utils.isclose(self.cfg.g_mass, 0):
                self._energy += self.mass_energy
            if not utils.isclose(self.cfg.g_int, 0):
                self._energy += self.int_energy
            if not np.allclose(self.cfg.g_chem, 0):
                self._energy += self.chem_energy
        return self._energy

    # Functions that return a term of the energy in the Hamiltonian, including all prefactors and energy from the entire lattice.
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
    def mag_energy(self) -> float:
        """Compute magnetic energy with shift for the whole system
        This is a get function.

        Returns:
            float: magnetic energy
        """
        nplaq = self.cfg.lattice.nplaquettes
        mag_energy = (
            self.cfg.g_mag * 2 * (nplaq - self.mag_energy_op)
        )  # The 2 is for the hermitian conjugate
        return mag_energy

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
    def chem_energy(self):
        """Compute chemical potential energy for the whole system
        This is a get function.

        Returns:
            float: chemical potential energy
        """
        chem_energy = 0.0
        for layer in range(self.cfg.nlayer):
            chem_energy += self.cfg.g_chem[layer] * self.chem_energy_op_vec[layer]
        return chem_energy

    # Functions that return the energy for the operator part of a term in the Hamiltonian, including the energy for the entire lattice, but not any shifts or prefactors.
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
            self._el_energy_op = nlinks * xnp.prod(self.el_energy_op_vec)
        return self._el_energy_op

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
    def mass_energy_op(self) -> float:
        """Compute the mass energy operator for the whole system without shift.
        This is a get function.

        Returns:
            float: Mass energy operator (w/o shift) for the whole system
        """
        if self._mass_energy_op is None:
            nsites = self.cfg.lattice.size
            self._mass_energy_op = xnp.sum(
                self.mass_energy_op_vec
            )  # don't multiply by the number of sites; for the mass term this is assumed to happen lower down in the stack.
        return self._mass_energy_op

    @property
    def int_energy_op(self) -> float:
        """Compute the interaction energy operator for the whole system without shift.
        This is a get function.

        Returns:
            float: Interaction energy operator (w/o shift) for the whole system
        """
        if self._int_energy_op is None:
            nsites = self.cfg.lattice.size
            self._int_energy_op = xnp.sum(self.int_energy_op_vec)
        return self._int_energy_op

    # Functions that return the layer-resolved energies of each energy operator
    @property
    def el_energy_op_vec(self):
        """Compute electric energy operator w/o shift for all layers for the whole system.
        This is a get function.

        Returns:
            list: Layer-resolved electric energy w/o shift
        """
        if self._el_energy_op_vec is None:
            # This vector is the electric energy on a single link.
            # Otherwise, we get a power of nlinks in the product and the electric energy term (with prefactors) gets negative
            self._el_energy_op_vec = self._compute_el_energy_op_vec()
        return self._el_energy_op_vec

    @property
    def mass_energy_op_vec(self):
        """Compute mass energy operator w/o shift for all layers for the whole system.
        This is a get function.

        Returns:
            list: Layer-resolved mass energy w/o shift
        """
        if self.cfg.g_mass == 0:
            self._mass_energy_op_vec = xnp.zeros(self.cfg.nlayer)
        elif self._mass_energy_op_vec is None:
            self._mass_energy_op_vec, self._mass_energy_op_grad_vec = (
                self._compute_mass_energy_op_vec_and_grad()
            )
        return self._mass_energy_op_vec

    @property
    def int_energy_op_vec(self):
        """Compute interaction energy operator w/o shift for all layers for the whole system.
        This is a get function.

        Returns:
            list: Layer-resolved interaction energy w/o shift
        """
        if self.cfg.g_int == 0:
            self._int_energy_op_vec = xnp.zeros(self.cfg.nlayer)
        elif self._int_energy_op_vec is None:
            # This vector is the interaction energy on a single site.
            self._int_energy_op_vec, self._int_energy_op_grad_vec = (
                self._compute_int_energy_op_vec_and_grad()
            )
        return self._int_energy_op_vec

    @property
    def chem_energy_op_vec(self):
        """Compute chemical potential energy operator w/o shift for all layers for the whole system.
        This is a get function.

        Returns:
            list: Layer-resolved interaction energy w/o shift
        """
        if xnp.allclose(self.cfg.g_chem, 0):
            self._chem_energy_op_vec = xnp.zeros(self.cfg.nlayer)

        elif self._chem_energy_op_vec is None:
            self._chem_energy_op_vec, self._chem_energy_op_grad_vec = (
                self._compute_chem_energy_op_vec_and_grad()
            )
        return self._chem_energy_op_vec

    # Functions that return the layer-resolved gradients of each energy operator
    @property
    def el_energy_op_grad_vec(self):
        """Compute the gradient of the electric operator (w/o shift) for all layers

        Returns:
            list: List of all electric energy gradients (w/o shift)
        """
        if self._el_energy_op_grad_vec is None:
            self._el_energy_op_grad_vec = self._compute_el_grad_vec()
        return self._el_energy_op_grad_vec

    @property
    def mass_energy_op_grad_vec(self):
        """Compute the gradient of the mass energy operator for the whole system without shift.
        This is a get function.

        Returns:
            float: gradient of the mass energy operator (w/o shift) for the whole system
        """
        if self.cfg.g_mass == 0:
            self._mass_energy_op_grad_vec = xnp.zeros(self.cfg.param_shape())

        if self._mass_energy_op_grad_vec is None:
            self._mass_energy_op_vec, self._mass_energy_op_grad_vec = (
                self._compute_mass_energy_op_vec_and_grad()
            )
            # self._mass_energy_op_grad_vec *= self.cfg.lattice.size
        return self._mass_energy_op_grad_vec

    @property
    def int_energy_op_grad_vec(self):
        """Compute the gradient of the interaction energy operator for the whole system without shift.
        This is a get function.

        Returns:
            float: Gradient of the interaction energy operator (w/o shift) for the whole system
        """
        if self.cfg.g_int == 0:
            self._int_energy_op_grad_vec = xnp.zeros(self.cfg.param_shape())

        elif self._int_energy_op_grad_vec is None:
            self._int_energy_op_vec, self._int_energy_op_grad_vec = (
                self._compute_int_energy_op_vec_and_grad()
            )
            # Do for whole system...
        return self._int_energy_op_grad_vec

    @property
    def chem_energy_op_grad_vec(self):
        """Compute the gradient of the chemical potential energy operator for the whole system without shift.
        This is a get function.

        Returns:
            float: Gradient of the chemical potential energy operator (w/o shift) for the whole system
        """
        if xnp.allclose(self.cfg.g_chem, 0):
            self._chem_energy_op_grad_vec = xnp.zeros(self.cfg.param_shape())

        elif self._chem_energy_op_grad_vec is None:
            self._chem_energy_op_vec, self._chem_energy_op_grad_vec = (
                self._compute_chem_energy_op_vec_and_grad()
            )
        return self._chem_energy_op_grad_vec

    ##################  ######################

    def occupation(self, lay: int, site: int, after_ph: bool = False) -> float:
        """Compute the occupation number for the given layer and site.

        Returns:
            float: the occupation number for the given layer and site
        """
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )

    def average_occupation(self, after_ph: bool = False) -> xnp.ndarray:
        """Compute the average occupation number for the system across all sites.

        Args:
            after_ph (bool, optional): If True, compute the occupation number using the operators defined after the particle-hole transformation. Defaults to False.

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
        path_product = (
            self.cfg.gaugemgr.get_neutral_gauge_value()
        )  # The identity matrix
        for ind, conj in path:
            if conj:
                path_product = path_product @ xnp.conjugate(
                    xnp.transpose(self.gaugefieldvec[ind])
                )
            else:
                path_product = path_product @ self.gaugefieldvec[ind]
        return xnp.trace(path_product)

    ## MOVE TO GLOBAL
    def compute_ferm_cov(self, layer: int) -> xnp.ndarray:
        """Compute the covariance matrix of the fermions in the system for the given layer.
        We do not calculate it for all layers automatically, since it is not needed for pure-gauge layers.

        Args:
            layer (int): the layer for which the covmat should be calculated
        """
        if self._ferm_covmat_vec is None:
            self._ferm_covmat_vec = [None] * self.cfg.nlayer
        if self._ferm_covmat_vec[layer] is None:
            self._ferm_covmat_vec[layer] = self.mat_a_vec[layer] + (
                self.mat_b_vec[layer]
                @ self.wi_gamma_out_vec[layer].inv()
                @ xnp.transpose(self.mat_b_vec[layer])
            )
        return self._ferm_covmat_vec[layer]

    ################## Mode Permutations ######################

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
        num_copies = (
            self.cfg.ncopy
        )  # The ncopy property is defined the config of any child class of System2DBase
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
            mode_str = (
                mode[0] + "_" + str(mode[1]) + "_" + str(mode[2]) + "_" + str(mode[3])
            )
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
        num_copies = (
            self.cfg.ncopy
        )  # The ncopy property is defined the config of any child class of System2DBase
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
            mode_str = (
                mode[0] + "_" + str(mode[1]) + "_" + str(mode[2]) + "_" + str(mode[3])
            )
            mode_order_str.append(mode_str)

        return mode_order_str

    def get_single_link_majorana_mode_order(self) -> list:
        """Generate the link-based majorana mode order for a single link. Where we first order by color and then by copy.
            This is the actual order we use in system implementaion.

        Returns:
            list: List of strings of the form <mode_letter:majorana mode>_<copy>_<color>
        """

        num_copies = (
            self.cfg.ncopy
        )  # The ncopy property is defined the config of any child class of System2DBase
        mode_order = []
        num_colors = self.cfg.gaugemgr.rep_dim
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

    def get_wrong_single_link_majorana_mode_order_by_copy_then_color(self) -> list:
        """Generate the link-based majorana mode order for a single link. Where we first order by copy and then by color.
            This is not the order we use in the code. This is just to change the generate_rotmat ordeing
        Returns:
            list: List of strings of the form <mode_letter:majorana mode>_<copy>_<color>
        """

        num_copies = (
            self.cfg.ncopy
        )  # The ncopy property is defined the config of any child class of System2DBase
        mode_order = []
        num_colors = self.cfg.gaugemgr.rep_dim
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


def get_pfaffian_arrays(modes, coefficients):
    """Generate the arrays used for list comprehension to extract the required pfaffians, with the correct
    prefactors, used in the calculation of the electric energy and electric gradients.

    Each element in the returned list is of the form
        (k, (a_1 ... a_2p))
    where k in a prefactor, and (a_1 ... a_2p) is a tuple containing the indices to extract from the full
    covariance matrix to build a submatrix and compute the pfaffian.
    The electric energy will then be sum of these pfaffians (weighted by the prefactors), with some further
    normalization.

    Args:
        modes (List of lists of tuples of ints): _description_
        coefficients (List of lists of complex floats): _description_
        neg (float): _description_

    Returns:
        List: index array in the format required for the calculation of the electric energy (and electric gradients).
    """
    submatrices = [k for k in it.product(*modes)]
    indices = [sum(sub, ()) for sub in submatrices]

    factors = [xnp.asarray(k) for k in it.product(*coefficients)]
    prefactors = [xnp.prod(k) for k in factors]
    idxarr = [(p, i) for p, i in zip(prefactors, indices)]

    return idxarr
