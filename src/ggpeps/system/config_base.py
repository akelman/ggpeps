from abc import ABC, abstractmethod
from typing import Union, Optional

import sys
import logging

import sympy

import numpy as np
import jax.numpy as jnp
from ggpeps import xnp as xnp

import ggpeps
from ggpeps import gauge
from ggpeps.lattice import Lattice2D

logger = logging.getLogger(ggpeps.LOGGER_NAME)

# Type aliases for the electric energy data structures
IdxTerm = tuple[complex, tuple[int, ...]]  # (prefactor, indices)
IdxTermList = tuple[IdxTerm, ...]  # all terms for one direction in one layer
IdxLayerPair = tuple[IdxTermList, IdxTermList]  # (horizontal, vertical)
IdxArrVec = tuple[IdxLayerPair, ...]  # over layers


################## Config2DBase ######################
class Config2DBase(ABC):
    """Configuration for a system in two dimensions

    This class inherits from the abstract base class to enable abstract methods that must be overwritten
    in a child class.
    This class cannot be instantiated directly.
    """

    # Ansatz settings
    # This will be overwritten by the specifications of each ansatz
    _nparams: int  # number of params per site per layer
    ncopy: int
    nvirtmodes_vertex: int
    nvirtmodes_link: int
    nphysmodes_site: int  # number of physical modes per site
    ncolors: int

    def __init__(
        self,
        gaugemgr: Union[gauge.ZNGauge, gauge.D2nGauge],
        lattice: Lattice2D,
        g_el: float,
        g_mag: float,
        g_int: float,
        g_mass: float,
        g_chem: Optional[np.ndarray],
        num_pg_layer: int = 1,
        num_fermionic_layer: int = 0,
        mod_link_inds: tuple[int, ...] = (0,),
        unitcell_size: int = 1,
        enforce_u1_symmetry: bool = True,
    ) -> None:
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
            mod_link_inds (tuple[int, ...], optional): indices of links on which to compute the electric energy.
                Defaults to (0,).
            unitcell_size (int, optional): size of the unit cell for translation invariance.
        """

        self.gaugemgr = gaugemgr
        self.lattice = lattice
        self.num_pg_layer = num_pg_layer
        self.num_fermionic_layer = num_fermionic_layer
        self.nlayer = self.num_pg_layer + self.num_fermionic_layer

        # Link indices for which the electric energy is computed - can be any set of horizontal links:
        self.mod_link_inds = mod_link_inds

        # Symbolvec - list of all the symbols, which are the same for each layer
        # (even if for some layers some are forced to zero)
        self._symbolvec: Optional[list[sympy.Symbol]] = None

        # Translation invariance (or variance)
        # define a map from site to index of independent parameters
        if unitcell_size == 1:
            self.site_params_dict = {site: 0 for site in range(self.lattice.size)}
        elif unitcell_size == 2:
            self.site_params_dict = {}
            for site in range(self.lattice.size):
                x, y = self.lattice.ind2coord(site)
                uc_ind = 1 if (x + y) % 2 else 0  # 0 for even sublattice, 1 for odd
                self.site_params_dict[site] = uc_ind
        elif unitcell_size == -1:
            # no unitcell - every site has its own parameters
            self.site_params_dict = {}
            for site in range(self.lattice.size):
                self.site_params_dict[site] = site

        # number of different sets of parameters across sites (min: 1, max: num_sites)
        self.unitcell_size: int = len(set(self.site_params_dict.values()))

        # U1 invariance
        # set to True if you want to enforce U(1) symmetry in the fermionic layers
        # (set to False to allow fermionic number to float between sectors)
        self.u1_symmetry = enforce_u1_symmetry

        # We store a list of the parameters forced to be zero by the ansatz
        # They are actually used in self.enforce_parameter_conditions(), as well as in other checks throughout
        self.zeroed_params: tuple[tuple[int, int, int], ...] = self.get_zeroed_params()

        # Parameters of the Hamiltonian
        self.g_el = g_el
        self.g_mag = g_mag
        self.g_int = g_int
        self.g_mass = g_mass
        if g_chem is None:
            self.g_chem = np.zeros(self.num_fermionic_layer)
        elif len(g_chem) != self.num_fermionic_layer:
            raise ValueError("The number of chemical potentials must match the number of fermionic layers.")
        else:
            self.g_chem = g_chem

        # Settings for the electric energy
        # these depend on the ansatz, so we only declare their type here
        # TODO: this data structure is very unwieldy and should be simplified/restructured
        self.idxarr_vec: IdxArrVec
        self.el_overall_factors: tuple[complex, ...]

    def __str__(self) -> str:
        """Define a string method that can be used, e.g., in filenaming.
        This string doesn't include enough information to reconstruct the config"""

        chem_str = "_".join([f"{val:.3f}" for val in self.g_chem])
        val = (
            f"L_{self.lattice.nx:02d}x{self.lattice.ny:02d}"
            + f"_ncopy_{self.ncopy}_nlayer_{self.nlayer}"
            + f"_gel_{self.g_el}_gmag_{self.g_mag}_gint_{self.g_int}"
            f"_gmass_{self.g_mass}_gchem_{chem_str}"
        )
        return val

    @property
    def paramvec(self) -> np.ndarray:
        return self._paramvec

    @paramvec.setter
    def paramvec(self, val: np.ndarray) -> None:
        if not isinstance(val, np.ndarray):
            val = np.array(val)
        if self.trans_inv and val.ndim == 2:
            # if the system is translation invariant, we add an extra dimension corresponding to the site index
            val = np.expand_dims(val, axis=1)
        if self.check_params(val):
            self._paramvec = val
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
    def nparams_per_layer(self) -> int:
        return self._nparams * self.unitcell_size

    def nvarparams(self) -> int:
        return self._nparams * self.unitcell_size * self.nlayer

    def param_shape(self) -> tuple[int, int, int]:
        """Return the shape required for valid parameters."""
        shape = (self.nlayer, self.unitcell_size, self._nparams)
        return shape

    def parse_params(self, paramvec: np.ndarray, layer: int, site: int) -> np.ndarray:
        """Process the parameters and return the parameters for the given layer and site.

        Args:
            paramvec (array): parameters
            layer (int): the layer for which the parameters are needed
            site (int): the site for which the parameters are needed

        Returns:
            array: parameters for the given layer and site (this will be a subarray of paramvec)
        """
        shape = self.param_shape()
        if len(shape) == 2:
            res = paramvec[layer]
        else:
            ind = 0  # TODO: modify this to account for not every site being independent
            res = paramvec[layer][ind]
        return res

    def print_parametervec(self) -> None:
        """Printing of the parametervec, labelled by layer, unitcell index, and symbol."""
        if self.paramvec is None:
            print("Parameter vector is not set.")
        else:
            for lay in range(self.nlayer):
                for uc_ind in range(self.unitcell_size):
                    for ind, symb in enumerate(self.symbolvec):
                        val = self.paramvec[lay][uc_ind][ind]
                        print(f"Layer {lay}, uc_ind {uc_ind}, symbol {ind} ({symb}): {val}")

    @property
    def trans_inv(self) -> bool:
        """Flag to indicate whether the system is translationally invariant.

        Returns:
            bool: True if ansatz is translationally invariant, False otherwise.
        """
        return self.unitcell_size == 1

    @abstractmethod
    def get_zeroed_params(self) -> tuple[tuple[int, int, int], ...]:
        """Create and return the list of parameters that are forced to zero by the ansatz.

        This abstract method must be overwritten by a subclass.
        We return a tuple rather than a list to emphasize immutability, and because this
        is required for jax.jit.

        Returns:
            tuple: is a tuple of tuples (layer, unitcell index, symbol index).
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @abstractmethod
    def init_el_energy_terms(self) -> None:
        """Initialize electric-energy-related structures for this ansatz.

        Implementations must set:
            - self.idxarr_vec: IdxArrVec
        """
        raise NotImplementedError("Implement in subclass: must set idxarr_vec.")

    def enforce_parameter_conditions(self, mat: xnp.ndarray) -> None:
        """Enforce conditions on the parameters according to the requirements of the ansatz.
        Examples:
            1. make the system pure gauge, i.e. no physical fermions;
            2. enforce a symmetry that is not enforced by ansatz automatically
               (this may not be enforced at the level of the t-mat in order to allow the symmmetry to be relaxed)

        It acts on arrays with the same shape as the paramvec.
        This function acts on the provided array in-place.

        Args:
            mat (array): array of parameters to which the conditions are applied.
        """
        if mat.shape != self.param_shape():
            raise ValueError(
                f"Invalid shape for mat in enforce_parameter_conditions: {mat.shape}. "
                f"Expected {self.param_shape()}."
            )

        if self.zeroed_params is None:
            self.zeroed_params = self.get_zeroed_params()

        for coord in self.zeroed_params:
            # We do not use the backend, because this function can accept numpy arrays even when using jax
            # (since the paramvec is always a numpy array)
            if isinstance(mat, np.ndarray):  # TODO: handle jax better
                mat[coord] = 0
            elif isinstance(mat, jnp.ndarray):
                mat = mat.at[coord].set(0)
            else:
                raise TypeError(
                    "Unsupported type for mat in enforce_parameter_conditions: "
                    f"{type(mat)}. Expected np.ndarray or jnp.ndarray."
                )
        return

    @abstractmethod
    def _create_symbolvec(self) -> list[sympy.Symbol]:
        """
        Function to define the list of parameters as sympy variables.
        We need these symbols to analytically derive T automatically.
        This function has to be overwritten in the child-class.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @property
    def symbolvec(self) -> list[sympy.Symbol]:
        """Return the symbolvec.
        This is a get function. It computes the symbolvec only if it does not exist yet.
        If it exists, then it will be returned directly. If not, it will be created and then stored in _symbolvec.

        Returns:
            list: Vector of analytic symbols
        """
        if self._symbolvec is None:
            self._symbolvec = self._create_symbolvec()
            assert len(self._symbolvec) == self._nparams
        return self._symbolvec

    @property
    @abstractmethod
    def tmat_symb(self) -> sympy.Matrix:
        """Create the symbolic version of the T matrix.
        This is an abstract function that has to be overwritten by the child class.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @abstractmethod
    def generate_gamma_gauge_neutral_dict(self) -> np.ndarray:
        """Abstract method to define the ungauged covariance matrix of a single link.
        The substitution method must ensure a consistent order of the modes.
        The direction parameter controls which covariance matrix is retrieved,
        since these can differ between directions.
        This method must be overwritten in a subclass.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")
