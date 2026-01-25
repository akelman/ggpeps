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
from ggpeps.lattice import Lattice2D, Direction

logger = logging.getLogger(ggpeps.LOGGER_NAME)

# Type aliases for the electric energy data structures
IdxTerm = tuple[complex, tuple[int, ...]]  # (prefactor, indices)
IdxTermQuad = tuple[IdxTerm, IdxTerm, IdxTerm, IdxTerm]  # (term_h0, term_h1, term_v0, term_v1) for one term
IdxLayerTerms = tuple[IdxTermQuad, ...]  # all (term_h, term_v) pairs for one layer
IdxArrVecLayer = tuple[IdxLayerTerms, ...]  # over layers
IdxArrVec = tuple[IdxArrVecLayer, ...]  # over group elements


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


# ==================== ZN Gauged Projector Terms ====================


from collections import defaultdict
from typing import Optional, Union, DefaultDict

MonomialAccumulator = DefaultDict[tuple[int, ...], complex]


def make_sigma(ncopy: int, mix_copies: bool) -> tuple[int, ...]:
    """
    Build the link-pairing permutation (sigma) for a single layer.

    Args:
        ncopy (int): Number of copies (must be 1 or even).
        mix_copies (bool): if True, mix copies: swaps (2a-1 <-> 2a) for a = 1..k/2.
                            if False, don't mix copies: identity, sigma(j) = j

    Returns:
        tuple[int, ...]: 1-based permutation list where entry j equals sigma(j).
    """
    if not (ncopy == 1 or ncopy % 2 == 0):
        # TODO: odd ncopy >1 could be allowed for unmixed copies
        raise ValueError("ncopy must be 1 or even (odd ncopy >1 is not supported).")
    elif not mix_copies:
        permutation = tuple(range(1, ncopy + 1))
    elif mix_copies:
        if ncopy == 1:
            return (1,)
        s = [0] * ncopy
        for a in range(1, ncopy // 2 + 1):
            i, j = 2 * a - 1, 2 * a
            s[i - 1] = j
            s[j - 1] = i
        permutation = tuple(s)
    return permutation


def bracket_terms(
    copy: int,
    sigma_copy: int,
    eta2: complex,
    color: int,
    ncolors: int,
    ncopies: int,
    group_element: xnp.ndarray,
    site,
) -> list[tuple[complex, tuple[int, ...]]]:
    """
    Assemble the polynomial terms for the gauged projector bracket for a specific copy and color.

    This function generates the Majorana operator polynomial corresponding to a single link
    in the projector. It includes:
    1.  **Base Terms:** 8 terms involving the local modes (c, d) from the incoming coupling_type
        (sigma_copy) and (a, b) from the outgoing coupling_type (copy).
    2.  **Gauge Mixing Terms:** A summation over all colors 'm', generating 16 terms per color.
        These terms mix the current color indices with the indices of color 'm', weighted
        by the matrix element D_{m, color}(h) (or its conjugate, depending on site parity).

    The indices are resolved using `get_cov_matrix_idx`:
    - a, b: Modes for (color, copy) at direction 2 (Right/Up).
    - c, d: Modes for (color, sigma_copy) at direction 1 (Left/Down).
    - a_m, b_m: Modes for (m, copy) at direction 2.

    Args:
        copy (int): The current copy index (1-based).
        sigma_copy (int): The permuted copy index sigma(copy) (1-based).
        eta2 (complex): Orientation-dependent parameter (eta^2).
        color (int): The current color index (1-based).
        ncolors (int): Total number of colors.
        ncopies (int): Total number of copies.
        group_element (xnp.ndarray): The matrix faithfully representation of the group element 'h'.
        site (int): The site index, used to determine if the matrix element should be conjugated
                    (staggering for fermionic modes).

    Returns:
        list[tuple[complex, tuple[int, ...]]]:
            A list of (coefficient, indices) tuples representing the sparse polynomial.
            Coefficients are snapped to remove numerical noise.
    """
    # r_{color, copy}^{1} or u_{color, copy}^{1}
    a = get_cov_matrix_idx(color, copy, direction=2, majorana=1, ncolors=ncolors, ncopies=ncopies)

    # r_{color, copy}^{2} or u_{color, copy}^{2}
    b = get_cov_matrix_idx(color, copy, direction=2, majorana=2, ncolors=ncolors, ncopies=ncopies)

    # l_{color, copy}^{1} or d_{color, copy}^{1}
    c = get_cov_matrix_idx(color, sigma_copy, direction=1, majorana=1, ncolors=ncolors, ncopies=ncopies)

    # l_{color, copy}^{2} or d_{color, copy}^{2}
    d = get_cov_matrix_idx(color, sigma_copy, direction=1, majorana=2, ncolors=ncolors, ncopies=ncopies)

    eta2_bar = complex(eta2).conjugate()

    raw_terms = [
        (0.5, ()),
        (0.5 * 1j, (c, d)),
        (0.5 * 1j, (a, b)),
        (-0.5, (c, d, a, b)),
        (0.5 * eta2_bar, (a, c)),
        (-0.5 * eta2_bar, (b, d)),
        (-0.5 * 1j * eta2_bar, (a, d)),
        (-0.5 * 1j * eta2_bar, (b, c)),
    ]

    # Apply conjugation on the odd sublattice
    # TODO: implement conjugation for copies that should transform with the conjugate representation,
    # i.e., d modes in the conventions here https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.054511
    if site % 2 == 0:
        gauging_matrix = group_element
    else:
        gauging_matrix = xnp.conjugate(group_element)

    for m in range(1, ncolors + 1):
        a_m = get_cov_matrix_idx(m, copy, direction=2, majorana=1, ncolors=ncolors, ncopies=ncopies)
        b_m = get_cov_matrix_idx(m, copy, direction=2, majorana=2, ncolors=ncolors, ncopies=ncopies)
        matrix_element = gauging_matrix[m - 1][color - 1]
        raw_terms += [
            (0.25 * eta2 * matrix_element, (c, a_m)),
            (-0.25 * eta2 * matrix_element, (d, b_m)),
            (1j * 0.25 * eta2 * matrix_element, (c, b_m)),
            (1j * 0.25 * eta2 * matrix_element, (d, a_m)),
            (1j * 0.25 * eta2 * matrix_element, (c, a_m, a, b)),
            (-1j * 0.25 * eta2 * matrix_element, (d, b_m, a, b)),
            (-0.25 * eta2 * matrix_element, (c, b_m, a, b)),
            (-0.25 * eta2 * matrix_element, (d, a_m, a, b)),
            (0.25 * matrix_element, (a_m, a)),
            (-1j * 0.25 * matrix_element, (a_m, b)),
            (1j * 0.25 * matrix_element, (b_m, a)),
            (0.25 * matrix_element, (b_m, b)),
            (-1j * 0.25 * matrix_element, (c, d, a_m, a)),
            (-0.25 * matrix_element, (c, d, a_m, b)),
            (0.25 * matrix_element, (c, d, b_m, a)),
            (-1j * 0.25 * matrix_element, (c, d, b_m, b)),
        ]

    # Snap small real/imag parts and drop zeros to reduce work upstream
    terms: list[tuple[complex, tuple[int, ...]]] = []
    for coef, inds in raw_terms:
        snapped = snap_complex(coef)
        if snapped != 0.0:
            terms.append((snapped, inds))
    return terms


def snap_complex(z: complex, eps=1e-12) -> complex:
    """
    Component-wise zeroing of tiny real/imag parts.

    Args:
        z (complex): Input complex number.
        eps (float, optional): Tolerance for zeroing components. Defaults to 1e-12.

    Returns:
        complex: Cleaned complex with near-zero components set to 0.0, or 0.0 if abs(z) is below tolerance.
    """
    if abs(z) <= eps:
        return 0.0
    zr = 0.0 if abs(z.real) <= eps else z.real
    zi = 0.0 if abs(z.imag) <= eps else z.imag
    if zr == 0.0 and zi == 0.0:
        return 0.0
    return zr if zi == 0.0 else complex(zr, zi)


def pfaffian_wick_phase(mon: tuple[int, ...]) -> complex:
    """
    Compute the Pfaffian-Wick phase i^(-len(mon)/2) for an even Majorana monomial.

    Args:
        mon (tuple[int, ...]): Ordered Majorana index tuple (even length).

    Returns:
        complex: One of {1, -1j, -1, 1j}, equal to i^(-len(mon)//2).

    Raises:
        ValueError: If len(mon) is odd.
    """
    n = len(mon)
    if n % 2 != 0:
        raise ValueError(f"Monomial length is odd ({n}); expected even.")
    m = n // 2
    r = m % 4  # map exponent class for i^(-m)
    if r == 0:
        return 1.0
    elif r == 1:
        return -1j
    elif r == 2:
        return -1.0
    else:  # r == 3
        return 1j


def generate_gauged_projector_terms(
    ncopy: int,
    ncolor: int,
    mix_copies: bool,
    orientation: Direction,
    group_element: xnp.ndarray,
    site: int = 0,
    drop_real_zero: bool = True,
) -> tuple[tuple[tuple[complex, tuple[int, ...]], ...], complex]:
    """
    Expand the gauged projector product and collect terms.

    This function computes the expansion of the projector operator for the GGFPEPS ansatz.
    Mathematically, it calculates:

    Sum_{h in G} [ Prefactor(h) * Product_{color=1}^{ncolor} Product_{copy=1}^{ncopy} Bracket(h, copy, color) ]

    where:
    - Prefactor(h) = (1/|G|) * 4^{-(ncopy * ncolor)} *
                    * Sum_{irrep} ( dim(irrep) * chi_irrep(h^-1) * electric_energy_factor(irrep) )
    - Bracket(h, copy, color) is the 16-term Majorana polynomial associated with the group element h.

    The function performs the following steps:
    1. Maps orientation to eta^2 (horizontal -> 1, vertical -> i).
    2. Sums over all group elements 'h', accumulating the polynomial product for each 'h'.
    3. Multiplies the result by the Pfaffian-Wick phase i^(-N/2) for each term.
    4. Returns a sparse polynomial representation sorted canonically.

    Args:
        ncopy (int): Number of copies.
        ncolor (int): Number of colors.
        mix_copies (bool): whether to mix copies in the projectors (controls sigma permutation).
        orientation (Direction): 'X' (horizontal, eta^2 = 1) or 'Y' (vertical, eta^2 = i).
        gaugemgr (Union[gauge.ZNGauge, gauge.D2nGauge]): Gauge manager handling group structure and irreps.
        site (int, optional): Site index (used for parity-dependent conjugation). Defaults to 0.
        drop_real_zero (bool, optional): Whether to drop terms with zero real part in coefficients. Defaults to True.

    Returns:
        tuple[tuple[tuple[complex, tuple[int, ...]], ...], complex]:
            (indices, constant), where:
            - indices: A tuple of (coefficient, monomial indices), sorted by length then lexicographically.
            - constant: The scalar constant term of the polynomial.

    Raises:
        ValueError: On invalid ncopy.
    """
    sigma = make_sigma(ncopy, mix_copies)

    # Map orientation -> eta^2
    eta2: Union[float, complex]
    if orientation == Direction.X:
        eta2 = 1.0
    elif orientation == Direction.Y:
        eta2 = 1j
    else:
        # got Direction.Z, which is not yet supported
        raise ValueError("Link orientation must be 'X' (horizontal) or 'Y' (vertical).")

    # Initialize the final polynomial accumulator
    final_polynomial: dict[tuple[int, ...], complex] = defaultdict(complex)
    # Multiply in each bracket

    pref = 4 ** (-ncopy * ncolor)

    acc: MonomialAccumulator = defaultdict(complex)  # Accumulator of partial expansions: monomial tuple -> coefficient
    acc[()] = 1.0  # multiplicative identity (empty monomial)
    for color in range(1, ncolor + 1):
        for copy in range(1, ncopy + 1):
            terms_j = bracket_terms(copy, sigma[copy - 1], eta2, color, ncolor, ncopy, group_element, site)
            new_acc: MonomialAccumulator = defaultdict(complex)
            for indsA, coefA in acc.items():
                for coefB, indsB in terms_j:
                    # Multiply polynomials (append indices, multiply coeffs)
                    new_acc[indsA + indsB] += coefA * coefB
            acc = simplify_majorana_acc(new_acc)
    # After finishing the product, add to the final sum weighted by pref
    for inds, coef in acc.items():
        final_polynomial[inds] += coef * pref
    # Final simplification to avoid computing same pfaffian multiple times
    final_polynomial = simplify_majorana_acc(final_polynomial)

    # Split constant vs others and build the output
    constant = snap_complex(final_polynomial.pop((), 0.0))
    items = [(mon, complex(coef)) for mon, coef in final_polynomial.items() if coef != 0.0]

    # Multiply each coefficient by i^(-len(mon)/2), the Pfaffian-Wick phase
    phased_items: list[tuple[tuple[int, ...], complex]] = []
    for mon, coef in items:
        factor = pfaffian_wick_phase(mon)
        new_coef = snap_complex(coef * factor)
        if new_coef != 0.0:
            phased_items.append((mon, new_coef))

    # Drop terms for which the coefficient has zero real part
    # TODO: explain why this is allowed / desired
    if drop_real_zero:
        phased_items = [(mon, coef) for mon, coef in phased_items if np.abs(np.real(coef)) > 1e-5]

    # Sort terms by monomial length (shorter first) then lexicographic tuple order for deterministic output.
    phased_items.sort(key=lambda kv: (len(kv[0]), kv[0]))
    indices = tuple((coef, mon) for mon, coef in phased_items)

    return indices, constant


def get_cov_matrix_idx(
    color: int, copy: int, direction: int, majorana: int, ncolors: int, ncopies: int, ndirections: int = 2
) -> int:
    """Get the index in the covariance matrix for a given mode.

    Args:
        color (int): color index (1 to ncolors) - 1-based
        copy (int): copy index (1 to ncopies) - 1-based
        direction (int): direction index (1 to ndirections) - 1-based.
                        for ndirections=2 - 1 to left or down and 2 to right or up.
        majorana (int): Majorana index (1 to 2) - 1-based
        ncolors (int): number of colors
        ncopies (int): number of copies
        ndirections (int, optional): number of directions (system's spatial dimension). Defaults to 2.

    Returns:
        int: index in the covariance matrix - 0-based
    """
    idx = (
        (color - 1) * (ncopies * ndirections * 2)
        + (copy - 1) * (ndirections * 2)
        + (direction - 1) * 2
        + (majorana - 1)
    )
    return idx


def simplify_majorana_acc(acc):
    """
    Simplifies the Majorana polynomial accumulator.
    1. Reorders indices to canonical order (applying sign flips for swaps).
    2. Contracts identical adjacent pairs (c_i^2 = 1).
    3. Aggregates identical terms using _snap_complex for numerical stability.

    """
    new_acc = defaultdict(complex)

    for indices, factor in acc.items():
        # Clean noise immediately
        factor = snap_complex(factor)
        if factor == 0.0:
            continue

        # Convert to list for in-place sorting
        idx_list = list(indices)

        # 1. Canonical Sorting
        # Majorana fermions anti-commute: c_i c_j = -c_j c_i
        swaps = 0
        n = len(idx_list)
        for i in range(n):
            for j in range(0, n - i - 1):
                if idx_list[j] > idx_list[j + 1]:
                    idx_list[j], idx_list[j + 1] = idx_list[j + 1], idx_list[j]
                    swaps += 1

        # Apply sign flip for odd number of swaps
        if swaps % 2 != 0:
            factor = -factor

        # 2. Contraction of pairs (Stack method)
        # Since the list is sorted, c_i^2 terms are adjacent.
        stack = []
        if len(idx_list) > 0:
            for idx in idx_list:
                if stack and stack[-1] == idx:
                    stack.pop()  # Remove pair (c^2 = 1)
                else:
                    stack.append(idx)
            reduced_indices = tuple(stack)
        else:
            reduced_indices = ()

        # 3. Aggregate
        new_acc[reduced_indices] += factor

    # Final cleanup of the simplified dictionary
    final_acc = defaultdict(complex)
    for k, v in new_acc.items():
        clean_v = snap_complex(v)
        if clean_v != 0.0:
            final_acc[k] = clean_v

    return final_acc
