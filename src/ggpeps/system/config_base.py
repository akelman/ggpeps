from abc import ABC, abstractmethod
from typing import Optional, Union
from collections import defaultdict

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
IdxTermSize = tuple[tuple[int, ...], ...]  # (tuple of tuples of indices, all inner tuples have the same size)
IdxTermLink = tuple[IdxTermSize, ...]  # All different size terms for a specific link
IdxTermsLayer = tuple[IdxTermLink, ...]  # all link terms for one layer
IdxGroup = tuple[IdxTermsLayer, ...]  # Per group element
IdxVec = tuple[IdxGroup, ...]  # over group elements

CoeffsTermSize = tuple[float, ...]
# (tuple of floats, all floats are the coefficient corresponding to inner tuples in IdxTermSize)
CoeffsTermLink = tuple[CoeffsTermSize, ...]  # All different size terms for a specific link
CoeffsTermsLayer = tuple[CoeffsTermLink, ...]  # all link terms for one layer
CoeffsGroup = tuple[CoeffsTermsLayer, ...]  # Per group element
CoeffsVec = tuple[CoeffsGroup, ...]  # over group elements

ConstantsTermLink = complex
# The constant added to the sum of terms in generate_gauged_projector
# (this is the term that does not come with a Pfaffian)
ConstantsLayer = tuple[ConstantsTermLink, ...]  # all link terms for one layer
ConstantsGroup = tuple[ConstantsLayer, ...]  # Per group element
ConstantsVec = tuple[ConstantsGroup, ...]  # over group elements


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
        gaugemgr: gauge.GaugeGroup,
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
        self.mod_link_inds = tuple(mod_link_inds)

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
        shape = (self.nlayer, self.unitcell_size, len(self.symbolvec))
        self.mask = np.ones(shape)
        for zeroed_param in self.zeroed_params:
            self.mask[zeroed_param] = 0

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
            self.g_chem = np.asarray(g_chem)

        # Settings for the electric energy
        # these depend on the ansatz, so we only declare their type here
        self.idx_vec: IdxVec
        self.coeffs_vec: CoeffsVec
        self.constants_vec: ConstantsVec

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
        uc_ind = self.site_params_dict[site]
        return paramvec[layer][uc_ind]

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

    def _bucket_sort_terms(self, term_list):
        """
        Helper to bucket terms by length, sort them, and return tuple-ized structure.
        Returns: (tuple_of_coeffs, tuple_of_indices, tuple_of_lengths)
        """
        # Bucket terms by length of indices
        # Structure: { length: ([coeffs], [indices]) }
        temp_buckets = {}
        for coef, indices in term_list:
            length = len(indices)
            if length not in temp_buckets:
                temp_buckets[length] = ([], [])
            temp_buckets[length][0].append(coef)
            temp_buckets[length][1].append(indices)

        # Sort buckets by length
        sorted_lengths = sorted(temp_buckets.keys())

        curr_l_coeffs = []
        curr_l_indices = []

        for length in sorted_lengths:
            c_list, i_list = temp_buckets[length]
            curr_l_coeffs.append(tuple(c_list))
            curr_l_indices.append(tuple(i_list))

        return tuple(curr_l_coeffs), tuple(curr_l_indices)

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
            - self.idx_vec: IdxVec
            - self.coeffs_vec: CoeffsVec
        """
        raise NotImplementedError("Implement in subclass: must set idx_vec and coeffs_vec.")

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


# ==================== Pfaffian indices Terms ====================

# ---------------------------------------------------------------------------
# The following functions are used to compute the indices and coefficients
# of the terms which appear in the calculation of the overlap of modified
# norms in the electric energy. The code is designed to match our notes
# as closely as possible, and sacrifices efficiency for that purpose.
#
# Per-(color,copy) operator.
#
# The full gauged projector on a link is
#   U_h^dag w |Omega><Omega| w^dag
# which equals
#   O = ( prod_a W_a ) ( prod_a V_a ) ( prod_a w_a^dag )
# with, per alpha=(color,copy),
#     W_a       = U_h^dag w_a U_h = 1 + eta2  l^dag_a  sum_b M_{b a} r^dag_b    (gauged w factor)
#     V_a       = l_a l^dag_a r_a r^dag_a                                       (vacuum projector)
#     w_a^dag   = 1 + eta2bar r_a l_a
# (l uses sigma_copy, r uses copy, matching the projector pairing.)
#
# ---------------------------------------------------------------------------


def make_sigma(ncopy: int, mix_copies: bool) -> tuple[int, ...]:
    """
    Build the link-pairing permutation ("sigma" in our notes) for a single layer.

    Args:
        ncopy (int): Number of copies (must be 1 or even).
        mix_copies (bool): if True, mix copies: swaps (2a-1 <-> 2a) for a = 1..k/2, i.e. sigma(1)=2 and sigma(2)=1
                            if False, don't mix copies: identity, sigma(c) = c

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


def make_sigma_matrix(ncopy: int, mix_copies: bool) -> np.ndarray:
    """
    Build the copy-space permutation matrix corresponding to `make_sigma`.

    The returned matrix acts on copy space only. For mixed-copy projectors it swaps
    copies pairwise, (1 <-> 2), (3 <-> 4), ... . For unmixed projectors it is the
    identity. For ncopy == 1 both choices reduce to the 1x1 identity matrix.

    Args:
        ncopy (int): Number of copies (must be 1 or even).
        mix_copies (bool): Whether to use the mixed-copy pairing convention.

    Returns:
        np.ndarray: The ncopy x ncopy copy-space permutation matrix.
    """
    sigma = make_sigma(ncopy, mix_copies)
    mat = np.zeros((ncopy, ncopy))
    for copy_ind, sigma_copy in enumerate(sigma):
        mat[copy_ind, sigma_copy - 1] = 1.0
    return mat


def _w_gauged_terms(
    copy: int, sigma_copy: int, eta2: complex, color: int, ncolors: int, ncopies: int, gauging_matrix: np.ndarray
) -> defaultdict[tuple[int, ...], complex]:
    """Compute the terms making up the gauged projector U_h^dag w.
    For each copy and color, this operator is:
        W_a = 1 + eta2 * l^dag_(color,sigma_copy) * sum_b M_{b,color} r^dag_(b,copy).
    """
    l1 = get_cov_matrix_idx(color, sigma_copy, side=1, majorana=1, ncolors=ncolors, ncopies=ncopies)
    l2 = get_cov_matrix_idx(color, sigma_copy, side=1, majorana=2, ncolors=ncolors, ncopies=ncopies)
    terms: defaultdict[tuple[int, ...], complex] = defaultdict(complex)
    terms[()] = 1.0 + 0j
    for m in range(1, ncolors + 1):
        a_m = get_cov_matrix_idx(m, copy, side=2, majorana=1, ncolors=ncolors, ncopies=ncopies)
        b_m = get_cov_matrix_idx(m, copy, side=2, majorana=2, ncolors=ncolors, ncopies=ncopies)
        mel = gauging_matrix[m - 1][color - 1]
        if abs(mel) < 1e-15:
            continue
        coef = eta2 * mel  # l^dag r^dag_m = 1/4[(c,a_m) + i(c,b_m) + i(d,a_m) - (d,b_m)]
        terms[(l1, a_m)] += 0.25 * coef
        terms[(l1, b_m)] += 0.25j * coef
        terms[(l2, a_m)] += 0.25j * coef
        terms[(l2, b_m)] += -0.25 * coef
    return terms


def vacuum_terms(
    copy: int, sigma_copy: int, color: int, ncolors: int, ncopies: int
) -> defaultdict[tuple[int, ...], complex]:
    """Compute the terms making up the projector to the vacuum.
    For each copy and color, this operator is:
        V_a = l l^dag r r^dag = 1/4 (1 + i l1 l2)(1 + i r1 r2)
            = 1/4 + 1/4 i l1 l2 + 1/4 i r1 r2 - 1/4 l1 l2 r1 r2
    """
    l1 = get_cov_matrix_idx(color, sigma_copy, side=1, majorana=1, ncolors=ncolors, ncopies=ncopies)
    l2 = get_cov_matrix_idx(color, sigma_copy, side=1, majorana=2, ncolors=ncolors, ncopies=ncopies)
    r1 = get_cov_matrix_idx(color, copy, side=2, majorana=1, ncolors=ncolors, ncopies=ncopies)
    r2 = get_cov_matrix_idx(color, copy, side=2, majorana=2, ncolors=ncolors, ncopies=ncopies)
    res: defaultdict[tuple[int, ...], complex] = defaultdict(complex)
    res[()] = 0.25 + 0j
    res[(l1, l2)] = 0.25j
    res[(r1, r2)] = 0.25j
    res[(l1, l2, r1, r2)] = -0.25 + 0j
    return res


def _w_dag_terms(
    copy: int, sigma_copy: int, eta2: complex, color: int, ncolors: int, ncopies: int
) -> defaultdict[tuple[int, ...], complex]:
    """Compute the terms making up the projector
    For each copy and color, this operator is:
        w_a^dag = 1 + conj(eta2) * r_(color,copy) l_(color,sigma_copy).
    """
    l1 = get_cov_matrix_idx(color, sigma_copy, side=1, majorana=1, ncolors=ncolors, ncopies=ncopies)
    l2 = get_cov_matrix_idx(color, sigma_copy, side=1, majorana=2, ncolors=ncolors, ncopies=ncopies)
    r1 = get_cov_matrix_idx(color, copy, side=2, majorana=1, ncolors=ncolors, ncopies=ncopies)
    r2 = get_cov_matrix_idx(color, copy, side=2, majorana=2, ncolors=ncolors, ncopies=ncopies)
    eb = complex(eta2).conjugate()  # r_a l_a = 1/4[(a,c) - i(a,d) - i(b,c) - (b,d)]
    res: defaultdict[tuple[int, ...], complex] = defaultdict(complex)
    res[()] = 1.0 + 0j
    res[(r1, l1)] = 0.25 * eb
    res[(r1, l2)] = -0.25j * eb
    res[(r2, l1)] = -0.25j * eb
    res[(r2, l2)] = -0.25 * eb
    return res


def poly_mul(
    p: defaultdict[tuple[int, ...], complex], q: defaultdict[tuple[int, ...], complex], tol: float = 0.0
) -> defaultdict[tuple[int, ...], complex]:
    """
    Multiply two polynomials, represented as dictionaries {indices: coefficients}
    and return the result as new dict in the same format.

    Note: the operation is not commutative, as we are working with indices of Majorana (fermionic) modes.
    """
    new: defaultdict[tuple[int, ...], complex] = defaultdict(complex)
    for inds_a, coef_a in p.items():
        for inds_b, coef_b in q.items():
            new[inds_a + inds_b] += coef_a * coef_b
    return simplify_polynomial(new, tol=tol)


def generate_gauged_projector_terms(
    ncopy: int,
    ncolor: int,
    mix_copies: bool,
    orientation: Direction,
    group_element: np.ndarray,
    site: int = 0,
    drop_imag: bool = True,
    tol: float = 1e-10,
) -> tuple[tuple[tuple[complex, tuple[int, ...]], ...], complex]:
    """
    Expand the gauged projector product and collect terms.

    The function computes the indices for which we take the Pfaffian of the modificed covariance matrix,
    needed for the electric energy calculation.

    Args:
        ncopy (int): Number of copies.
        ncolor (int): Number of colors.
        mix_copies (bool): whether to mix copies in the projectors (controls sigma permutation).
        orientation (Direction): 'X' (horizontal, eta^2 = 1) or 'Y' (vertical, eta^2 = i).
        group_element (np.ndarray): Group element h.
        site (int, optional): Site index (used for parity-dependent conjugation). Defaults to 0.
        drop_imag (bool, optional): Drop imaginary parts of coefficients. Defaults to True.
        tol (float, optional): Tolerance for dropping small coefficients. Defaults to 1e-10.

    Returns:
        tuple[tuple[tuple[complex, tuple[int, ...]], ...], complex]:
            (indices, constant), where:
            - indices: A tuple of (coefficient, monomial indices), sorted by length then lexicographically.
            - constant: The scalar constant term of the polynomial.

    Raises:
        ValueError: On invalid ncopy.
    """
    ## Stage 1: Preliminaries
    sigma = make_sigma(ncopy, mix_copies)

    # Map orientation of the link -> eta^2
    # TODO: this should be done inside the config, to ensure consistency
    eta2: Union[float, complex]  # this is the extra factor that appears in the projectors
    if orientation == Direction.X:
        eta2 = 1.0
    elif orientation == Direction.Y:
        eta2 = 1.0j
    else:
        # got Direction.Z, which is not yet supported
        raise ValueError("Link orientation must be 'X' (horizontal) or 'Y' (vertical).")

    # Normalization factor from the projectors
    pref = 2 ** (-ncopy * ncolor)

    # Conjugate the representation on the odd sublattice
    gauging_matrix = group_element if site % 2 == 0 else np.conjugate(group_element)

    ## Stage 2: Create a dictionary mapping Majorana index tuples to coefficients

    # Initialize the polynomial accumulator
    # maps each tuple of Majorana indices -> complex coefficient (prefactor)
    polynom: defaultdict[tuple[int, ...], complex] = defaultdict(complex)
    polynom[()] = 1.0  # We initialize this way to allow multiplication of polynomials from the get go

    # Build the Majorana polynomial corresponding to U_h^dag w |Omega><Omega| w^dag
    # a) Majorana terms arising from U_h^dag w
    for color in range(1, ncolor + 1):
        for copy in range(1, ncopy + 1):
            sc = sigma[copy - 1]
            gauged_projector_terms = _w_gauged_terms(copy, sc, eta2, color, ncolor, ncopy, gauging_matrix)
            polynom = poly_mul(polynom, gauged_projector_terms)
    # b) Majorana terms arising from the projection onto the vacuum
    for color in range(1, ncolor + 1):
        for copy in range(1, ncopy + 1):
            sc = sigma[copy - 1]  # the copy to which this copy is coupled by the projector
            vac_terms = vacuum_terms(copy, sc, color, ncolor, ncopy)
            polynom = poly_mul(polynom, vac_terms)
    # c) Majorana terms arising from w^dag
    for color in range(1, ncolor + 1):
        for copy in range(1, ncopy + 1):
            sc = sigma[copy - 1]
            w_dag_terms = _w_dag_terms(copy, sc, eta2, color, ncolor, ncopy)
            polynom = poly_mul(polynom, w_dag_terms)

    ## Stage 3: Cleanup

    # Final simplification to avoid computing same pfaffian multiple times
    polynom = simplify_polynomial(polynom, tol=tol)

    # Get the constant term and handle it separately
    constant = polynom.pop((), 0.0)
    constant *= pref  # add in global prefactor

    # Filter and account for extra factors
    polynom_list: list[tuple[complex, tuple[int, ...]]] = []
    for mon, coef in polynom.items():
        pfaffian_wick_phase = 1.0j ** (-len(mon) // 2)  # the Pfaffian-Wick phase
        new_coef = coef * pfaffian_wick_phase * pref  # add in global prefactor, and Pfaffian-Wick phase

        if abs(new_coef) < tol:
            continue  # Skip terms with negligible coefficients

        # Drop terms for which the coefficient has zero real part
        # TODO: explain why/when this is allowed/desired
        if drop_imag:
            if np.abs(np.real(new_coef)) > tol:
                polynom_list.append((np.real(new_coef), mon))
        else:
            polynom_list.append((new_coef, mon))

    # Sort terms by monomial length (shorter first) then lexicographic order for deterministic output
    polynom_list.sort(key=lambda kv: (len(kv[1]), kv[1]))
    indices = tuple(polynom_list)

    return indices, constant


def get_cov_matrix_idx(color: int, copy: int, side: int, majorana: int, ncolors: int, ncopies: int) -> int:
    """Get the index in the covariance matrix for a given mode.
    This indexes into a matrix containing just the Majorana virtual modes on a single link.

    Args:
        color (int): color index (1 to ncolors) - 1-based
        copy (int): copy index (1 to ncopies) - 1-based
        side (int): index for outgoing or incoming mode, i.e. (l or r), (u or d) - 1-based.
            1 for left/down, 2 for right/up.
        majorana (int): Majorana index (1 to 2) - 1-based
        ncolors (int): number of colors
        ncopies (int): number of copies

    Returns:
        int: index in the covariance matrix - 0-based
    """
    idx = (
        (color - 1) * (ncopies * 2 * 2)  # one factor of 2 for two Majorana modes; another for l and r modes
        + (copy - 1) * (2 * 2)
        + (side - 1) * 2
        + (majorana - 1)
    )
    return idx


def simplify_polynomial(
    polynomial: defaultdict[tuple[int, ...], complex], tol: float = 1e-10
) -> defaultdict[tuple[int, ...], complex]:
    """
    Simplifies a Majorana polynomial.
    1. Sorts indices to a canonical order (applying sign flips for swaps).
    2. Contracts identical adjacent pairs (c_i^2 = 1).
    3. Combines terms with identical indices by summing their coefficients.
    """
    simplified: defaultdict[tuple[int, ...], complex] = defaultdict(complex)

    for indices, coeff in polynomial.items():

        # Convert to list for in-place sorting
        idx_list = list(indices)

        # 1. Canonical Sorting
        # Majorana fermions anti-commute: c_i c_j = -c_j c_i
        # We need to sort the indices while keeping track of the sign changes due to swaps (i.e. whether the result
        # is an even or odd permutation). We do this with a bubble sort type algorithm in order to count swaps.
        swaps = 0
        n = len(idx_list)
        for i in range(n):
            for j in range(0, n - i - 1):  # after i iterations of the outer loops, the last i elements are in place
                if idx_list[j] > idx_list[j + 1]:
                    idx_list[j], idx_list[j + 1] = idx_list[j + 1], idx_list[j]
                    swaps += 1

        # Apply sign flip for odd number of swaps
        if swaps % 2 != 0:
            coeff = -coeff

        # 2. Elimination of pairs
        # Since the list is sorted, c_i^2 terms are adjacent
        stack: list[int] = []
        if len(idx_list) > 0:  # TODO: this check is probably not necessary
            for idx in idx_list:
                if stack and stack[-1] == idx:
                    stack.pop()  # Remove pair (c^2 = 1)
                else:
                    stack.append(idx)
            reduced_indices = tuple(stack)
        else:
            reduced_indices = ()

        # 3. Aggregate
        simplified[reduced_indices] += coeff

    # Final cleanup of the simplified dictionary
    # Without this check, some coefficients may be zero, but we want to drop those terms
    final_polynomial = defaultdict(complex)
    for k, v in simplified.items():
        if abs(v) > tol:
            final_polynomial[k] = v

    return final_polynomial
