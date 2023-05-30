from abc import ABC, abstractmethod
from typing import Union, List # used in type hints; this approach might be deprecated in later python versions

import sys
import logging
import itertools as it

import sympy
import numpy as np

from ggpeps import gauge, utils
from ggpeps.lattice import Direction, Lattice2D, Lattice3D


class Config2DBase(ABC):
    """ Configuration for a system in two dimensions

    This class inherits from the abstract base class to enable abstract methods that has to be overwritten in a child class.
    This class cannot be instantiated directly.
    """

    # Number of parameters of the parameters
    # This will be overwritten by the specifications
    _nparams = 1

    def __init__(self, lattice: Union[Lattice2D, Lattice3D], g_el: float, g_mag: float, g_int: float,  g_mass: float, nlayer: int = 1):
        """Constructor.

        Args:
            lattice (Union[Lattice2D, Lattice3D]): lattice. 
            g_el (float): Hamiltonian prefactor for electric energy
            g_mag (float): prefactor for magnetic energy
            g_int (float): prefactor for gauge-matter coupling
            g_mass (float): mass of physical fermions (i.e. prefactor on the mass term).
            nlayer (int, optional): number of layers. Defaults to 1.
        """
        # The parameters have the following order: [[t1,y1,z1],[t2,y2,z2],....]
        self.nlayer = nlayer
        self.lattice = lattice

        self._paramvec = None

        # Parameters of the Hamiltonian
        self.g_el = g_el
        self.g_mag = g_mag
        self.g_int = g_int
        self.g_mass = g_mass

    @property
    def paramvec(self):
        return self._paramvec

    @paramvec.setter
    def paramvec(self, val):
        if self.check_params(val):
            self._paramvec = val
            self.nlayer = len(val)
        else:
            logging.error("The set of parameters is not consistent.")
            sys.exit(1)

    def check_params(self, params):
        """Check the consistency of the input parameters.
        All arrays must have the same length.

        Args:
            params (list or np.ndarray): two dimensional array of input parameters
        """
        lenvec = np.asarray([len(x) for x in params])
        # We know that we need _nparams parameters for each layer
        return np.all(lenvec == self._nparams)

    def nvarparams(self):
        return self._nparams*self.nlayer

    def print_parametervec(self, symbolvec):
        """Printing of the parametervec

        Args:
            symbolvec (list): List of the symbolvecs
        """
        for ind in range(self.nlayer):
            for symb, val in zip(symbolvec, self._paramvec[ind]):
                print(str(symb), val)

    @abstractmethod
    def make_pure_gauge(self):
        """Ensure that the system is pure gauge, i.e. no physical fermions.
        This abstract method must be overwritten by a subclass.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")
    
    def enforce_parameter_conditions(self, mat):
        """In some cases, there are extra conditions we wish to impose on the parameters."""
        return

################## Utility Functions ######################


def extract_partial_covmats(mat, corner):
    """Extract the partial covariance matrices from a gaussian mapping

    Args:
        mat (np.ndarray): Full covariance matrix
        corner (int): Index of the top left element of the bottom right matrix

    Returns:
        tuple: Matrices (A,B,D)
    """
    mat_a = mat[:corner, :corner]
    mat_b = mat[:corner, corner:]
    mat_d = mat[corner:, corner:]
    return mat_a, mat_b, mat_d


def calculate_lognormvec(gamma_in_sys_vec: List[np.ndarray], mat_d_vec: np.ndarray, all_factors=False) -> float:
    # This is still the plain formula, without any update mechanism
    nlayer = len(mat_d_vec)
    dest = np.zeros(nlayer)

    for ind in range(nlayer):
        gamma_in_sys = gamma_in_sys_vec[ind]
        mat_d = mat_d_vec[ind]
        if all_factors:
            sign, logval = np.linalg.slogdet(
                (np.eye(mat_d.shape[0]) - gamma_in_sys @ mat_d)) - mat_d.shape[0] * np.log(2)
        else:
            # We are skipping a global factor of 2**(-n) here, to get a reasonable size of the norm
            sign, logval = np.linalg.slogdet(
                (np.eye(mat_d.shape[0]) - gamma_in_sys @ mat_d))
        dest[ind] = logval
    # The factor 1/2 is the square-root
    return dest / 2


def calculate_lognorm(gamma_in_sys_vec: List[np.ndarray], mat_d_vec: np.ndarray, all_factors=False) -> float:
    # This is still the plain formula, without any update mechanism
    normvec = calculate_lognormvec(gamma_in_sys_vec, mat_d_vec, all_factors=all_factors)
    return np.sum(normvec)


def compute_grad_over_norm(gamma_in_sys: np.ndarray, diff: np.ndarray,
                           deriv_d: np.ndarray,
                           mat_d_inv: np.ndarray) -> float:
    """Compute the gradient of the norm divided by the norm.
    The expression of deriv_d given to this function decides which derivative is computed

    Args:
        gamma_in_sys (np.ndarray): Gauged covariance matrix of the projectors
        diff (np.ndarray): (D^{-1}-gamma_in_sys)^{-1}
        deriv_d (np.ndarray): dD/d{alpha}: Derivative of the virtual-virtual covariance matrix
        mat_d_inv (np.ndarray): Inverse of D: D^{-1}

    Returns:
        float: Gradient of the norm divided by the norm.
    """
    # Extract only the part of the virtual-virtual correlations
    dest = -0.5 * np.trace(gamma_in_sys @ deriv_d @ mat_d_inv @ diff)
    return dest


def calculate_lognormvec_inc(incdet_vec, det_mat_d_vec, n, all_factors=False):
    dest = []
    for ind in range(len(incdet_vec)):
        detval = incdet_vec[ind].det()
        if all_factors:
            detval -= n * np.log(2)
            detval += det_mat_d_vec[ind]
        # The factor 0.5 is the sqrt of the formula. We are storing the logarithm of the norm.
        # The addition of the cumval is the multiplication of the indpendent PEPS
        dest.append(0.5 * detval)
    return dest


def calculate_lognorm_inc(incdet_vec, det_mat_d_vec, n, all_factors=False):
    lognormvec = calculate_lognormvec_inc(incdet_vec,
                                          det_mat_d_vec,
                                          n,
                                          all_factors=all_factors)
    return np.sum(lognormvec)


################## System2DBase ######################

class System2DBase(ABC):
    """ Base class for two dimensional systems

    This class inherits from the abstract base class to enable abstract methods that has to be overwritten in a child class.
    This class cannot be instantiated directly.
    """

    def __init__(self, cfg: Config2DBase):
        self.cfg = cfg

        # Parameter based matrices
        self._symbolvec = None
        self._tmat_vec = None
        self._gamma_dirac_vec = None
        self._gamma_maj_vec = None
        self._gamma_maj_sys_vec = None

        # Partial covariance matrices
        self._mat_a_vec = None
        self._mat_b_vec = None
        self._mat_d_vec = None
        self._det_mat_d_vec = None
        self._mat_d_inv_vec = None

        # Full covariance matrix (gamma_out) of the fermions
        self._ferm_covmat = [None]*self.cfg.nlayer

        # Parameter dependent quantities for the electric energy
        self._mat_a_mod_vec = None
        self._mat_b_mod_vec = None
        self._mat_d_mod_vec = None
        self._det_mat_d_mod_vec = None
        self._mat_d_mod_inv_vec = None

        # Management of the gaugefields
        self._gamma_gauge_neutral_list_dict = None # list for various possible choices of projectors (not "vec", since that is used for layers), dict for directions
        self._gamma_in_sys_vec = None # in cases when different layers use the same projectors, all elements will point to the same gamma_in_sys
        self._gaugefieldvec = np.zeros(self.cfg.lattice.nlinks)
        self.gaugemgr = gauge.ZNGauge(2) # needs to be changed for cases other than Z2

        # Weight
        self._weight = None

        # Gradients
        self._gamma_maj_sys_deriv_dict = None
        self._el_energy_op_grad_vec = None
        self._mass_energy_op_grad_vec = None
        self._int_energy_op_grad_vec = None
        self._d_gamma_out_symbolvec = [None]*self.cfg.nlayer # gradients of gamma_out for all symbols
        self._grad_over_norm_dict = {(var,ind):None for var,ind in it.product(self.symbolvec,range(self.cfg.nlayer))}

        # Observables
        self._energy = None
        self._el_energy_op = None
        self._el_energy_op_vec = None
        self._mag_energy_op = None
        self._mass_energy_op = None
        self._mass_energy_op_vec = None
        self._int_energy_op = None
        self._int_energy_op_vec = None

        # Woodbury Update and Matrix Inversion
        self._wi_gamma_in_vec = None  # Tracks (D^-1 - gammain)^-1
        self._wi_gamma_out_vec = None  # Tracks (D - gammain)^-1
        self._incdet_vec = None  # Tracks det(D^-1 - gammain)

        self._wi_gamma_in_mod_vec = None  # Tracks (Dmod^-1 - gammain)^-1
        self._wi_gamma_out_mod_vec = None  # Tracks (Dmod - gammain)^-1
        self._incdet_mod_vec = None  # Tracks det(Dmod^-1 - gammain)

    def initialize(self):
        """Initialization function. 
        This is a good spot to copy essential data from the configuration.
        """
        return None

    def _exract_partial_covmatvec(self, offset):
        # We are assuming one physical mode per site
        mat_a_vec = []
        mat_b_vec = []
        mat_d_vec = []
        for ind in range(self.cfg.nlayer):
            mat_a, mat_b, mat_d = extract_partial_covmats(self.gamma_maj_sys_vec[ind], offset)
            mat_a_vec.append(mat_a)
            mat_b_vec.append(mat_b)
            mat_d_vec.append(mat_d)
        return mat_a_vec, mat_b_vec, mat_d_vec

    @abstractmethod
    def _create_symbolvec(self):
        """
        Function to define the list of parameters as sympy variables.
        We need these symbols to analytically derive T automatically.
        This function has to be overwritten in the child-class.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @property
    def symbolvec(self):
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
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    def compute_tmat_deriv(self, symb):
        """Return the derivative of the T matrix with respect to the symbol

        Args:
            symb (sympy.Symbol): Symbol to be derived with respect to

        Returns:
            np.array: Array of symbols
        """
        tmat_symb = self.tmat_symb
        return np.asarray(sympy.diff(tmat_symb, symb)).astype(complex)

    def _eval_tmat_symb(self, paramvec):
        """Compute the numerical representation of the T matrix

        Args:
            paramvec (list): List of parameter values (numerical)

        Returns:
            np.array: T matrix with numerical values
        """
        tmat_eval = self.tmat_symb.evalf(subs={self.symbolvec[i]:paramvec[i] for i in range(len(paramvec))})
        return np.asarray(tmat_eval).astype(complex)

    @property
    def tmat_vec(self):
        """
        Generate the T-matrix vector (single virtual fermion on the link).
        Analytically, this mode order is not advantageous, 
        but is makes the reshuffling of the modes easier for gamma_in and M_D in the covariance matrix.

        Returns:
            np.array: parameter matrix T
        """
        if self._tmat_vec is None:
            self._tmat_vec = [
                self._eval_tmat_symb(params) for params in self.cfg.paramvec
            ]
        return self._tmat_vec

    @property
    def gamma_dirac_vec(self):
        """Return the vector of covariance matrices in dirac modes.

        Returns:
            np.array: Vector of covariance matrices in Dirac modes
        """
        if self._gamma_dirac_vec is None:
            self._gamma_dirac_vec = [
                utils.tmat_to_covariance_matrix(tmat) for tmat in self.tmat_vec
            ]
        return self._gamma_dirac_vec

    @property
    def gamma_maj_vec(self):
        """Return the covariance matrix in Majorana modes.
        The definition of Majorana modes used is
            \\gamma_1=c+c^\\dagger
            \\gamma_2=i(c-c^\\dagger)

        This is a get function.

        Returns:
            np.array: Covariance matrix in Majorana modes
        """
        if self._gamma_maj_vec is None:
            # We know that the gamma dirac matrices have all the same shape
            m, _ = self.gamma_dirac_vec[-1].shape
            smat = utils.generate_smat(m)
            self._gamma_maj_vec = [
                np.real(smat @ gamma_dirac @ np.transpose(smat))
                for gamma_dirac in self.gamma_dirac_vec
            ]
        return self._gamma_maj_vec

    @abstractmethod
    def _expand_gamma_maj_to_system(self, covmat):
        """Expand the covariance matrix of a single site to the full system

        Args:
            covmat (np.array): Covariance matrix for the single site
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    def d_gamma_out_symbolvec(self, layer:int) -> np.ndarray:
        """Return a vector containing the derivatives of gamma_out (for the given layer) for each symbol.

        Returns:
            [List]: List of np.arrays, with length equal to the number of symbols.
        """
        if self._d_gamma_out_symbolvec[layer] is None:
            self._d_gamma_out_symbolvec[layer] = []
            offset = 2 * self.cfg.lattice.size

            for symbol in self.symbolvec:
                mat_b = self.mat_b_vec[layer]
                deriv_gamma_maj_sys = self.gamma_maj_sys_deriv_vec(symbol)[layer]
                d_mat_a, d_mat_b, d_mat_d = extract_partial_covmats(deriv_gamma_maj_sys, offset)
                diff_d_gamma_inv = self.wi_gamma_out_vec[layer].inv()
                d_gamma_out = d_mat_a \
                        + d_mat_b @ diff_d_gamma_inv @ np.transpose(mat_b) \
                        + mat_b @ diff_d_gamma_inv @ np.transpose(d_mat_b) \
                        - mat_b @ diff_d_gamma_inv @ d_mat_d @ diff_d_gamma_inv @ np.transpose(mat_b)
                self._d_gamma_out_symbolvec[layer].append(d_gamma_out)
        
        return self._d_gamma_out_symbolvec[layer]

    @property
    def gamma_maj_sys_vec(self):
        """Return the covariance matrix of the full system in Majorana modes.
        The mode order is changed to fit the mode order of gamma_in.
        See documentation of gamma_in for details.

        This is a get function.

        Returns:
            [np.array]: Covariance matrix of the full system
        """
        if self._gamma_maj_sys_vec is None:
            self._gamma_maj_sys_vec = [
                self._expand_gamma_maj_to_system(gamma_maj)
                for gamma_maj in self.gamma_maj_vec
            ]
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
            [np.array]: Correlations of the physcial modes for the full system.
        """
        if self._mat_a_vec is None:
            offset = 2 * self.cfg.lattice.size
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = self._exract_partial_covmatvec(offset)
        return self._mat_a_vec

    @property
    def mat_b_vec(self):
        """Extract the matrix for physical-virtual correlations.
        There is a vector of B matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            [np.array]: Correlations of the physcial modes with the virtual modes for the full system.
        """
        if self._mat_b_vec is None:
            offset = 2 * self.cfg.lattice.size
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = self._exract_partial_covmatvec(
                offset)
        return self._mat_b_vec

    @property
    def mat_d_vec(self):
        """Extract the matrix for virtual-virtual correlations (aka D).
        There is a vector of D matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            [np.array]: Correlations of the virtual modes for the full system.
        """
        if self._mat_d_vec is None:
            offset = 2 * self.cfg.lattice.size
            self._mat_a_vec, self._mat_b_vec, self._mat_d_vec = self._exract_partial_covmatvec(
                offset)
        return self._mat_d_vec

    @property
    def det_mat_d_vec(self):
        """Compute the determinant of the virtual-virtual correlation matrix.
        There is a vector of D matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            list: List of log-determinants
        """
        if self._det_mat_d_vec is None:
            self._det_mat_d_vec = [
                np.linalg.slogdet(mat_d)[1] for mat_d in self.mat_d_vec
            ]
        return self._det_mat_d_vec

    @property
    def mat_d_inv_vec(self):
        """Compute the vector of the inverses of the D matrix.
        The D matrix is the correlation matrix of virtual-virtual correlations.
        There is a vector of D matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            list: List of inverses of matrix invserses of D
        """
        if self._mat_d_inv_vec is None:
            self._mat_d_inv_vec = [
                np.linalg.inv(mat_d) for mat_d in self.mat_d_vec
            ]
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
            [np.array]: Correlations of the physcial modes for the full system.
        """
        if self._mat_a_mod_vec is None:
            offset = 2 * self.cfg.lattice.size + 2 * self.cfg.nvirtmodes_link
            self._mat_a_mod_vec, self._mat_b_mod_vec, self._mat_d_mod_vec = self._exract_partial_covmatvec(
                offset)
        return self._mat_a_mod_vec

    @property
    def mat_b_mod_vec(self):
        """Extract the matrix for physical-virtual correlations.
        This matrix contains one link less than the original matrix (used for the electric energy computation)
        There is a vector of B matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            [np.array]: Correlations of the physcial modes with the virtual modes for the full system.
        """
        if self._mat_b_mod_vec is None:
            offset = 2 * self.cfg.lattice.size + 2 * self.cfg.nvirtmodes_link
            self._mat_a_mod_vec, self._mat_b_mod_vec, self._mat_d_mod_vec = self._exract_partial_covmatvec(
                offset)
        return self._mat_b_mod_vec

    @property
    def mat_d_mod_vec(self):
        """Extract the matrix for virtual-virtual correlations.
        This matrix contains one link less than the original matrix (used for the electric energy computation)
        There is a vector of D matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            [np.array]: Correlations of the virtual modes for the full system.
        """
        if self._mat_d_mod_vec is None:
            offset = 2 * self.cfg.lattice.size + 2 * self.cfg.nvirtmodes_link
            self._mat_a_mod_vec, self._mat_b_mod_vec, self._mat_d_mod_vec = self._exract_partial_covmatvec(
                offset)
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
            self._det_mat_d_mod_vec = [
                np.linalg.slogdet(mat_d)[1] for mat_d in self.mat_d_mod_vec
            ]
        return self._det_mat_d_mod_vec

    @property
    def mat_d_mod_inv_vec(self):
        """Compute the inverse of modified D matrices.
        There is a vector of inverses of D matrices if multiple layers are used; len(vec)==# of copies
        This is a get function.

        Returns:
            np.array: List of inverses of modified D matrices
        """
        if self._mat_d_mod_inv_vec is None:
            self._mat_d_mod_inv_vec = [
                np.linalg.inv(mat_d) for mat_d in self.mat_d_mod_vec
            ]
        return self._mat_d_mod_inv_vec

    @abstractmethod
    def initialize_gamma_in_sys(self):
        """Abstract function to initialize gamma_in (the covariance matrix of the projectors) in a child class
        This function has to be overwritten in a child class.

        This function returns gamma_in_sys_vec even for cases where gamma_in_sys does not vary between layers.
        In that case, each element of gamma_in_sys_vec points to the same gamma_in_sys
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @property
    def gamma_in_sys(self):
        """Get function to return the gauged gamma_in_sys, the covariance matrix of the links for the whole system.
        This is required to maintain compatibility with early development, in which gamma_in did not vary between layers.
        Possibly the code should be modified to use gamma_in_sys_vec everywhere; this can be done without significant memory cost.

        Returns:
            np.array: Gauged covariance matrix of the system
        """
        if self._gamma_in_sys_vec is None:
            self._gamma_in_sys_vec, full_tuple, mod_tuple = self.initialize_gamma_in_sys()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            self._wi_gamma_in_mod_vec, self._wi_gamma_out_mod_vec, self._incdet_mod_vec = mod_tuple
        return self._gamma_in_sys_vec[0]

    @property
    def gamma_in_sys_vec(self):
        """Get function to return the gauged gamma_in_sys_vec, the covariance matrices of the links for the whole system for each layer.
        This function is required to allow for gamma_in to vary between layers.

        Returns:
            np.array: vector of gauged covariance matrices of the system
        """
        if self._gamma_in_sys_vec is None:
            self._gamma_in_sys_vec, full_tuple, mod_tuple = self.initialize_gamma_in_sys()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            self._wi_gamma_in_mod_vec, self._wi_gamma_out_mod_vec, self._incdet_mod_vec = mod_tuple
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
            self._gamma_in_sys_vec, full_tuple, mod_tuple = self.initialize_gamma_in_sys()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            self._wi_gamma_in_mod_vec, self._wi_gamma_out_mod_vec, self._incdet_mod_vec = mod_tuple
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
            self._gamma_in_sys_vec, full_tuple, mod_tuple = self.initialize_gamma_in_sys()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            self._wi_gamma_in_mod_vec, self._wi_gamma_out_mod_vec, self._incdet_mod_vec = mod_tuple
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
            self._gamma_in_sys_vec, full_tuple, mod_tuple = self.initialize_gamma_in_sys()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            self._wi_gamma_in_mod_vec, self._wi_gamma_out_mod_vec, self._incdet_mod_vec = mod_tuple
        return self._wi_gamma_out_vec

    @property
    def gamma_in_sys_mod(self):
        """Get function to return the gauged gamma_in_sys with a single link modification (to compute the electric energy), 
        the covariance matrix of the links for the whole system.

        Returns:
            np.array: Gauged, modified covariance matrix of the system
        """
        single_link_offset = 2 * self.cfg.nvirtmodes_link
        return self.gamma_in_sys[single_link_offset:, single_link_offset:]

    @property
    def gamma_in_sys_mod_vec(self):
        """Get function to return the gauged gamma_in_sys_vec with a single link modification (to compute the electric energy), 
        the covariance matrix of the links for the whole system.

        Returns:
            np.array: Gauged, modified covariance matrices of the system for each layer
        """
        gamma_in_sys_mod_vec = []
        single_link_offset = 2 * self.cfg.nvirtmodes_link # we can use the same offset for all layers, since all dimensions, mode ordering, etc. are the same
        for layer in range(self.cfg.nlayer):
            gamma_in_sys_mod_vec.append(self.gamma_in_sys_vec[layer][single_link_offset:, single_link_offset:])
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
            self._gamma_in_sys_vec, full_tuple, mod_tuple = self.initialize_gamma_in_sys()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            self._wi_gamma_in_mod_vec, self._wi_gamma_out_mod_vec, self._incdet_mod_vec = mod_tuple
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
            self._gamma_in_sys_vec, full_tuple, mod_tuple = self.initialize_gamma_in_sys()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            self._wi_gamma_in_mod_vec, self._wi_gamma_out_mod_vec, self._incdet_mod_vec = mod_tuple
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
            self._gamma_in_sys_vec, full_tuple, mod_tuple = self.initialize_gamma_in_sys()
            self._wi_gamma_in_vec, self._wi_gamma_out_vec, self._incdet_vec = full_tuple
            self._wi_gamma_in_mod_vec, self._wi_gamma_out_mod_vec, self._incdet_mod_vec = mod_tuple
        return self._wi_gamma_out_mod_vec

    ################## Computation of derivatives ######################

    def compute_gamma_dirac_deriv(self, symb: sympy.Symbol, layerind: int):
        """Return the numerical derivative of the gamma_dirac, the Dirac covariance matrix of one fiducial state.

        Args:
            symb (sympy.Symbol): Symbol with respect to which we derive
            layerind (int): index of the layer

        Returns:
            np.array: Derivative of gamma_dirac wrt to symb
        """
        deriv_t = self.compute_tmat_deriv(symb)
        tmat = self.tmat_vec[layerind]
        tmatc = np.conjugate(tmat)
        idttinv_minus = np.linalg.inv(np.eye(deriv_t.shape[0]) - tmat @ tmatc)
        idtt_plus = np.eye(deriv_t.shape[0]) + tmat @ tmatc
        d_idtt_minus = -(deriv_t @ tmatc + tmat @ np.conjugate(deriv_t))
        d_idtt_plus = -d_idtt_minus
        d_lt = idttinv_minus @ d_idtt_minus @ idttinv_minus @ tmat - idttinv_minus @ deriv_t
        d_rt = 0.5 * idttinv_minus @ d_idtt_plus @ idttinv_minus @ idtt_plus + \
            0.5 * idttinv_minus @ d_idtt_plus
        d_lb = -np.conjugate(d_rt)
        d_rb = -np.conjugate(d_lt)
        return 1.j*np.block([[d_lt, d_rt], [d_lb, d_rb]])

    def compute_gamma_maj_deriv(self, symb: sympy.Symbol, layerind: int):
        """Return the numerical derivative of the gamma_maj, the Majorana covariance matrix of one fiducial state.

        Args:
            symb (sympy.Symbol): Symbol with respect to which we derive
            layerind (int): index of the layer

        Returns:
            np.array: Derivative of gamma_maj wrt to symb
        """
        gamma_dirac_deriv = self.compute_gamma_dirac_deriv(symb, layerind)
        m, _ = gamma_dirac_deriv.shape
        smat = utils.generate_smat(m)
        return np.real(smat @ gamma_dirac_deriv @ np.transpose(smat))

    def _generate_gamma_maj_sys_deriv_dict(self):
        """Internal function to generate a dictionary of all possible derivatives of gamma_maj_sys, the system-wide covariance matrix of the fiducial state.
        The key to the dictionary is the symbol with respect to which we derived.
        Each entry contains a list with len(list) = nlayer.

        Returns:
            dict: Dictionary with all derivatives
        """
        dest = {}
        for symb in self.symbolvec:
            dest[symb] = [self._expand_gamma_maj_to_system(
                self.compute_gamma_maj_deriv(symb, i)) for i in range(self.cfg.nlayer)]
        return dest

    def gamma_maj_sys_deriv_vec(self, symb: sympy.Symbol) -> np.ndarray:
        """Return a list of derivatives of all layers of gamma_maj_sys with respect to a given symbol.
        This is a get function.

        Args:
            symb (sympy.Symbol): Symbol with repsect to which we derive

        Returns:
            np.ndarray: List of derived gamma_maj_sys
        """
        if symb in self.symbolvec:
            if self._gamma_maj_sys_deriv_dict is None:
                self._gamma_maj_sys_deriv_dict = self._generate_gamma_maj_sys_deriv_dict()
            return self._gamma_maj_sys_deriv_dict[symb]
        else:
            print("gamma_maj_sys_deriv: Invalid variable name", sys.stderr)
        return None

    def compute_grad_norm_vec(self) -> np.ndarray:
        """Compute the gradient of the norm for all layers with respect to all parameters.
        The parameter order is [[dt1, dy1, dz1...],[dt2,dy2,dz2...]...]

        Returns:
            np.ndarray: Vector of gradients of the norm with respect to all parameters
        """
        dest = []
        for layerind in range(self.cfg.nlayer):
            dest.append(self.compute_grad_norm(layerind))
        return np.asarray(dest)

    def compute_grad_norm(self, layerind: int) -> np.ndarray:
        """Compute the gradient of the norm for a given layer wrt to all parameters.
        The parameter order is the same as in the symbolvec [t1,y1,z1....]

        Args:
            layerind (int): layer index

        Returns:
            np.ndarray: Vector of gradients for the norm
        """

        dest = np.zeros(len(self.symbolvec))
        for ind, symbol in enumerate(self.symbolvec):
            dest[ind] = self.compute_grad_over_norm(symbol, layerind)
        return dest

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
        """ Setter of the weight """
        self._weight = val

    def calculate_weight_attempt(self, link_ind: int, theta: float, all_factors=False):
        """Compute the weight of an update attempt in which the link index link_ind is substituted for theta
        The inclusion of all constant pre-factors can be switched on and off.
        
        Currently this function does not work when physical fermions are included, because it does not use the 
        correct projectors. However, this function is not currently set to be used outside of tests.

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
        gamma_neutral_gauge_layers = self.gamma_gauge_neutral
        gamma_in_subst_layers = [rotmat @ gamma_neutral_gauge[dir] @ np.transpose(rotmat) for gamma_neutral_gauge in gamma_neutral_gauge_layers]
        updates = [self.calculate_update_gamma_in(ind_mat, gamma_in_subst, gamma_in_sys) for gamma_in_subst, gamma_in_sys in zip(gamma_in_subst_layers, self.gamma_in_sys_vec) ] 
        return self.update_lognorm_inc(ind_mat, updates, all_factors)

    def calculate_lognorm(self, all_factors=False):
        """Compute the logarithm of the norm

        Args:
            all_factors (bool, optional): Include all constant prefactors. Defaults to False.

        Returns:
            float: Logarithm of the norm
        """
        return calculate_lognorm(self.gamma_in_sys_vec, self.mat_d_vec, all_factors=all_factors)
    
    def calculate_lognormvec(self, all_factors=False):
        """Compute the logarithm of the norm for each layer

        Args:
            all_factors (bool, optional): Include all constant prefactors. Defaults to False.

        Returns:
            float: Logarithm of the norm
        """
        return calculate_lognormvec(self.gamma_in_sys_vec, self.mat_d_vec, all_factors=all_factors)

    def calculate_lognormvec_inc(self, all_factors=False):
        """Compute the logarithm of the norm for all layers by incrementally updating the previous value (using IncDet and Woodbury)

        Args:
            all_factors (bool, optional): Include all pre-factors in the computation. Defaults to False.

        Returns:
            list: Vector of the incrementally updated norms for all layers
        """
        return calculate_lognormvec_inc(self.incdet_vec, self.det_mat_d_vec, self.gamma_in_sys.shape[0], all_factors=all_factors)

    def calculate_lognorm_inc(self, all_factors=False):
        """Update the logarithm of the norm incrementally (using IncDet and Woodbury)

        Args:
            all_factors (bool, optional): Include all pre-factors in the computation. Defaults to False.

        Returns:
            float: Logarithm of the norm (computed from IncDet and Woodbury)
        """
        normvec = self.calculate_lognormvec_inc(all_factors=all_factors)
        return np.sum(normvec)

    def update_lognorm_inc(self, offset, updates, all_factors=False):
        """Updat the logarithm of the norm incrementally with the given update.

        Args:
            offset (int): Offset into the matrix.
            update (np.array): Update matrix to replace the current sub-matrix
            all_factors (bool, optional): Include all constant pre-factors. Defaults to False.

        Returns:
            float: Updated logarithmic value of the norm
        """
        cumval = 0
        for ind in range(self.cfg.nlayer):
            detval = self.incdet_vec[ind].update_index(self.wi_gamma_in_vec[ind].inv(), updates[ind], offset, offset, store=False)
            if all_factors:
                detval -= self.gamma_in_sys.shape[0]*np.log(2)
                detval += np.linalg.slogdet(self.mat_d_vec[ind])[1]
            # The factor 0.5 is the sqrt of the formula. We are storing the logarithm of the norm.
            # The addition of the cumval is the multiplication of the indpendent PEPS
            cumval += 0.5 * detval
        return cumval

    def compute_grad_over_norm(self, var: sympy.Symbol, layerind: int) -> float:
        """Compute the quotient of derivative of the norm over the norm itself.
        We can avoid a lot of factors by computing the quotient directly.

        Args:
            var (sympy.Symbol): Name of the variable
            layerind (int): Index of the layer

        Returns:
            float: Value of the gradient divided by the norm of the state
        """
        if self._grad_over_norm_dict[(var,layerind)] is None:
            diff = self.wi_gamma_in_vec[layerind].inv()
            # 2 phys. Majorana modes per vertex, this is indepent of the number of copies or layers
            offset = 2 * self.cfg.lattice.size
            # Extract only the part of the virtual-virtual correlations
            deriv_d = self.gamma_maj_sys_deriv_vec(var)[layerind][offset:, offset:]
            mat_d_inv = self.mat_d_inv_vec[layerind]

            # TODO: We might save one matrix-matrix multiplication here
            # The derivd and mat_d_inv are constant
            self._grad_over_norm_dict[(var,layerind)]=compute_grad_over_norm(self.gamma_in_sys_vec[layerind], diff, deriv_d, mat_d_inv)
        return self._grad_over_norm_dict[(var,layerind)]

    ################## Local Gauge ######################

    @property
    def gaugefieldvec(self):
        """Return the vector of gauge fields.

        Returns:
            np.ndarray: Gauge fields of the simulation
        """
        return self._gaugefieldvec

    @gaugefieldvec.setter
    def gaugefieldvec(self, val):
        print(
            "Do not set the gaugefieldvec explicitly. Use 'update_gauge_ind'.", file=sys.stderr)

    @property
    def gamma_gauge_neutral(self):
        if not self._gamma_gauge_neutral_list_dict:
            self._gamma_gauge_neutral_list_dict = self._generate_gamma_gauge_neutral_dict()
        return self._gamma_gauge_neutral_list_dict

    @abstractmethod
    def _generate_gamma_gauge_neutral_dict(self):
        """Abstract method to define the ungauged covariance matrix of a single link.
        The substitution method must ensure a consistent order of the modes.
        The direction parameter controls which covariance matrix is retrieved, since these can differ between directions.
        This method must be overwritten in a subclass.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @abstractmethod
    def generate_rotmat(self, theta, coord, dir):
        """Abstract method to define the rotation matrix of a single link.
        The substitution method must ensure a consistent order of the modes.
        This method must be overwritten in a subclass.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @abstractmethod
    def update_gauge_ind(self, link_ind, theta):
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    def update_gauge_full_system(self, gaugeconfig):
        """Replace all gauge fields on the links by the values given in gaugeconfig.

        Args:
            gaugeconfig (np.array): Array of new values for the gauge field
        """
        for ind, gauge in enumerate(gaugeconfig):
            self.update_gauge_ind(ind, gauge)

    def update_gauge_coord(self, coord, dir, theta):
        """Update a gauge field at a given coordinate and direction by a new value

        Args:
            coord (tuple): Coordinate of the vertex
            dir (Direction): Direction of the link
            theta (float): New value for the gauge field
        """
        ind = self.cfg.lattice.coord2ind_dir(coord, dir)
        self.update_gauge_ind(ind, theta)

    def calculate_update_gamma_in(self, offset, update_mat, gamma_in_sys=None):
        """Compute an update between the current gamma_in and the new gamma_in

        Args:
            offset (int): Offset in the matrix
            update_mat (np.array): Array to replace the current content of gamma_in at offset
            gamma_in_sys (np.array): gamma_in_sys. This is given as an argument so that different gamma_in_sys can be passed in when gamma_in_sys differs between layers.

        Returns:
            np.array: Additional update to reach update_mat at gamma_in[offset:,offset:]
        """
        if gamma_in_sys is None:
            gamma_in_sys = self.gamma_in_sys # take the first element, which is shared between all the layers
        m_up, n_up = update_mat.shape
        gamma_in_old = gamma_in_sys[offset:offset + m_up,
                                    offset:offset + n_up] 
        return -(update_mat - gamma_in_old)

    def invalidate_gauge_update(self):
        """Reset the values of computed quantitities to avoid spillover from previous computations.
        """
        self._ferm_covmat = [None]*self.cfg.nlayer # maybe it's possible to update this locally?
        self._d_gamma_out_symbolvec = [None]*self.cfg.nlayer # maybe it's possible to update this locally?
        
        self._energy = None
        self._el_energy_op = None
        self._el_energy_op_vec = None
        self._mag_energy_op = None
        self._mass_energy_op = None
        self._mass_energy_op_vec = None
        self._int_energy_op = None
        self._int_energy_op_vec = None
        
        self._el_energy_op_grad_vec = None
        self._mass_energy_op_grad_vec = None
        self._int_energy_op_grad_vec = None
        self._grad_over_norm_dict = {(var,ind):None for var,ind in it.product(self.symbolvec,range(self.cfg.nlayer))}

    ################## Observables ######################
    @abstractmethod
    def _compute_mag_energy_op(self):
        """Compute the bare operator (without shift) of the magnetic energy.
        This is an abstract method and has to be overwritten in a subclass.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @abstractmethod
    def _compute_el_energy_op_vec_and_grad(self):
        """Compute the electric energy and the gradient in each layer.
        This is an abstract method and has to be overwritten in a subclass.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")
    
    @abstractmethod
    def _compute_mass_energy_op_vec_and_grad(self):
        """Compute the mass energy and the gradient (for a single layer).
        This is an abstract method and has to be overwritten in a subclass.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @abstractmethod
    def _compute_int_energy_op_vec_and_grad(self):
        """Compute the interaction energy and the gradient (for a single layer).
        This is an abstract method and has to be overwritten in a subclass.
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")



    ################## Energy Calculations ######################
    @property
    def energy(self):
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
        return self._energy

    # Functions that return a term of the energy in the Hamiltonian, including all prefactors and energy from the entire lattice.
    @property
    def el_energy(self):
        """Compute electric energy with shift for the whole system
        This is a get function.

        Returns:
            float: electric energy
        """
        nlinks = self.cfg.lattice.nlinks
        el_energy = self.cfg.g_el * 2 * (nlinks - self.el_energy_op)
        return el_energy
    
    @property
    def mag_energy(self):
        """Compute magnetic energy with shift for the whole system
        This is a get function.

        Returns:
            float: magnetic energy
        """
        nplaq = self.cfg.lattice.nplaquettes
        mag_energy = self.cfg.g_mag * 2 * (nplaq - self.mag_energy_op)
        return mag_energy

    @property
    def mass_energy(self):
        """Compute mass energy for the whole system
        This is a get function.

        Returns:
            float: mass energy
        """
        mass_energy = self.cfg.g_mass * self.mass_energy_op
        return mass_energy

    @property
    def int_energy(self):
        """Compute interaction (of matter and gauge fields) energy for the whole system
        This is a get function.

        Returns:
            float: interaction energy
        """
        int_energy = self.cfg.g_int * self.int_energy_op
        return int_energy

    # Functions that return the energy for the operator part of a term in the Hamiltonian, including the energy for the entire lattice, but not any shifts or prefactors.
    @property
    def el_energy_op(self):
        """Compute electric energy (w/o shift) for the whole system.
        This is a get function.

        Returns:
            float: electric energy for the whole system w/o shift
        """
        if self._el_energy_op is None:
            # The different layers can be separated into separate PEPS and then multiplied together.
            nlinks = self.cfg.lattice.nlinks
            self._el_energy_op = nlinks * np.prod(self.el_energy_op_vec)
        return self._el_energy_op
    
    @property
    def mag_energy_op(self):
        """Compute the magnetic energy operator for the whole system without shift.
        This is a get function.

        Returns:
            float: Magnetic energy operator (w/o shift) for the whole system
        """
        if self._mag_energy_op is None:
            nplaq = self.cfg.lattice.nplaquettes
            self._mag_energy_op = nplaq * self._compute_mag_energy_op()
        return self._mag_energy_op

    @property
    def mass_energy_op(self):
        """Compute the mass energy operator for the whole system without shift.
        This is a get function.

        Returns:
            float: Mass energy operator (w/o shift) for the whole system
        """
        if self._mass_energy_op is None:
            nsites = self.cfg.lattice.size
            self._mass_energy_op = np.prod(self.mass_energy_op_vec) # don't multiply by the number of sites; for the mass term this is assumed to happen lower down in the stack.
        return self._mass_energy_op

    @property
    def int_energy_op(self):
        """Compute the interaction energy operator for the whole system without shift.
        This is a get function.

        Returns:
            float: Interaction energy operator (w/o shift) for the whole system
        """
        if self._int_energy_op is None:
            nsites = self.cfg.lattice.size
            self._int_energy_op = np.prod(self.int_energy_op_vec)
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
            self._el_energy_op_vec, self._el_energy_op_grad_vec = self._compute_el_energy_op_vec_and_grad()
        return self._el_energy_op_vec
    
    @property
    def mass_energy_op_vec(self):
        """Compute mass energy operator w/o shift for all layers for the whole system.
        This is a get function.

        Returns:
            list: Layer-resolved mass energy w/o shift
        """
        if self._mass_energy_op_vec is None:
            self._mass_energy_op_vec, self._mass_energy_op_grad_vec = self._compute_mass_energy_op_vec_and_grad()
        return self._mass_energy_op_vec
    
    @property
    def int_energy_op_vec(self):
        """Compute interaction energy operator w/o shift for all layers for the whole system.
        This is a get function.

        Returns:
            list: Layer-resolved interaction energy w/o shift
        """
        if self._int_energy_op_vec is None:
            # This vector is the electric energy on a single site.
            self._int_energy_op_vec, self._int_energy_op_grad_vec = self._compute_int_energy_op_vec_and_grad()
        return self._int_energy_op_vec

    # Functions that return the layer-resolved gradients of each energy operator
    @property
    def el_energy_op_grad_vec(self):
        """Compute the gradient of the electric operator (w/o shift) for all layers

        Returns:
            list: List of all electric energy gradients (w/o shift)
        """
        if self._el_energy_op_grad_vec is None:
            self._el_energy_op_vec, self._el_energy_op_grad_vec = self._compute_el_energy_op_vec_and_grad()
        return self._el_energy_op_grad_vec
    
    @property
    def mass_energy_op_grad_vec(self):
        """Compute the gradient of the mass energy operator for the whole system without shift.
        This is a get function.

        Returns:
            float: gradient of the mass energy operator (w/o shift) for the whole system
        """
        if self._mass_energy_op_grad_vec is None:
            self._mass_energy_op_vec, self._mass_energy_op_grad_vec = self._compute_mass_energy_op_vec_and_grad()
            self._mass_energy_op_grad_vec *= self.cfg.lattice.size
        return self._mass_energy_op_grad_vec
    
    @property
    def int_energy_op_grad_vec(self):
        """Compute the gradient of the interaction energy operator for the whole system without shift.
        This is a get function.

        Returns:
            float: Gradient of the interaction energy operator (w/o shift) for the whole system
        """
        if self._int_energy_op_vec is None:
            self._int_energy_op_vec, self._int_energy_op_grad_vec = self._compute_int_energy_op_vec_and_grad()
            # Do for whole system...
        return self._int_energy_op_grad_vec




    ##################  ######################

    @property
    def number_per_site(self):
        """Compute the occupation number per site. 
        Since we assume translation invariance, this can be simply calculated from the mass energy op.
        We don't store the occupation number per site, since it is cheap to calculate (just one division).

        Returns:
            float: the occupation number per site
        """
        return self.mass_energy_op / self.cfg.lattice.size 

    def compute_path(self, path):
        """Compute the observable corresponding the path given as an argument

        Args:
            path (list): List of tuples [(index,conj),....]. conj indicates whether the argument should be conjugated.
            This is the case if the link is traversed from right to left or from top to bottom.
        """
        theta_sum = 0.
        for ind, conj in path:
            if conj:
                theta_sum -= self.gaugefieldvec[ind]
            else:
                theta_sum += self.gaugefieldvec[ind]
        return np.exp(1.j*theta_sum)

    def compute_ferm_cov(self, layer:int) -> np.ndarray:
        """Compute the covariance matrix of the fermions in the system for the given layer

        Args:
            layer (int): the layer for which the covmat should be calculated
        """
        if self._ferm_covmat[layer] is None:
            self._ferm_covmat[layer] = self.mat_a_vec[layer] + (self.mat_b_vec[layer] @ self.wi_gamma_out_vec[layer].inv() @ np.transpose(self.mat_b_vec[layer]))
        return self._ferm_covmat[layer]



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
        num_copies = self.cfg.ncopy # The ncopy property is defined the config of any child class of System2DBase
        mode_order = []

        # Horizontal first
        for link in range(lat.nx * lat.ny):
            for c in range(num_copies):
                copy = c + 1
                mode1 = ( "l1", copy, link ) # majorana mode l1
                mode2 = ( "l2", copy, link ) # majorana mode l2
                mode3 = ( "r1", copy, link )
                mode4 = ( "r2", copy, link )
                mode_order += [ mode1, mode2, mode3, mode4 ]
        
        # Vertical
        for link in range(lat.nx * lat.ny):
            link_num = link + lat.nx * lat.ny # vertical link numbers start at the number of horizontal links that there are
            for c in range(num_copies):
                copy = c + 1
                mode1 = ( "d1", copy, link_num ) # majorana mode d1
                mode2 = ( "d2", copy, link_num ) # majorana mode d2
                mode3 = ( "u1", copy, link_num )
                mode4 = ( "u2", copy, link_num )
                mode_order += [ mode1, mode2, mode3, mode4 ]

        # Convert to a list of strings
        # This was left as a tupple above in case there was ever any use for that format
        mode_order_str = []
        for mode in mode_order:
            mode_str = mode[0] + "_" + str(mode[1]) + "_" + str(mode[2])
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
        and then the same thing for the second copy (if there is one).

        Returns:
            list: List of strings of the form <mode_letter:majorana mode>_<copy>_<link_id>
        """

        lat = self.cfg.lattice
        num_copies = self.cfg.ncopy # The ncopy property is defined the config of any child class of System2DBase
        mode_order = []

        for site in range(lat.nx * lat.ny):
            for c in range(num_copies):
                copy = c + 1
                x, y = lat.ind2coord(site) # coordinates of the site

                # Horizontal
                mode1 = ("l1", copy, lat.coord2ind_dir( (x-1,y), Direction.X ) )
                mode2 = ("l2", copy, lat.coord2ind_dir( (x-1,y), Direction.X ) )
                mode3 = ("r1", copy, lat.coord2ind_dir( (x,y), Direction.X ) )
                mode4 = ("r2", copy, lat.coord2ind_dir( (x,y), Direction.X ) )

                # Vertical
                mode5 = ("d1", copy, lat.coord2ind_dir( (x,y-1), Direction.Y ) )
                mode6 = ("d2", copy, lat.coord2ind_dir( (x,y-1), Direction.Y ) )
                mode7 = ("u1", copy, lat.coord2ind_dir( (x,y), Direction.Y ) )
                mode8 = ("u2", copy, lat.coord2ind_dir( (x,y), Direction.Y ) )

                mode_order += [ mode1, mode2, mode3, mode4, mode5, mode6, mode7, mode8 ]
        
        # Convert to a list of strings
        mode_order_str = []
        for mode in mode_order:
            mode_str = mode[0] + "_" + str(mode[1]) + "_" + str(mode[2])
            mode_order_str.append(mode_str)

        return mode_order_str