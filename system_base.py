import numpy as np
import logging
import sys

class Z2System2DConfigBase():
    _nparams = 1

    def __init__(self, lattice, g2, g_gm, g_mag, nlayer=1):
        #The parameters have the following order: [[t1,y1,z1],[t2,y2,z2],....]
        self.nlayer = nlayer
        self.lattice = lattice

        self._parametervec = None

        #Parameters of the Hamiltonian
        self.g2 = g2
        self.g_el = g2/2
        if g_mag is None:
            self.g_mag = 1./(2*g2)
        else:
            self.g_mag = g_mag
        self.g_gm = g_gm

    @property
    def paramvec(self):
        return self._parametervec

    @paramvec.setter
    def paramvec(self,val):
        if self.check_params(val):
            self._parametervec=val
            self.nlayer = len(val)
        else:
            logging.error("The set of parameters is not consistent.")
            sys.exit(1)

    def check_params(self,params):
        """Check the consistency of the input parameters.
        All arrays must have the same length.

        Args:
            params (list or np.ndarray): two dimensional array of input parameters
        """
        lenvec = np.asarray([len(x) for x in params])
        #We know that we need _nparams parameters for each layer
        return np.all(lenvec == self._nparams)

    def nvarparams(self):
        return self._nparams*self.nlayer

################## Utility Functions ######################

def extract_partial_covmats(mat,corner):
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

def calculate_lognormvec(gamma_in_sys: np.ndarray,
                      mat_d_vec: np.ndarray,
                      all_factors=False) -> float:
    # This is still the plain formula, without any update mechanism
    nlayer=len(mat_d_vec)
    dest=np.zeros(nlayer)
    for ind in range(nlayer):
        mat_d = mat_d_vec[ind]
        if all_factors:
            sign, logval = np.linalg.slogdet(
                (np.eye(mat_d.shape[0]) - gamma_in_sys @ mat_d) / 2)
        else:
            # We are skipping a global factor of 2**(-n) here, to get a reasonable size of the norm
            sign, logval = np.linalg.slogdet(
                (np.eye(mat_d.shape[0]) - gamma_in_sys @ mat_d))
        dest[ind]= logval
    #The factor 1/2 is the square-root
    return dest / 2

def calculate_lognorm(gamma_in_sys: np.ndarray,
                      mat_d_vec: np.ndarray,
                      all_factors=False) -> float:
    # This is still the plain formula, without any update mechanism
    normvec=calculate_lognormvec(gamma_in_sys,mat_d_vec,all_factors=all_factors)
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