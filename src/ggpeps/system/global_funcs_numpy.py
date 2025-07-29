############## NUMPY CPU VERSIONS ##############

import numpy as np
from pfapack import pfaffian as pf

import ggpeps
import ggpeps.utils as utils
from ggpeps.system.backend_base import BackendBase


def derivative_pfaffian_numpy(mat, d_mat, pfaval=None):
    """Compute the derivative of a Pfaffian of a matrix A.
    The explicit derivative dA/dx is given as a second argument

    The given formula is only valid if A is not singular.

    Args:
        mat (np.ndarray): Input Matrix A
        d_mat (np.ndarray): Derivative dA/dx

    Returns:
        np.ndarray: d(Pf(A))/dx
    """
    if pfaval is None:
        pfaval = pf.pfaffian(mat)

    if not ggpeps.utils.isclose(pfaval, 0):
        return 0.5 * pfaval * np.trace(np.linalg.inv(mat) @ d_mat)
    else:
        return 0.0


def calculate_lognormvec_numpy(
    gamma_in_sys_vec: list[np.ndarray],
    mat_d_vec: list[np.ndarray],
    all_factors: bool = False,
) -> float:
    # This is still the plain formula, without any update mechanism
    nlayer = len(mat_d_vec)
    dest = np.zeros(nlayer)

    for ind in range(nlayer):
        gamma_in_sys = gamma_in_sys_vec[ind]
        mat_d = mat_d_vec[ind]

        sign, logval = np.linalg.slogdet((np.eye(mat_d.shape[0]) - gamma_in_sys @ mat_d))

        if all_factors:
            logval -= mat_d.shape[0] * np.log(2)
        else:
            # We are skipping a global factor of 2**(-n) here, to get a reasonable size of the norm
            pass
        dest[ind] = logval

    # The factor 1/2 is the square-root
    return dest / 2


def compute_grad_over_norm_numpy(
    gamma_in_sys: np.ndarray,
    diff: np.ndarray,
    deriv_d: np.ndarray,
    mat_d_inv: np.ndarray,
) -> float:
    r"""Compute the gradient of the norm divided by the norm.
    The expression of deriv_d given to this function decides which derivative is computed

    The gradient of the norm divided by the norm is given by
        -0.5 * np.trace(gamma_in_sys @ deriv_d @ mat_d_inv @ diff)
    which is very expensive to calculate.
    To reduce the number of expensive matrix multiplications, we use the fact that
        Tr(A @ B.T) = \sum_ij a_ij b_ij
    i.e. trace of a square matrix which is the product of two real matrices can be rewritten as
    the sum of entry-wise products of their elements, i.e. as the sum of all elements of their Hadamard product [1].
    Note that for current systems, the input matrices are always real, but this should be checked if the system changes
    (e.g. for other groups).

    When using a GPU (in which case this function is not used) it is faster to do all the matrix multiplications
    and then take the trace.

    Refs:
        [1] Trace, Wikipedia, https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Trace_of_a_product

    Args:
        gamma_in_sys (np.ndarray): Gauged covariance matrix of the projectors
        diff (np.ndarray): (D^{-1} - gamma_in_sys)^{-1}
        deriv_d (np.ndarray): dD/d{alpha}: Derivative of the virtual-virtual covariance matrix
        mat_d_inv (np.ndarray): Inverse of D: D^{-1}

    Returns:
        float: Gradient of the norm divided by the norm.
    """
    A = gamma_in_sys @ deriv_d
    B = mat_d_inv @ diff
    dest = -0.5 * (A * B.T).sum()
    return dest


class BackendNumpy_Z2(BackendBase):
    """Backend for Z2 systems using numpy."""

    backend_type = "numpy"
    gauge_group = "Z2"

    def __init__(self) -> None:
        pass

    @staticmethod
    def array_assign(mat, inds, val):
        mat[inds] = val
        return mat

    @staticmethod
    def array_add(mat, inds, val):
        mat[inds] += val
        return mat

    @staticmethod
    def array_mult(mat, inds, val):
        mat[inds] *= val
        return mat

    @staticmethod
    def calculate_lognormvec(gamma_in_sys_vec, mat_d_vec, all_factors=False):
        return calculate_lognormvec_numpy(gamma_in_sys_vec, mat_d_vec, all_factors=all_factors)
