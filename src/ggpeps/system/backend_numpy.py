############## NUMPY CPU VERSIONS ##############

from typing import Optional

import numpy as np
from pfapack import pfaffian as pf

import ggpeps
from ggpeps.system.backend_base import BackendBase


def derivative_pfaffian_numpy(mat: np.ndarray, d_mat: np.ndarray, pfaval: Optional[float] = None) -> float:
    """Compute the derivative of a Pfaffian of a matrix A.
    The explicit derivative dA/dx is given as a second argument

    The given formula is only valid if A is not singular.

    Args:
        mat (np.ndarray): Input Matrix A
        d_mat (np.ndarray): Derivative dA/dx
        pfaval (Optional[float]): Pfaffian value of mat, if already known. If None, it will be computed.

    Returns:
        float: d(Pf(A))/dx
    """
    if pfaval is None:
        pfaval = pf.pfaffian(mat)

    if not ggpeps.utils.isclose(pfaval, 0):
        return 0.5 * pfaval * np.trace(np.linalg.inv(mat) @ d_mat)
    else:
        return 0.0


def calculate_lognormvec_numpy(
    gamma_in_sys_vec: np.ndarray,
    mat_d_vec: np.ndarray,
    all_factors: bool = False,
) -> np.ndarray:
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


class BackendNumpy(BackendBase):
    """Backend for numpy."""

    backend_type = "numpy"

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
    def pfaffian(mat):
        return pf.pfaffian(mat)

    @staticmethod
    def calculate_lognormvec(gamma_in_sys_vec, mat_d_vec, all_factors=False):
        return calculate_lognormvec_numpy(gamma_in_sys_vec, mat_d_vec, all_factors=all_factors)
