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


def pfaffian_LTL_stack(A, overwrite_a=False):
    """
    This is a vectorized implementation of the LTL algorithm for computing the Pfaffian of a batch of
    skew-symmetric matrices. It was created by starting with the pfapack implementation and batching it.

    Args:
        A: (B, n, n) stack of skew-symmetric matrices

    Returns:
        (B,): pfaffians
    """
    A = np.asarray(A)
    assert A.ndim == 3
    B, n, m = A.shape
    assert n == m and n > 0

    # skew-symmetry check
    assert np.max(np.abs(A + np.swapaxes(A, -1, -2))) < 1e-14

    if n % 2 == 1:
        return np.zeros(B, dtype=A.dtype)

    if not overwrite_a:
        A = A.copy()

    pf = np.ones(B, dtype=A.dtype)
    alive = np.ones(B, dtype=bool)

    for k in range(0, n - 1, 2):

        # pivot selection (vectorized over batch)
        col = np.abs(A[:, k + 1 :, k])
        kp = k + 1 + np.argmax(col, axis=1)

        # row/column swaps
        swap = kp != (k + 1)
        if np.any(swap):
            # swap the rows and columns in one go
            b = np.where(swap)[0]
            A[b, [k + 1], k:], A[b, kp[b], k:] = (
                A[b, kp[b], k:].copy(),
                A[b, [k + 1], k:].copy(),
            )
            A[b, k:, [k + 1]], A[b, k:, kp[b]] = (
                A[b, k:, kp[b]].copy(),
                A[b, k:, [k + 1]].copy(),
            )
            pf[b] *= -1  # every interchange corresponds to a "-"

        piv = A[:, k + 1, k]
        zero = piv == 0
        alive &= ~zero

        # early termination handled via mask
        if not np.any(alive):
            return np.zeros(B, dtype=A.dtype)

        tau = np.zeros((B, n - k - 2), dtype=A.dtype)
        tau[alive] = A[alive, k, k + 2 :] / A[alive, k, k + 1, None]

        pf[alive] *= A[alive, k, k + 1]

        if k + 2 < n:
            v = A[:, k + 2 :, k + 1]
            A[:, k + 2 :, k + 2 :] += tau[:, :, None] * v[:, None, :]
            A[:, k + 2 :, k + 2 :] -= v[:, :, None] * tau[:, None, :]

    pf[~alive] = 0
    return pf


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
    def pfaffian_vectorized(mat):
        return pfaffian_LTL_stack(mat)

    @staticmethod
    def calculate_lognormvec(gamma_in_sys_vec, mat_d_vec, all_factors=False):
        return calculate_lognormvec_numpy(gamma_in_sys_vec, mat_d_vec, all_factors=all_factors)
