import os
import re
import sys
import gzip
import pickle
import logging
import subprocess  # Start process for git hash
from typing import Optional

import numba as nb
import pandas as pd
from scipy.sparse import issparse
from scipy.linalg import svd, block_diag

import numpy as np
import jax.numpy as jnp
from ggpeps import xnp as xnp

import py_pfaffian.jax
from pfapack import pfaffian as pf

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import ggpeps
import ggpeps.measurement as meas
from ggpeps.system.global_funcs_jax import derivative_pfaffian_jax
from ggpeps.system.global_funcs_numpy import derivative_pfaffian_numpy

logger = logging.getLogger(ggpeps.LOGGER_NAME)

# Global constants
paulix = np.array([[0, 1], [1, 0]])
pauliy = np.array([[0, -1.0j], [1.0j, 0]])
pauliz = np.array([[1, 0], [0, -1]])

# ========== Utility Functions ====================


def setup_logger(logger: logging.Logger, log_file: str, level: str, runner_msg: str = ""):
    log_file_handler = logging.FileHandler(log_file)
    h_stdout = logging.StreamHandler(stream=sys.stdout)
    h_stderr = logging.StreamHandler(stream=sys.stderr)
    h_stderr.addFilter(lambda record: record.levelno >= logging.WARNING)
    formatter = logging.Formatter(f"%(asctime)s [{runner_msg}%(levelname)s] %(message)s")
    h_stdout.setFormatter(formatter)
    log_file_handler.setFormatter(formatter)
    logger.addHandler(h_stdout)
    logger.addHandler(h_stderr)
    logger.addHandler(log_file_handler)
    logger.setLevel(level.upper())
    return


def fname2nlayer(fname):
    """Extract the number of layers from a filename"""
    pattern = r"(?<=nlayer_)[\d]*"
    result = re.search(pattern, fname)
    if result is not None:
        return int(result.group(0))
    else:
        return None


def fname2ncopy(fname):
    """Extract the number of copies from a filename"""
    pattern = r"(?<=ncopy_)[\d]*"
    result = re.search(pattern, fname)
    if result is not None:
        return int(result.group(0))
    else:
        return None


def fname2g(fname):
    """Extract the coupling from a filename"""
    pattern = r"(?<=g_)[\d]*\.[\d]*"
    result = re.search(pattern, fname)
    if result is not None:
        return float(result.group(0))
    else:
        return None


def fname2gel(fname):
    """Extract the electric coupling from a filename"""
    pattern = r"(?<=gel_)[\d]*\.[\d]*"
    result = re.search(pattern, fname)
    if result is not None:
        return float(result.group(0))
    else:
        return None


def fname2L(fname):
    """Extract the system size from a filename"""
    pattern = r"(?<=L_)[\d]*"
    result = re.search(pattern, fname)
    return int(result.group(0))


def isclose(x, y, rtol=1.0e-5, atol=1.0e-8):
    return abs(x - y) <= atol + rtol * abs(y)


def load_matrix_dat_fmt(path, is_complex=True):
    """Load matrix format exported from C++.

    Args:
        path (str): Path to file
        is_complex (bool, optional): Matrix is complex or not. Defaults to True.
    """
    complexptrn = re.compile(r"\(([^,\)]+),([^,\)]+)\)")

    def parse_complex(s):
        return complex(*map(float, complexptrn.match(s).groups()))

    def parse_real(s):
        return float(s)

    dest = []
    with open(path, "r") as f:
        for line in f:
            line_short = re.sub(" +", " ", line.strip())
            strvec = line_short.split(" ")
            numvec = []
            for s in strvec:
                if is_complex:
                    num = parse_complex(s)
                else:
                    num = parse_real(s)
                numvec.append(num)
            dest.append(numvec)
    return np.array(dest)


def merge_measurements(meas1: meas.Measurement, meas2: meas.Measurement):
    """Merge two measurements by merging their timeseries

    Args:
        meas1 (Measurement): First measurement
        meas2 (Measurement): Second measurement

    Returns:
        Measurement: Merged measurement
    """
    dest = meas.Measurement(meas1.name, meas1.binsize)
    dest.extend(meas1.get_timeseries())
    dest.extend(meas2.get_timeseries())
    return dest


def mergeDict(dict1, dict2):
    """Left Merge dictionaries that contain only lists and append lists if values are common"""
    dest = {}
    for key in dict1:
        if key in dict2:
            # We assume that there are only lists in the dictionaries
            dest[key] = merge_measurements(dict1[key], dict2[key])
        else:
            dest[key] = dict1[key]
    return dest


def print_columns(listvals, padding=4, header=False):
    """Print a multi-dimensional list in a table

    Args:
        listvals (list of lists): Input data
        padding (int, optional): Padding of the columns. Defaults to 4.
        header (bool, optional): Print a header on top of the table. Defaults to False.
    """
    col_width = max([len(str(word)) for row in listvals for word in row]) + padding
    for ind, row in enumerate(listvals):
        print("".join(str(word).ljust(col_width) for word in row))
        if header and ind == 0:
            print("")


def sizeof_fmt(num, suffix="B"):
    """Pretty print a size as mutliples of 1024."""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%3.1f %s%s" % (num, "Yi", suffix)


def get_git_hash():
    """Get the git hash of the current commit in the repository.

    Returns:
        str: git hash
    """
    # This assumes that .git is in the parent folder of util.py
    packagedir = os.path.dirname(os.path.realpath(__file__))
    srcdir = os.path.join(packagedir, os.path.pardir)
    rootdir = os.path.join(srcdir, os.path.pardir)
    gitdir = os.path.join(rootdir, ".git")
    githash = subprocess.check_output(["git", f"--git-dir={gitdir}", "rev-parse", "HEAD"])
    return githash.decode("utf-8").strip()


def extract_partial_covmats(mat: xnp.ndarray, corner: int):
    """Extract the partial covariance matrices from a gaussian mapping

    Args:
        mat (xnp.ndarray): Full covariance matrix
        corner (int): Index of the top left element of the bottom right matrix

    Returns:
        tuple: Matrices (A,B,D)
    """
    mat_a = mat[:corner, :corner]
    mat_b = mat[:corner, corner:]
    mat_d = mat[corner:, corner:]
    return mat_a, mat_b, mat_d


def select_except(arr, ind: int):
    """Return all elements of a list except the indicated one

    Args:
        arr (list/np.array): list of values
        ind (int): index

    Returns:
        np.array: Array with all elements of arr except for arr[ind]
    """
    # This function works only on the outer-most layer
    if isinstance(arr, list):
        arr = xnp.asarray(arr)
    mask = xnp.ones(len(arr), dtype=bool)
    if ggpeps.PREFERRED_BACKEND == "jax":  # TODO: handle based on type checking instead
        mask = mask.at[ind].set(False)
    else:
        mask[ind] = False
    return arr[mask]  # TODO: fix for JAX


def multiply_except(arr, ind: int):
    """Product of all array values except for arr[ind]

    Args:
        arr (list/np.arr): list of values
        ind (int): index

    Returns:
        float: Multiplication of all array values except for arr[ind]
    """
    if len(arr) > 1:
        others = select_except(arr, ind)
        return xnp.prod(others)
    else:
        # It does not make sense to execute this function with only one element
        return arr[0]


@nb.njit(cache=True)
def pfaffian_explicit_4x4_masked(mat, ind):
    i0, i1, i2, i3 = ind
    return (mat[i0, i1] * mat[i2, i3]) - (mat[i0, i2] * mat[i1, i3]) + (mat[i1, i2] * mat[i0, i3])


@nb.njit(cache=True)
def pfaffian_explicit_4x4(mat):
    return (mat[0, 1] * mat[2, 3]) - (mat[0, 2] * mat[1, 3]) + (mat[1, 2] * mat[0, 3])


# @nb.njit(cache=True)
def derivative_pfaffian_covariance_mat(pfarr, matvec, d_matvec):
    dest = 0.0
    for pfaval, mat, d_mat in zip(pfarr, matvec, d_matvec):
        if not isclose(pfaval, 0):
            mat_inv = xnp.linalg.inv(mat)
            dest += 0.5 * pfaval * xnp.trace(mat_inv @ d_mat)
    return dest


def derivative_pfaffian(mat, d_mat, pfaval=None):
    """Compute the derivative of a Pfaffian of a matrix A.
    The explicit derivative dA/dx is given as a second argument

    The given formula is only valid if A is not singular.

    Args:
        mat (xnp.ndarray): Input Matrix A
        d_mat (xnp.ndarray): Derivative dA/dx

    Returns:
        xnp.ndarray: d(Pf(A))/dx
    """
    # We assume the types of all the provided arguments match
    if isinstance(mat, jnp.ndarray):
        return derivative_pfaffian_jax(mat, d_mat, pfaval=pfaval)
    else:
        return derivative_pfaffian_numpy(mat, d_mat, pfaval=pfaval)


def get_obs_mean_df(df: pd.DataFrame, obs: str) -> float:
    """Get the mean of an observable from the summary dataframe.

    Args:
        obs (str): Name of the observable.
        df (pd.DataFrame): Summary dataframe.

    Returns:
        float: Mean value of the observable.
    """
    return df.loc[df["name"] == obs, "mean"].values[0]


def save_summary_df(df, fname_summary: str):
    """Save the evaluation summary to a given filename

    Args:
        df (pd.DataFrame): Dataframe containing the summary
        fname_summary (str): Output filename for the summary
    """
    df.to_pickle(fname_summary)


# =========== Matrix Evaluation Functions ====================


def is_hermitian(mat):
    """Returns true if the matrix is hermitian."""
    if issparse(mat):
        return xnp.allclose(mat.todense(), mat.H.todense())
    else:
        return xnp.allclose(xnp.conjugate(xnp.transpose(mat)), mat)


def is_diagonal(mat):
    """Returns true if the matrix is diagonal."""
    if issparse(mat):
        return xnp.allclose((mat - mat.diagonal()).todense(), xnp.zeros(mat.shape))
    else:
        return xnp.allclose(mat - xnp.diag(xnp.diag(mat)), xnp.zeros_like(mat))


def is_symmetric(mat):
    """Returns true if the matrix is symmetric."""
    if issparse(mat):
        return xnp.allclose(mat.todense(), mat.T.todense())
    else:
        return xnp.allclose(xnp.transpose(mat), mat)


def is_permutation(mat):
    """Returns true if the matrix is a permutation matrix."""
    n, m = mat.shape
    if issparse(mat):
        raise NotImplementedError("Checking for sparse permutation matrices is not implemented.")
    else:
        square = n == m
        id = xnp.allclose(xnp.eye(n), mat @ xnp.transpose(mat))
        sum_rows = xnp.all(xnp.sum(mat, axis=0) == 1)
        sum_cols = xnp.all(xnp.sum(mat, axis=1) == 1)
        return square and id and sum_rows and sum_cols


def is_antisymmetric(mat, rtol: float = 1e-5, atol: float = 1e-8):
    """Returns true if the matrix mat is anti-symmetric."""
    if issparse(mat):
        return xnp.allclose(mat.todense(), -mat.T.todense(), rtol=rtol, atol=atol)
    else:
        return xnp.allclose(-xnp.transpose(mat), mat, rtol=rtol, atol=atol)


def is_covmat(mat: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Returns true if the given matrix satisfies all the conditions to be a covariance matrix."""
    m, n = mat.shape
    if (
        m == n
        and is_antisymmetric(mat, rtol=rtol, atol=atol)
        and xnp.allclose(mat @ mat, -xnp.eye(m), rtol=rtol, atol=atol)
        and xnp.allclose(mat @ xnp.transpose(mat), xnp.eye(m), rtol=rtol, atol=atol)
    ):
        # note that the last check should be mat @ mat^dagger = 1, but transpose gets
        # the same information for a matrix with real elements
        return True
    return False


def anti_symmetrize(mat):
    """Force a matrix to be anti-symmetirc."""
    return 0.5 * (mat - mat.T)


def get_nonzero_fraction(mat):
    """Returns fraction of non-zero elements."""
    return xnp.count_nonzero(mat) / xnp.prod(mat.shape)


def herm_conj(mat):
    """Returns the hermitian conjugate of a matrix."""
    return xnp.conjugate(xnp.transpose(mat))


def commutator(mat1, mat2):
    """Calculate the commutator of two matrices

    Args:
        mat1 (2d np.ndarray): First argument of commutator
        mat2 (2d np.ndarray): Second argument of commutator

    Returns:
        2d np.ndarray: Commutator
    """
    return (mat1 @ mat2) - (mat2 @ mat1)


def anticommutator(mat1, mat2):
    """Calculate the anti-commutator of two matrices

    Args:
        mat1 (2d np.ndarray): First argument of anti-commutator
        mat2 (2d np.ndarray): Second argument of anti-commutator

    Returns:
        2d np.ndarray: Anti-commutator
    """
    return (mat1 @ mat2) + (mat2 @ mat1)


# =========== Covariance Utility Funcitons ===========


def tmat_to_covariance_matrix(tmat: np.ndarray) -> np.ndarray:
    """Transforms a T matrix into the corresponding covariance matrix in terms of Dirac modes.
    This function assumes that the fiducial operator has a certain form: A=exp(T_{ij}a_i^\dagger a_j^\dagger)

    Args:
        tmat (np.array): Matrix of parameters

    Returns:
        np.array: Covariance matrix in terms of Dirac modes
    """
    m, n = tmat.shape
    id = xnp.eye(m)
    idinv = xnp.linalg.inv(id - tmat @ xnp.conjugate(tmat))
    lt = -idinv @ tmat
    rt = 0.5 * idinv @ (id + tmat @ xnp.conjugate(tmat))
    lb = -xnp.conjugate(rt)
    rb = -xnp.conjugate(lt)
    return 1.0j * xnp.block([[lt, rt], [lb, rb]])


def generate_smat(n: int):
    r"""Generate matrix to transform Dirac modes into Majorana modes.
    The function assumes the modes order of [a_1, a_2,....., a_n, a_1^\dagger,.....,a_n^\dagger].

    Args:
        n (int): Size of the matrix

    Returns:
        np.array: n x n matrix
    """
    pattern = xnp.array([[1], [1.0j]])
    halfmat = xnp.kron(np.eye(n // 2), pattern)
    return xnp.block([halfmat, xnp.conjugate(halfmat)])


def compute_grad_over_norm(
    gamma_in_sys: np.ndarray,
    diff: np.ndarray,
    deriv_d: np.ndarray,
    mat_d_inv: np.ndarray,
    method: str = "hadamard",
) -> float:
    r"""Compute the gradient of the norm divided by the norm.
    The expression of deriv_d given to this function decides which derivative is computed.

    The gradient of the norm divided by the norm is given by
        -0.5 * np.trace(gamma_in_sys @ deriv_d @ mat_d_inv @ diff)
    which is very expensive to calculate.
    To reduce the number of expensive matrix multiplications, we use the fact that
        Tr(A @ B.T) = \sum_ij a_ij b_ij
    i.e. trace of a square matrix which is the product of two real matrices can be rewritten as
    the sum of entry-wise products of their elements, i.e. as the sum of all elements of their Hadamard product [1].
    Note that for current systems, the input matrices are always real, but this should be checked if the system changes
    (e.g. for other groups).

    When using a GPU it is faster to do all the matrix multiplications
    and then take the trace.

    The choice of which method to use is given by the `method` parameter.

    Refs:
        [1] Trace, Wikipedia, https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Trace_of_a_product

    Args:
        gamma_in_sys (np.ndarray): Gauged covariance matrix of the projectors
        diff (np.ndarray): (D^{-1} - gamma_in_sys)^{-1}
        deriv_d (np.ndarray): dD/d{alpha}: Derivative of the virtual-virtual covariance matrix
        mat_d_inv (np.ndarray): Inverse of D: D^{-1}
        method (str, optional): Method to use to compute the gradient over norm.

    Returns:
        float: Gradient of the norm divided by the norm.
    """
    if method == "hadamard":
        A = gamma_in_sys @ deriv_d
        B = mat_d_inv @ diff
        dest = -0.5 * (A * B.T).sum()
    elif method == "trace":
        dest = -0.5 * xnp.trace(xnp.matmul(xnp.matmul(gamma_in_sys, deriv_d), xnp.matmul(mat_d_inv, diff)))
    else:
        raise ValueError(f"Unknown method {method} for computing the gradient over norm.")
    return dest


# =========================== Cache Server =================================


class CacheServer:
    """Storage Server for arbitrary data that can be stored in dictionaries"""

    def __init__(self):
        self.store = {}

    def add(self, name, mat):
        self.store[name] = mat

    def get(self, name):
        try:
            return self.store[name]
        except KeyError:
            return None

    def load(self, fname):
        if os.path.isfile(fname):
            with gzip.open(fname, "rb") as infile:
                self.store = pickle.load(infile)

    def save(self, fname):
        # We only save if the file does not exist yet
        if not os.path.isfile(fname):
            with gzip.open(fname, "wb") as outfile:
                pickle.dump(self.store, outfile)

    def list(self):
        print(self.store.keys)

    def __str__(self):
        print(f"CacheServer: {len(self.store)} Entries")


# =========================== WoodburyInverter ===============================


class WoodburyInverter:
    def __init__(self, mat):
        self.ainv = xnp.linalg.inv(mat)

    def inv(self):
        return self.ainv

    def update(self, u, c, v):
        # We ware updating the matrix A according to A=A+UCV and recalculate the inverse afterwards
        if not xnp.allclose(c, 0):
            # We cannot update with C being zero since this matrix has no inverse
            cinv = xnp.linalg.inv(c)
            self.ainv -= ((self.ainv @ u) @ xnp.linalg.inv(cinv + v @ self.ainv @ u)) @ (v @ self.ainv)
        return self.ainv

    def update_index(self, m, indi, indj):
        # Construct two matrices to shift M to the correct position in A
        if not xnp.allclose(m, 0):
            # We cannot update with C being zero since this matrix has no inverse
            m_m, n_m = m.shape
            m_a, n_a = self.ainv.shape
            idmat = xnp.eye(m_m, n_m)
            u = xnp.zeros((m_a, m_m))
            v = xnp.zeros((n_m, n_a))
            if ggpeps.PREFERRED_BACKEND == "jax":  # TODO: handle based on type checking instead
                u = u.at[indi : indi + m_m, 0:n_m].set(idmat)
                v = v.at[0:m_m, indj : indj + n_m].set(idmat)
            else:
                u[indi : indi + m_m, 0:n_m] = idmat  # TODO: fix for JAX - DONE
                v[0:m_m, indj : indj + n_m] = idmat
            return self.update(u, m, v)
        else:
            return self.inv()


# =========================== IncDeterminant ===============================
class IncDeterminant:
    def __init__(self, a):
        self.detval = xnp.linalg.det(a)

    def update(self, ainv, u, c, v, store=True):
        # We ware updating the matrix A according to A=A+UCV and recalculate the inverse afterwards
        dest = self.detval
        if not xnp.allclose(c, 0):
            cinv = xnp.linalg.inv(c)
            dest = self.detval * xnp.linalg.det(cinv + v @ ainv @ u) * xnp.linalg.det(c)
            if store:
                self.detval = dest
        return dest

    def det(self):
        return self.detval


def update_index(self, ainv, m, indi, indj, store=True):
    # Construct two matrices to shift M to the correct position in A
    if not xnp.allclose(m, 0):
        m_m, n_m = m.shape
        m_a, n_a = ainv.shape
        idmat = xnp.eye(m_m, n_m)
        u = xnp.zeros(m_a, m_m)
        v = xnp.zeros(n_m, n_a)
        if ggpeps.PREFERRED_BACKEND == "jax":  # TODO: handle based on type checking instead
            u = u.at[indi : indi + m_m, 0:n_m].set(idmat)
            v = v.at[0:m_m, indj : indj + n_m].set(idmat)
        else:
            u[indi : indi + m_m, 0:n_m] = idmat  # TODO: fix for JAX - DONE
            v[0:m_m, indj : indj + n_m] = idmat
        return self.update(ainv, u, m, v, store)
    else:
        return self.detval


# =========================== IncLogAbsDeterminant ===============================


class IncLogAbsDeterminant:
    def __init__(self, a):
        # We are not using the sign right now.
        # We know that the sign has to be positive
        self.sign, self.detval = xnp.linalg.slogdet(a)

    def det(self):
        return self.detval

    def update(self, ainv, u, c, v, store=True):
        # We are updating the matrix A according to A=A+UCV and recalculate the inverse afterwards
        dest = self.detval
        converged = True
        if not xnp.allclose(c, 0):
            # We cannot update if c is zero because we cannot invert it
            # There might also be problems if c is singular !
            sign, cdetval = xnp.linalg.slogdet(c)
            sign, combined_detval = xnp.linalg.slogdet(xnp.linalg.inv(c) + v @ ainv @ u)
            if xnp.isnan(combined_detval) or xnp.isnan(cdetval):
                converged = False
            if converged:
                dest = self.detval + cdetval + combined_detval
            if store:
                self.detval = dest
        return dest

    def update_index(self, ainv, m, indi, indj, store=True):
        # Construct two matrices to shift M to the correct position in A
        if not xnp.allclose(m, 0):
            # We cannot update if m is zero because we cannot invert it
            m_m, n_m = m.shape
            m_a, n_a = ainv.shape
            idmat = xnp.eye(m_m, n_m)
            u = xnp.zeros((m_a, m_m))
            v = xnp.zeros((n_m, n_a))
            if ggpeps.PREFERRED_BACKEND == "jax":  # TODO: handle based on type checking instead
                u = u.at[indi : indi + m_m, 0:n_m].set(idmat)
                v = v.at[0:m_m, indj : indj + n_m].set(idmat)
            else:
                u[indi : indi + m_m, 0:n_m] = idmat  # TODO: fix for JAX - DONE
                v[0:m_m, indj : indj + n_m] = idmat
            return self.update(ainv, u, m, v, store)
        else:
            return self.det()


# Not used (though still appears in tests)
class BgbTransform:
    def __init__(self, mat_in, pure_gauge=True):
        self.mat_in = mat_in
        self.is_pure_gauge = pure_gauge
        self._mat_out = None

    @property
    def mat_out(self):
        if self._mat_out is None:
            wn, s, wp = svd(self.mat_in, full_matrices=True, compute_uv=True)  # self.mat_in is the T matrix
            wp = herm_conj(wp)
            if not self.is_pure_gauge:
                # TODO: Fix this
                # We are shuffling the physical mode to the front again
                # It would look like s=perm*s
                # TODO: This does not work properly yet. But the function is not used anywhere.
                perm = np.zeros((wn.shape[0], wn.shape[0]))
                i, j = np.indices(perm.shape)
                perm[i == j + 1] = 1
                perm[0, -1:] = 1
                # Apply the permutation
                wn = wn * perm.transpose()
            un = herm_conj(wn)
            # now we got the transpose of wp
            up = np.transpose(wp)
            un_rows, un_cols = un.shape
            up_rows, up_cols = up.shape
            unitary_transform = np.zeros((un.shape[0] + up.shape[0], un.shape[1] + up.shape[1]), dtype=complex)
            unitary_transform[:un_rows, :un_cols] = un
            unitary_transform[-up_rows:, -up_cols:] = up

            trafo_size = len(s) * 2 if self.is_pure_gauge else len(s) * 2 + 1
            start_ind = 0 if self.is_pure_gauge else 1
            r0_diagonal = np.zeros(trafo_size, dtype=complex)
            if not self.is_pure_gauge:
                r0_diagonal[0] = 1j / 2.0
            r0_diagonal[start_ind : start_ind + len(s)] = 1j / 2.0 * (1 - s**2) / (1 + s**2)
            r0_diagonal[-len(s) :] = 1j / 2.0 * (1 - s**2) / (1 + s**2)
            r0 = np.diag(r0_diagonal)

            q0_offdiagonal = np.zeros(len(s), dtype=complex)
            q0_offdiagonal = 1j * s / (1 + s**2)
            q0_block = np.diag(q0_offdiagonal)
            q0 = np.zeros((trafo_size, trafo_size), dtype=complex)
            if not self.is_pure_gauge:
                q0[0, 0] = 0
            q0[
                start_ind : start_ind + len(s),
                start_ind + len(s) : start_ind + 2 * len(s),
            ] = -q0_block
            q0[
                start_ind + len(s) : start_ind + 2 * len(s),
                start_ind : start_ind + len(s),
            ] = q0_block

            gamma0 = np.zeros((2 * trafo_size, 2 * trafo_size), dtype=complex)
            gamma0 = np.block([[q0, r0], [np.conj(r0), np.conj(q0)]])
            trafo_0 = block_diag(herm_conj(unitary_transform), np.transpose(unitary_transform))
            trafo_1 = block_diag(np.conj(unitary_transform), unitary_transform)
            # This matrix has the following order: psi, r+, u-, l-, d+,t,b, r-, l+,
            # u+, d-,t,b psi_dag, r+_dag, l-_dag, u-_dag, d+_dag,t_dag,b_dag,
            # r-_dag, l+_dag, u+_dag, d-_dag, t_dag, b_dag.
            self._mat_out = trafo_0 @ gamma0 @ trafo_1
        return self._mat_out


# ========= Rebinning Functions ====================


def autocorr_fft(arr):
    arr = arr - np.mean(arr)
    fft_vals = np.fft.fft(arr)
    spectrum = fft_vals * np.conjugate(fft_vals)
    dest = np.fft.ifft(spectrum)
    return dest / dest[0]


def rebin_array(a, R):
    """Rebin an array into bins of length R"""
    if isinstance(a, list):
        a = np.asarray(a)
    R = int(R)
    max_fit = int(len(a) - len(a) % R)
    if a.ndim == 1:
        # Shape (N): N samples of scalars
        dest = np.mean(a[:max_fit].reshape(-1, R), axis=1)
    elif a.ndim == 2:
        # Shape (N,m): N samples of m-dim vecotrs
        N, m = a.shape
        dest = np.mean(a[:max_fit].reshape(-1, m, R), axis=2)
    elif a.ndim == 3:
        # Shape (N,m,n): N samples of m x n matrices
        N, m, n = a.shape
        dest = np.mean(a[:max_fit].reshape(-1, m, n, R), axis=3)
    elif a.ndim == 4:
        # Shape (N,p,m,n): N samples of p x m x n tensors
        N, p, m, n = a.shape
        dest = np.mean(a[:max_fit].reshape(-1, p, m, n, R), axis=4)
    else:
        logger.error("rebin_array not implemented for dimensions greater than 4.")
        return a
    return dest


def rebin_error(arr):
    """Rebin the given error to avoid autocorrelation in the error estimation

    Args:
        arr (np.ndarray): Timeseries of a measurement

    Returns:
        tuple: (value of binning, mean estimations, error on mean estimations, std dev estimations)
    """
    N = len(arr)
    max_exp = int(np.floor(np.log2(N / 10)))
    rangevals = [2**i for i in range(max_exp + 1)]
    eomarr = []
    stdarr = []
    meanarr = []
    for i in rangevals:
        data_rebin = rebin_array(arr, i)
        eom = np.std(data_rebin, ddof=1) / np.sqrt(len(data_rebin))
        std = np.std(data_rebin, ddof=1)
        eomarr.append(eom)
        meanarr.append(np.mean(data_rebin))
        stdarr.append(std)
    return rangevals, meanarr, eomarr, stdarr


def rebin_eom(arr, num_of_bins=20):
    """Calculate the error on the mean (EOM) by rebinning.
    As a heuristic for the EOM we use that the biggest bin will give the best estimate.
    We do not rebin to the maximal extent, but use the heuristic of taking the largest
    binsize of the form 2^i that can fit N/20.

    Args:
        arr (np.ndarray): Timeseries of a measurement

    Returns:
        float or arr: Best estimate of the EOM on the given array. The output shape depends on the input shape of arr.
    """
    N = len(arr)
    # We want to leave a sufficient number of samples to build a reasonable mean
    max_exp = int(np.floor(np.log2(N / (num_of_bins / 2))))
    if max_exp > 0:
        binsize = 2 ** (max_exp - 1)
        data_rebin = rebin_array(arr, binsize)
    else:
        # We cannot rebin if we have too few data. We will just return the normal EOM
        data_rebin = arr
    eom = np.std(data_rebin, ddof=1, axis=0) / np.sqrt(len(data_rebin))
    return eom


def autocorr_rebin_eom(arr):
    """Calculate the autocorrelation, find the corrrelation decay time
    (when the auto-correlation decays below 1/100),
    and calculate the error using bins with the correlation time size

    Args:
        arr (np.ndarray): Timeseries of a measurement

    Returns:
        tuple of
            eom: float with the EOM estimation
            decay_time: float with the decay time (in terms of step number) of the autocorrelation
    """
    N = len(arr)
    autocorr_array = autocorr_fft(arr)
    for i in range(len(autocorr_array)):  # find first two elements below 1/100
        if i >= N / 10:  # limit the number of bins to a minimum of 10.
            eom = rebin_eom(arr, 10)
            decay_time = i
            return eom, decay_time
        elif autocorr_array[i] <= 1 / 100 and autocorr_array[i + 1] <= 1 / 100:
            num_of_bins = N // i
            eom = rebin_eom(arr, num_of_bins)
            decay_time = i
            return eom, decay_time


def autocorr_rebin_data(arr):
    """
    Rebin the data to remove autocorrelation.
    The binsize is determined by the first two elements of the autocorrelation function that are below 1/100.

    Args:
        arr (np.ndarray): Timeseries of a measurement
    Returns:
        np.ndarray: Rebinend data
    """
    N = len(arr)
    autocorr_array = autocorr_fft(arr)
    for i in range(len(autocorr_array)):  # find first two elements below 1/100
        if i >= N / 10:  # limit the number of binsto a minimum of 10.
            binsize = i
            break
        elif autocorr_array[i] <= 1 / 100 and autocorr_array[i + 1] <= 1 / 100:
            binsize = i
            break
    rebinned_array = rebin_array(arr, binsize)
    return rebinned_array, binsize


def jackknife_resampling(data):
    """Generate jackknife resamples of the data."""
    n = len(data)
    indices = np.arange(n)
    resamples = np.zeros(n)
    for i in range(n):
        resamples[i] = np.mean(data[indices != i])
    return resamples


def jacknife_gradient_error_propagation(op_datavec, op_grad_datavec, grad_norm_datavec):
    """Calculate the error propagation of the gradient of an observable using jackknife resampling.

    Args:
        op_datavec (np.ndarray): Timeseries of the observable - rebinned data, i.e., not autocorrelation
        op_grad_datavec (np.ndarray): Timeseries of the gradient of the observable - rebinned data, i.e., not autocorrelation
        grad_norm_datavec (np.ndarray): Timeseries of the gradient of the norm of the ansatz divided by the norm of the ansatz
        - rebinned data, i.e., not autocorrelation

    Returns:
        float: Error of the gradient of the observable
    """
    op_datavec_resamples = jackknife_resampling(op_datavec)
    op_grad_datavec_resamples = jackknife_resampling(op_grad_datavec)
    grad_norm_datavec_resamples = jackknife_resampling(grad_norm_datavec)
    op_times_grad_norm_resamples = jackknife_resampling(op_datavec * grad_norm_datavec)
    mean_grad = np.mean(
        op_grad_datavec_resamples + op_times_grad_norm_resamples - op_datavec_resamples * grad_norm_datavec_resamples
    )
    grad_jacknife = (
        op_grad_datavec_resamples + op_times_grad_norm_resamples - op_datavec_resamples * grad_norm_datavec_resamples
    )
    n = len(grad_jacknife)

    return np.sqrt((n - 1) * np.mean((grad_jacknife - mean_grad) ** 2))


def compute_grad_err(op_datavec, op_grad_datavec, grad_norm_datavec):
    """Compute the error of the gradient of an observable.

    Args:
        op_datavec(np.ndarray): Timeseries of the observable
        op_grad_datavec(np.ndarray): Timeseries of the gradient of the observable
        grad_norm_datavec(np.ndarray): Timeseries of the gradient of the norm of the ansatz divided by the norm of the ansatz
    Returns:
        float: Error of the gradient of the observable
    """
    op_datavec_rebinned, op_datavec_rebinned_binsize = autocorr_rebin_data(op_datavec)
    op_grad_datavec_rebinned, op_grad_datavec_rebinned_binsize = autocorr_rebin_data(op_grad_datavec)
    grad_norm_datavec_rebinned, grad_norm_datavec_rebinned_binsize = autocorr_rebin_data(grad_norm_datavec)
    max_binsize = max(
        op_datavec_rebinned_binsize,
        op_grad_datavec_rebinned_binsize,
        grad_norm_datavec_rebinned_binsize,
    )

    if (
        max_binsize > op_datavec_rebinned_binsize
    ):  # All arrays should be of the same size, so we pick the largest binsize
        op_datavec_rebinned = rebin_array(op_datavec, max_binsize)
    if max_binsize > op_grad_datavec_rebinned_binsize:
        op_grad_datavec_rebinned = rebin_array(op_grad_datavec, max_binsize)
    if max_binsize > grad_norm_datavec_rebinned_binsize:
        grad_norm_datavec_rebinned = rebin_array(
            grad_norm_datavec,
            max_binsize,
        )
    return jacknife_gradient_error_propagation(
        op_datavec_rebinned, op_grad_datavec_rebinned, grad_norm_datavec_rebinned
    )


def compute_grad_mean(op_datavec, op_grad_datavec, grad_norm_datavec):
    """Compute the mean of the gradient of an observable.

    Args:
        op_datavec(np.ndarray): Timeseries of the observable
        op_grad_datavec(np.ndarray): Timeseries of the gradient of the observable
        grad_norm_datavec(np.ndarray): Timeseries of the gradient of the norm of the ansatz divided by the norm of the ansatz
    Returns:
        float: Mean of the gradient of the observable
    """
    mean = np.mean(op_grad_datavec + op_datavec * grad_norm_datavec)
    mean = mean - np.mean(op_datavec) * np.mean(grad_norm_datavec)
    return mean


# ========== Debugging Functions ====================


def show_vector(vec, title=None):
    """Display a matrix and interrupt the program."""
    f, ax = plt.subplots(1, 1)
    ax.plot(vec)
    if title is not None and len(title) > 0:
        plt.title(title)
    plt.show()


def show_matrix(mat, title=None, **kwargs):
    """Display a matrix and interrupt the program."""
    show_matrixvec([mat], title=[title], **kwargs)


def show_matrixvec(matvec, title=None, log=False):
    """Display a matrix and interrupt the program."""
    f, axvec = plt.subplots(1, len(matvec))
    if len(matvec) == 1:
        axvec = [axvec]
    for ind, mat in enumerate(matvec):
        if log:
            minval = np.min(mat)
            if minval == 0:
                # This is a dirty hack to display the 0 in a log plot
                mat += 1e-10
                minval += 1e-10
            matax = axvec[ind].matshow(mat, norm=LogNorm(vmin=minval, vmax=np.max(mat)))
        else:
            matax = axvec[ind].matshow(mat)
        f.colorbar(matax, ax=axvec[ind])

    if title is not None:
        if type(title) is list and len(title) == len(matvec):
            for ax, titleval in zip(axvec, title):
                ax.set_title(titleval)
        elif type(title) is str and len(title) > 0:
            plt.title(title)
    plt.show()


def print_mat_stats(mat, title=None):
    """Display general information about matrix."""
    print(f"Min:\t{np.min(mat)}")
    print(f"Max:\t{np.max(mat)}")
    print(f"Avg:\t{np.mean(mat)}")
    print(f"Norm:\t{np.linalg.norm(mat)}")


def show_eigenvalues(mat):
    """Display the eigenvalues of a matrix"""
    if is_hermitian(mat):
        # Plot the real eigenvalues
        f, ax = plt.subplots(1, 1)
        eigvals = np.linalg.eigvalsh(mat)
        ax.plot(eigvals, "o")
    else:
        # Plot the real eigenvalues
        f, ax = plt.subplots(1, 2)
        eigvals = np.linalg.eigvals(mat)
        ax[0].set_title("Real part")
        ax[0].plot(np.real(eigvals), "o")
        ax[0].set_title("Imaginary part")
        ax[1].plot(np.imag(eigvals), "o")
    plt.show()


# ========== Workflow & Tooling Functions ====================


def get_couplings_from_foldername(fname: str) -> str:
    couplings = ["g", "el", "mag", "int", "mass"]
    res = ""
    for arg in couplings:
        pattern = rf"(?<={arg}_)[\d]*.[\d]*"
        result = re.search(pattern, fname)
        if result is not None:
            res += f"{arg}_{result.group(0)}_"

    # chem - we treat this differently because it is a vector for different flavors
    pattern = rf"(?<=chem_)(-?\d+\.\d+)_(-?\d+\.\d+)"
    result = re.search(pattern, fname)
    if result is not None:
        vals = f"_".join(result.groups())
        res += f"chem_{vals}_"
    return res


def extract_params_from_results_file(fname: str, dest_dir: Optional[str] = "") -> bool:
    """Extract parameters from a results file and save to a new .npy file

    Args:
        fname (str): results file path
        dest_dir (str, optional): destination directory for param file.
                                  If none is given, defaults to current directory.

    Returns:
        bool: True if succesful, false otherwise.
    """
    if fname is not None and os.path.isfile(fname):
        fname_base = os.path.basename(fname)
        name, ext = os.path.splitext(fname_base)
        if name.startswith("result_min"):
            with open(fname, "rb") as infile:
                data = pickle.load(infile)
                couplings = get_couplings_from_foldername(fname)
                # Deal with renaming
                if hasattr(data, "paramvec"):
                    np.save(
                        os.path.join(dest_dir, f"{couplings}extracted_paramvec.npy"),
                        data.paramvec,
                    )
                elif hasattr(data, "parametervec"):
                    np.save(
                        os.path.join(dest_dir, f"{couplings}extracted_paramvec.npy"),
                        data.parametervec,
                    )
    else:
        print(f"File '{fname}' not found. Aborting.", file=sys.stderr)
        return False

    return True


def extract_params_from_run(source_dir, dest_dir):
    """Extracts all the parameters from the results files of a run (with varying
    couplings), and stores them as .npy files.

    Args:
        source_dir (str): a source directory containing directories, each of which is the result of a run.
        dest_dir (str): a destination directory to story the resulting .npy files.
    """

    for dir in os.listdir(source_dir):
        if os.path.isdir(os.path.join(source_dir, dir)):
            inner_dir = os.path.join(source_dir, dir)
            files = os.listdir(inner_dir)
            for f in files:
                if os.path.isfile(os.path.join(inner_dir, f)):
                    extract_params_from_results_file(os.path.join(inner_dir, f), dest_dir)


# ========== Testing Functions ====================


def compare_array_elementwise(testcase, ref, res, print_vals=True):
    testcase.assertEqual(ref.shape, res.shape)
    if print_vals:
        for i in range(ref.shape[0]):
            for j in range(ref.shape[1]):
                if not np.isclose(ref[i, j], res[i, j]):
                    print(f"{i},{j}: ref: {ref[i,j]}, res:{res[i,j]}")
    testcase.assertTrue(np.allclose(ref, res))
