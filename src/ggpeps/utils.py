import os
import re
import sys
import gzip
import pickle
import logging
import subprocess  # Start process for git hash
from typing import Optional, Union

import numba as nb
import pandas as pd
from scipy.sparse import issparse
from scipy.linalg import svd, block_diag

import numpy as np
import jax.numpy as jnp
from ggpeps import xnp as xnp

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import ggpeps
import ggpeps.measurement as meas
from ggpeps.system.backend import backend
from ggpeps.system.system_base import maybe_jit
from ggpeps.system.backend_jax import derivative_pfaffian_jax
from ggpeps.system.backend_numpy import derivative_pfaffian_numpy

logger = logging.getLogger(ggpeps.LOGGER_NAME)

# Global constants
paulix = np.array([[0, 1], [1, 0]])
pauliy = np.array([[0, -1.0j], [1.0j, 0]])
pauliz = np.array([[1, 0], [0, -1]])

# ========== Utility Functions ====================


def setup_logger(logger: logging.Logger, log_file: str, level: str, runner_msg: str = "") -> None:
    """
    Setup the logger to log to a file and stdout/stderr.
    """
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


def fname2nlayer(fname: str) -> Optional[int]:
    """Extract the number of layers from a filename"""
    pattern = r"(?<=nlayer_)[\d]*"
    result = re.search(pattern, fname)
    if result is not None:
        return int(result.group(0))
    else:
        return None


def fname2ncopy(fname: str) -> Optional[int]:
    """Extract the number of copies from a filename"""
    pattern = r"(?<=ncopy_)[\d]*"
    result = re.search(pattern, fname)
    if result is not None:
        return int(result.group(0))
    else:
        return None


def fname2g(fname: str) -> Optional[float]:
    """Extract the coupling from a filename"""
    pattern = r"(?<=g_)[\d]*\.[\d]*"
    result = re.search(pattern, fname)
    if result is not None:
        return float(result.group(0))
    else:
        return None


def fname2gel(fname: str) -> Optional[float]:
    """Extract the electric coupling from a filename"""
    pattern = r"(?<=gel_)[\d]*\.[\d]*"
    result = re.search(pattern, fname)
    if result is not None:
        return float(result.group(0))
    else:
        return None


def fname2L(fname: str) -> Optional[int]:
    """Extract the system size from a filename"""
    pattern = r"(?<=L_)[\d]*"
    result = re.search(pattern, fname)
    if result is not None:
        return int(result.group(0))
    else:
        return None


def isclose(x: float, y: float, rtol: float = 1.0e-5, atol: float = 1.0e-8) -> bool:
    return abs(x - y) <= atol + rtol * abs(y)


def load_matrix_dat_fmt(path: str, is_complex: bool = True) -> np.ndarray:
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


def merge_measurements(meas1: meas.Measurement, meas2: meas.Measurement) -> meas.Measurement:
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


def mergeDict(dict1: dict, dict2: dict) -> dict:
    """Left Merge dictionaries that contain only lists and append lists if values are common

    Args:
        dict1 (dict): First dictionary
        dict2 (dict): Second dictionary
    Returns:
        dict: Merged dictionary
    """
    dest = {}
    for key in dict1:
        if key in dict2:
            # We assume that there are only lists in the dictionaries
            dest[key] = merge_measurements(dict1[key], dict2[key])
        else:
            dest[key] = dict1[key]
    return dest


def print_columns(listvals: list[list], padding: int = 4, header: bool = False) -> None:
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


def sizeof_fmt(num: float, suffix: str = "B") -> str:
    """Print nicely a size as multiples of 1024."""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%3.1f %s%s" % (num, "Yi", suffix)


def get_git_hash() -> str:
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


def extract_partial_covmats(mat: xnp.ndarray, corner: int) -> tuple[xnp.ndarray, xnp.ndarray, xnp.ndarray]:
    """Extract the partial covariance matrices from a gaussian mapping.
    This function can accept a 2D matrix, or a stack of 2D matrices (i.e. a 3D array).

    Args:
        mat (xnp.ndarray): Full covariance matrix
        corner (int): Index of the top left element of the bottom right matrix

    Returns:
        tuple: Matrices (A,B,D)
    """
    mat_a = mat[..., :corner, :corner]
    mat_b = mat[..., :corner, corner:]
    mat_d = mat[..., corner:, corner:]
    return mat_a, mat_b, mat_d


@maybe_jit(static_argnames=["link_inds", "lattice_size", "nphysmodes_site", "nvirtmodes_link"])
def extract_mod_covmats(
    mat: xnp.ndarray, link_inds: tuple[int, ...], lattice_size: int, nphysmodes_site: int, nvirtmodes_link: int
) -> tuple[xnp.ndarray, xnp.ndarray, xnp.ndarray]:
    """Extract the A, B, D submatrices, but including the virtual modes on the link specified by link_ind.
    This function can accept a 2D matrix, or a stack of 2D matrices (i.e. a 3D array).

    Args:
        mat (xnp.ndarray): The mat(s) from which to extract the submatrices.
        link_inds (tuple[int, ...]): a list of link indices to include in the physical-physical set.
        lattice_size (int): the length of a side of the square lattice.
        nphysmodes_site (int): number of physical modes per site.
        nvirtmodes_link (int): number of virtual modes per link.

    Returns:
        tuple[xnp.ndarray, xnp.ndarray, xnp.ndarray]: the A, B, D submatrices, across layers and links
    """
    single_matrix = mat.ndim == 2
    if single_matrix:
        mat = mat[None, ...]

    _, size, __ = mat.shape

    phys_offset = 2 * lattice_size * nphysmodes_site

    A_list = []
    B_list = []
    D_list = []

    for link_ind in link_inds:
        virt_start = phys_offset + 2 * nvirtmodes_link * link_ind
        virt_end = virt_start + 2 * nvirtmodes_link

        # Get indices of physical and virtual modes
        phys_inds = xnp.concatenate([xnp.arange(phys_offset), xnp.arange(virt_start, virt_end)])
        virt_inds = xnp.concatenate([xnp.arange(phys_offset, virt_start), xnp.arange(virt_end, size)])

        A_list.append(mat[..., phys_inds, :][..., :, phys_inds])  # get rows, then columns
        B_list.append(mat[..., phys_inds, :][..., :, virt_inds])
        D_list.append(mat[..., virt_inds, :][..., :, virt_inds])

    # Stack into (layers, links, ...)
    A = xnp.stack(A_list, axis=1)
    B = xnp.stack(B_list, axis=1)
    D = xnp.stack(D_list, axis=1)

    if single_matrix:
        A = A[0]
        B = B[0]
        D = D[0]

    return A, B, D


def select_except(arr: Union[list, xnp.ndarray], ind: int) -> xnp.ndarray:
    """Return all elements of a list except the indicated one

    Args:
        arr (list/np.array): list of values
        ind (int): index

    Returns:
        xnp.ndarray: Array with all elements of arr except for arr[ind]
    """
    # This function works only on the outer-most layer
    if isinstance(arr, list):
        arr = xnp.asarray(arr)
    mask = xnp.ones(len(arr), dtype=bool)
    mask = backend.array_assign(mask, ind, False)
    return arr[mask]  # TODO: fix for JAX jit


@maybe_jit(static_argnames=[])
def add_except(arr: xnp.ndarray, ind: int) -> float:
    """Sum all array values except for arr[ind]

    Args:
        arr (xnp.ndarray): list of values
        ind (int): index

    Returns:
        float: Sum of all array values except for arr[ind]
    """
    mask = xnp.ones(len(arr), dtype=bool)
    mask = backend.array_assign(mask, ind, False)
    sum_other = xnp.where(mask, arr, 0.0).sum()
    return sum_other


def multiply_except(arr: Union[xnp.ndarray, list], ind: int) -> float:
    """Multiply all array values except for arr[ind]

    Args:
        arr (list/xnp.ndarray): list of values
        ind (int): index

    Returns:
        float: Multiplication of all array values except for arr[ind]
    """
    if len(arr) > 1:
        mask = xnp.ones(len(arr), dtype=bool)
        mask = backend.array_assign(mask, ind, False)
        prod_other = xnp.where(mask, arr, 1.0).prod()
        return prod_other
    else:
        # It does not make sense to execute this function with only one element
        return arr[0]


@nb.njit(cache=True)
def pfaffian_explicit_4x4_masked(
    mat: xnp.ndarray, ind: Union[tuple[int, int, int, int], list[int], xnp.ndarray]
) -> float:
    """
    Calculate the Pfaffian of a 4x4 block of a matrix explicitly using the indices provided
    (the indices from which the block is sliced).

    Args:
        mat (xnp.ndarray): Input matrix
        ind (Union[tuple[int,int,int,int], list[int], xnp.ndarray[int]]): Indices for the 4x4 block
    """
    i0, i1, i2, i3 = ind
    return (mat[i0, i1] * mat[i2, i3]) - (mat[i0, i2] * mat[i1, i3]) + (mat[i1, i2] * mat[i0, i3])


@nb.njit(cache=True)
def pfaffian_explicit_4x4(mat: xnp.ndarray) -> float:
    """Calculate the Pfaffian of a 4x4 matrix explicitly.
    Args:
        mat (np.ndarray): 4x4 matrix
    Returns:
        float: Pfaffian value
    """
    return (mat[0, 1] * mat[2, 3]) - (mat[0, 2] * mat[1, 3]) + (mat[1, 2] * mat[0, 3])


# @nb.njit(cache=True)
def derivative_pfaffian_covariance_mat(pfarr, matvec, d_matvec):
    dest = 0.0
    for pfaval, mat, d_mat in zip(pfarr, matvec, d_matvec):
        if not isclose(pfaval, 0):
            mat_inv = xnp.linalg.inv(mat)
            dest += 0.5 * pfaval * xnp.trace(mat_inv @ d_mat)
    return dest


def derivative_pfaffian(mat: xnp.ndarray, d_mat: xnp.ndarray, pfaval=None) -> float:
    """Compute the derivative of a Pfaffian of a matrix A.
        The explicit derivative dA/dx is given as a second argument

        The given formula is only valid if A is not singular.

        Args:
            mat (xnp.ndarray): Input Matrix A
            d_mat (xnp.ndarray): Derivative dA/dx

        Returns:
    <<<<<<< HEAD
            xnp.ndarray: d(Pf(A))/dx
    =======
            float: d(Pf(A))/dx
    >>>>>>> dev
    """
    # We assume the types of all the provided arguments match
    if isinstance(mat, jnp.ndarray):
        return derivative_pfaffian_jax(mat, d_mat, pfaval=pfaval)
    else:
        return derivative_pfaffian_numpy(mat, d_mat, pfaval=pfaval)


def get_obs_mean_df(df: pd.DataFrame, obs: str, column: str = "mean"):
    """Get the <column> (mean, err, paramvec, etc) of an observable from the summary dataframe.

    Args:
        df (pd.DataFrame): Summary dataframe.
        obs (str): Name of the observable.
        column (str, optional): Column to extract. Defaults to "mean".

    Returns:
        float or xnp.ndarray: Mean value of the observable.
    """
    return df.loc[df["name"] == obs, column].values[0]


def save_summary_df(df: pd.DataFrame, fname_summary: str) -> None:
    """Save the evaluation summary to a given filename

    Args:
        df (pd.DataFrame): Dataframe containing the summary
        fname_summary (str): Output filename for the summary
    """
    df.to_pickle(fname_summary)


def deepcopy_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deep copy a summary DataFrame, including numpy arrays in paramvec, mean, err, etc.

    Args:
        df: DataFrame of the format returned by Evaluator.summary()

    Returns:
        A fully independent copy of the DataFrame.
    """
    df_copy = df.copy(deep=True)  # does not make deep copies of np.ndarrays in the dataframe

    for col in df.columns:
        for idx, val in df_copy[col].items():
            if isinstance(val, np.ndarray):
                # We intentionally store np.ndarray objects inside DataFrame cells (object dtype) in the summary.
                # pandas supports this at runtime, but pandas-stubs types `.at[...]` as scalar-only, so mypy flags
                # assigning an ndarray to a cell.
                # It is safe to ignore here because (1) we guard with `isinstance(val, np.ndarray)`, and
                # (2) the assignment preserves the existing runtime behavior while preventing shared ndarray
                # references between `df` and `df_copy`.
                df_copy.at[idx, col] = np.copy(val)  # type: ignore[assignment]

    return df_copy


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


def is_permutation(mat: xnp.ndarray) -> bool:
    """Returns true if the matrix is a permutation matrix."""
    n, m = mat.shape
    if issparse(mat):
        raise NotImplementedError("Checking for sparse permutation matrices is not implemented.")
    else:
        square = n == m
        id = xnp.allclose(xnp.eye(n), mat @ xnp.transpose(mat))
        sum_rows = xnp.all(xnp.sum(mat, axis=0) == 1)
        sum_cols = xnp.all(xnp.sum(mat, axis=1) == 1)
        return bool(square and id and sum_rows and sum_cols)


def is_antisymmetric(mat, rtol=1e-5, atol=1e-8):
    """Returns true if the matrix mat is anti-symmetric."""
    if issparse(mat):
        return xnp.allclose(mat.todense(), -mat.T.todense(), rtol=rtol, atol=atol)
    else:
        return xnp.allclose(-xnp.transpose(mat), mat, rtol=rtol, atol=atol)


def is_covmat(mat: xnp.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
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


def anti_symmetrize(mat: xnp.ndarray) -> xnp.ndarray:
    """Force a matrix to be anti-symmetirc."""
    return 0.5 * (mat - mat.T)


def get_nonzero_fraction(mat: xnp.ndarray):
    """Returns fraction of non-zero elements."""
    return xnp.count_nonzero(mat) / xnp.prod(mat.shape)


def herm_conj(mat: xnp.ndarray) -> xnp.ndarray:
    """Returns the hermitian conjugate of a matrix."""
    return xnp.conjugate(xnp.transpose(mat))


def commutator(mat1: xnp.ndarray, mat2: xnp.ndarray) -> xnp.ndarray:
    """Calculate the commutator of two matrices

    Args:
        mat1 (2d np.ndarray): First argument of commutator
        mat2 (2d np.ndarray): Second argument of commutator

    Returns:
        2d np.ndarray: Commutator
    """
    return (mat1 @ mat2) - (mat2 @ mat1)


def anticommutator(mat1: xnp.ndarray, mat2: xnp.ndarray) -> xnp.ndarray:
    """Calculate the anti-commutator of two matrices

    Args:
        mat1 (2d np.ndarray): First argument of anti-commutator
        mat2 (2d np.ndarray): Second argument of anti-commutator

    Returns:
        2d np.ndarray: Anti-commutator
    """
    return (mat1 @ mat2) + (mat2 @ mat1)


# =========== Covariance Utility Funcitons ===========


def tmat_to_covariance_matrix(tmat: xnp.ndarray) -> xnp.ndarray:
    r"""Transform a T matrix into the corresponding covariance matrix in terms of Dirac modes.
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


def generate_smat(n: int) -> xnp.ndarray:
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
    gamma_in_sys: xnp.ndarray,
    diff: xnp.ndarray,
    deriv_d: xnp.ndarray,
    mat_d_inv: xnp.ndarray,
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

    def __init__(self) -> None:
        self.store: dict = {}

    def add(self, name: str, mat) -> None:
        self.store[name] = mat

    def get(self, name: str) -> Optional[xnp.ndarray]:
        """Get data from the cache server by name."""
        try:
            return self.store[name]
        except KeyError:
            return None

    def load(self, fname: str) -> None:
        """Load data from pkl file into the cache server."""
        if os.path.isfile(fname):
            with gzip.open(fname, "rb") as infile:
                self.store = pickle.load(infile)

    def save(self, fname: str) -> None:
        """Save the cache server to a pkl file."""
        # We only save if the file does not exist yet
        if not os.path.isfile(fname):
            with gzip.open(fname, "wb") as outfile:
                pickle.dump(self.store, outfile)

    def list(self) -> None:
        """Print the keys of the cache server."""
        print(self.store.keys)

    def __str__(self) -> str:
        """Print the number of entries in the cache server."""
        return f"CacheServer: {len(self.store)} Entries"


# =========================== WoodburyInverter ===============================


class WoodburyInverter:
    def __init__(self, mat: xnp.ndarray):
        self.ainv = xnp.linalg.inv(mat)

    def inv(self) -> xnp.ndarray:
        return self.ainv

    def update(self, u: xnp.ndarray, c: xnp.ndarray, v: xnp.ndarray) -> xnp.ndarray:
        """Update the inverse of a matrix A using the Woodbury formula.
        The formula is: (A+UCV)^{-1}=A^{-1} - A^{-1}U(C^{-1}+VA^{-1}U)^{-1}VA^{-1}.
        Args:
            u (xnp.ndarray): U matrix - Contains zeroes and identity blocks, along with V this matrix is
                                    used to place the update C to match the dimensions of M.
            v (xnp.ndarray): V matrix - Contains zeroes and identity blocks, along with U this matrix is
                                    used to place the update C to match the dimensions of M.
            c (xnp.ndarray): Local update matrix C
        Returns:
            xnp.ndarray: Updated inverse matrix (A+UCV)^{-1}

        """
        # We ware updating the matrix A according to A=A+UCV and recalculate the inverse afterwards
        if not xnp.allclose(c, 0):
            # We cannot update with C being zero since this matrix has no inverse
            cinv = xnp.linalg.inv(c)
            self.ainv -= ((self.ainv @ u) @ xnp.linalg.inv(cinv + v @ self.ainv @ u)) @ (v @ self.ainv)
        return self.ainv

    def update_index(self, m: xnp.ndarray, indi: int, indj: int) -> xnp.ndarray:
        """
        Update the inverse of the matrix A using the Woodbury formula, given indices indicating the positions in A
        where the update M is placed. This is done by generating the U and V matrix for the update method.

        Args:
            m (xnp.ndarray): M matrix - The local update matrix to A.
            indi (int): Index in the first dimension of A where the update m is placed.
            indj (int): Index in the second dimension of A where the update m is placed.
        Returns:
            xnp.ndarray: Updated inverse matrix (A+UMV)^{-1}
        """
        # Construct two matrices to shift m to the correct position in A
        if not xnp.allclose(m, 0):
            # We cannot update with m being zero since this matrix has no inverse
            m_m, n_m = m.shape
            m_a, n_a = self.ainv.shape
            idmat = xnp.eye(m_m, n_m)
            u = xnp.zeros((m_a, m_m))
            v = xnp.zeros((n_m, n_a))

            inds_u = (slice(indi, indi + m_m), slice(0, n_m))
            u = backend.array_assign(u, inds_u, idmat)

            inds_v = (slice(0, m_m), slice(indj, indj + n_m))
            v = backend.array_assign(v, inds_v, idmat)
            return self.update(u, m, v)
        else:
            return self.inv()


# =========================== IncDeterminant ===============================
class IncDeterminant:
    def __init__(self, a: xnp.ndarray) -> None:
        self.detval = xnp.linalg.det(a)

    def update(self, ainv: xnp.ndarray, u: xnp.ndarray, c: xnp.ndarray, v: xnp.ndarray, store: bool = True) -> float:
        """Update the determinant of a matrix A using the matrix determinant lemma.
        The formula is: det(A+UCV)=det(A) * det(C^{-1}+VA^{-1}U) * det(C).
        Args:
            ainv (xnp.ndarray): Inverse of the matrix A
            u (xnp.ndarray): U matrix - Contains zeroes and identity blocks, along with V this matrix is
                                    used to place the update C to match the dimensions of A.
            c (xnp.ndarray): Local update matrix C
            v (xnp.ndarray): V matrix - Contains zeroes and identity blocks, along with U this matrix is
                                    used to place the update C to match the dimensions of A.
            store (bool, optional): Store the updated determinant value. Defaults to True.
        """
        # We ware updating the matrix A according to A=A+UCV and recalculate the inverse afterwards
        dest = self.detval
        if not xnp.allclose(c, 0):
            cinv = xnp.linalg.inv(c)
            dest = self.detval * xnp.linalg.det(cinv + v @ ainv @ u) * xnp.linalg.det(c)
            if store:
                self.detval = dest
        return dest

    def det(self) -> float:
        return self.detval


# =========================== IncLogAbsDeterminant ===============================


class IncLogAbsDeterminant:
    def __init__(self, a: xnp.ndarray) -> None:
        # We are not using the sign right now.
        # We know that the sign has to be positive
        self.sign, self.detval = xnp.linalg.slogdet(a)

    def det(self) -> float:
        return self.detval

    def update(self, ainv: xnp.ndarray, u: xnp.ndarray, c: xnp.ndarray, v: xnp.ndarray, store: bool = True) -> float:
        """Update the log of the determinant of a matrix A using the matrix determinant lemma.
        The formula is: det(A+UCV)=det(A) * det(C^{-1}+VA^{-1}U) * det(C).
        Args:
            ainv (xnp.ndarray): Inverse of the matrix A
            u (xnp.ndarray): U matrix - Contains zeroes and identity blocks, along with V this matrix is
                                    used to place the update C to match the dimensions of A.
            c (xnp.ndarray): Local update matrix C
            v (xnp.ndarray): V matrix - Contains zeroes and identity blocks, along with U this matrix is
                                    used to place the update C to match the dimensions of A.
            store (bool, optional): Store the updated determinant value. Defaults to True.
        """
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

    def update_index(self, ainv: xnp.ndarray, m: xnp.ndarray, indi: int, indj: int, store: bool = True) -> float:
        """Update the log of the determinant of a matrix A using the matrix determinant lemma,
        given indices indicating the positions in A where the update M is placed.
        This is done by generating the U and V matrix for the update method.
        Args:
            ainv (xnp.ndarray): Inverse of the matrix A
            m (xnp.ndarray): M matrix - The local update matrix to A.
            indi (int): Index in the first dimension of A where the update m is placed.
            indj (int): Index in the second dimension of A where the update m is placed
            store (bool, optional): Store the updated determinant value. Defaults to True."""

        # Construct two matrices to shift M to the correct position in A
        if not xnp.allclose(m, 0):
            # We cannot update if m is zero because we cannot invert it
            m_m, n_m = m.shape
            m_a, n_a = ainv.shape
            idmat = xnp.eye(m_m, n_m)
            u = xnp.zeros((m_a, m_m))
            v = xnp.zeros((n_m, n_a))

            inds_u = (slice(indi, indi + m_m), slice(0, n_m))
            u = backend.array_assign(u, inds_u, idmat)

            inds_v = (slice(0, m_m), slice(indj, indj + n_m))
            v = backend.array_assign(v, inds_v, idmat)
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


def autocorr_fft(arr: np.ndarray) -> np.ndarray:
    """Calculate autocorrelation of a timeseries using FFT (which is much faster than doing it naively).
    Args:
        arr (np.ndarray): Timeseries of a measurement
    Returns:
        np.ndarray: Autocorrelation of the timeseries"""
    arr = arr - np.mean(arr)
    if np.allclose(arr, 0.0):
        # If the timeseries is constant, the autocorrelation is maximal - all ones.
        return np.ones(arr.shape)
    fft_vals = np.fft.fft(arr)
    spectrum = fft_vals * np.conjugate(fft_vals)
    dest = np.fft.ifft(spectrum)
    return dest / dest[0]


def rebin_array(a: Union[list, np.ndarray], R: Union[int, float]) -> np.ndarray:
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


def rebin_error(arr: Union[np.ndarray, list]) -> tuple[list, list, list, list]:
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


def rebin_eom(arr: Union[np.ndarray, list], num_of_bins=20) -> Union[float, np.ndarray]:
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
        data_rebin = np.asarray(arr)
    eom = np.std(data_rebin, ddof=1, axis=0) / np.sqrt(len(data_rebin))
    return eom


def autocorr_rebin_eom(arr: Union[np.ndarray, list]):
    """Calculate the autocorrelation, find the corrrelation decay time
    (when the auto-correlation decays below 1/100),
    and calculate the error using bins of the decay time size

    Args:
        arr (np.ndarray): Timeseries of a measurement

    Returns:
        tuple of
            eom: float with the EOM estimation
            decay_time: float with the decay time (in terms of step number) of the autocorrelation
    """
    N = len(arr)
    autocorr_array = autocorr_fft(np.asarray(arr))
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
    return


def autocorr_rebin_data(arr: np.ndarray) -> tuple[np.ndarray, int]:
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
        if len(autocorr_array) < 10:
            binsize = 1
            break
        elif i >= N / 10:  # limit the number of bins to a minimum of 10.
            binsize = i
            break
        elif autocorr_array[i] <= 1 / 100 and autocorr_array[i + 1] <= 1 / 100:
            binsize = i
            break
    rebinned_array = rebin_array(arr, binsize)
    return rebinned_array, binsize


def jackknife_resampling(data: np.ndarray) -> np.ndarray:
    """Generate jackknife resamples of the data."""
    n = len(data)
    indices = np.arange(n)
    resamples = np.zeros(n)
    for i in range(n):
        resamples[i] = np.mean(data[indices != i])
    return resamples


def jacknife_gradient_error_propagation(
    op_datavec: np.ndarray, op_grad_datavec: np.ndarray, grad_norm_datavec: np.ndarray
) -> float:
    """Calculate the error propagation of a specific component of the gradient of an observable
    using jackknife resampling.
    Without rebinning (we usually use this after rebinning the data).

    Args:
        op_datavec (np.ndarray): Timeseries of the observable - rebinned (not autocorrelation)
        op_grad_datavec (np.ndarray): Timeseries of the gradient of the observable - rebinned (not autocorrelation)
        grad_norm_datavec (np.ndarray): Timeseries of the gradient of the norm of the ansatz divided by the
                                        norm of the state - rebinned (not autocorrelation)

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


def compute_grad_err(op_datavec: np.ndarray, op_grad_datavec: np.ndarray, grad_norm_datavec: np.ndarray) -> float:
    """Compute the error of a specific component of the gradient of an observable.
       Here we rebin the data to avoid autocorrelation.

    Args:
        op_datavec(np.ndarray): Timeseries of the observable
        op_grad_datavec(np.ndarray): Timeseries of the gradient of the observable
        grad_norm_datavec(np.ndarray): Timeseries of the gradient of the norm of the ansatz divided by the norm

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


def compute_grad_mean(op_datavec: np.ndarray, op_grad_datavec: np.ndarray, grad_norm_datavec: np.ndarray) -> float:
    """Compute the mean of a gradient component of an observable.

    Args:
        op_datavec(np.ndarray): Timeseries of the observable
        op_grad_datavec(np.ndarray): Timeseries of the gradient of the observable
        grad_norm_datavec(np.ndarray): Timeseries of the gradient of the norm of the ansatz divided by the norm
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
def folder2arg(foldername: str, arg: str) -> list[str]:
    """Extract from a string formatted in the form arg1_val_val_arg2_val_val... the values
    corresponding to 'arg', as a list."""

    if arg == "gf" or arg == "gauge_fixing":
        # handle gauge_fixing separately, since it does not take a numeric value
        pattern = rf"(?<={arg}_)([^_]+)"
    else:
        # allow for any number of numeric values (ints or floats) separated by underscores
        pattern = rf"(?<={arg}_)(-?\d+(?:\.\d+)?(?:_-?\d+(?:\.\d+)?)*)"

    result = re.search(pattern, foldername)

    if result is not None:
        vals = result.group(1).split("_")
    else:
        vals = []
    return vals


def get_couplings_from_foldername(fname: str, couplings: Optional[list[str]] = None) -> str:
    """Extract the couplings from a folder name and format."""
    if couplings is None:
        couplings = ["g", "int", "mass", "chem"]

    res = ""
    for arg in couplings:
        vals = folder2arg(fname, arg)
        if len(vals) >= 1:
            res += f"{arg}_{'_'.join(vals)}_"

    return res.strip("_")


def extract_params_from_results_file(fname: str, dest_dir: str = "") -> bool:
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
                        os.path.join(dest_dir, f"{couplings}_extracted_paramvec.npy"),
                        data.paramvec,
                    )
                elif hasattr(data, "parametervec"):
                    np.save(
                        os.path.join(dest_dir, f"{couplings}_extracted_paramvec.npy"),
                        data.parametervec,
                    )
    else:
        print(f"File '{fname}' not found. Aborting.", file=sys.stderr)
        return False

    return True


def extract_params_from_run(source_dir: str, dest_dir: str) -> None:
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


def compare_array_elementwise(testcase, ref: np.ndarray, res: np.ndarray, print_vals: bool = True) -> None:
    testcase.assertEqual(ref.shape, res.shape)
    if print_vals:
        for i in range(ref.shape[0]):
            for j in range(ref.shape[1]):
                if not np.isclose(ref[i, j], res[i, j]):
                    print(f"{i},{j}: ref: {ref[i, j]}, res:{res[i, j]}")
    testcase.assertTrue(np.allclose(ref, res))
