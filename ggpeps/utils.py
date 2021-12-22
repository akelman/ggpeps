import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import select
from scipy.sparse import issparse
from scipy.linalg import svd, block_diag
import scipy.sparse as sparse
import os
import sys
import ggpeps.measurement as meas
import gzip
import pickle
import subprocess  # Start process for git hash
import re

from matplotlib.colors import LogNorm

paulix = np.array([[0, 1], [1, 0]])
pauliy = np.array([[0, -1.j], [1.j, 0]])
pauliz = np.array([[1, 0], [0, -1]])

# ========== Utility Functions ====================

def fname2nlayer(fname):
    """Extract the number of layers from a filename"""
    pattern=r"(?<=nlayer_)[\d]*"
    result = re.search(pattern, fname)
    if result is not None:
        return int(result.group(0))
    else:
        return None

def fname2ncopy(fname):
    """Extract the number of layers from a filename"""
    pattern=r"(?<=ncopy_)[\d]*"
    result = re.search(pattern, fname)
    if result is not None:
        return int(result.group(0))
    else:
        return None

def fname2g2(fname):
    """Extract the number of layers from a filename"""
    pattern=r"(?<=g2_)[\d]*\.[\d]*"
    result = re.search(pattern, fname)
    if result is not None:
        return float(result.group(0))
    else:
        return None

def fname2g2el(fname):
    """Extract the number of layers from a filename"""
    pattern=r"(?<=g2el_)[\d]*\.[\d]*"
    result = re.search(pattern, fname)
    if result is not None:
        return float(result.group(0))
    else:
        return None


def load_matrix_dat_fmt(path,is_complex=True):
    """Load matrix format exported from C++.

    Args:
        path (str): Path to file
        is_complex (bool, optional): Matrix is complex or not. Defaults to True.
    """
    complexptrn= re.compile(r'\(([^,\)]+),([^,\)]+)\)')

    def parse_complex(s):
        return complex(*map(float, complexptrn.match(s).groups()))
    def parse_real(s):
        return float(s)
    dest=[]
    with open(path,'r') as f:
        for line in f:
            line_short=re.sub(' +', ' ', line.strip())
            strvec=line_short.split(" ")
            numvec=[]
            for s in strvec:
                if is_complex:
                    num=parse_complex(s)
                else:
                    num=parse_real(s)
                numvec.append(num)
            dest.append(numvec)
    return np.array(dest)


def merge_measurements(meas1, meas2):
    dest = meas.Measurement(meas1.name, meas1.binsize)
    dest.extend(meas1.get_timeseries())
    dest.extend(meas2.get_timeseries())
    return dest


def mergeDict(dict1, dict2):
    """ Left Merge dictionaries that contain only lists and append lists if values are common"""
    dest = {}
    for key in dict1:
        if key in dict2:
            #We assume that there are only lists in the dictionaries
            dest[key] = merge_measurements(dict1[key], dict2[key])
        else:
            dest[key] = dict1[key]
    return dest


def print_columns(listvals, padding=4, header=False):
    col_width = max([len(str(word))
                     for row in listvals for word in row]) + padding
    for ind, row in enumerate(listvals):
        print("".join(str(word).ljust(col_width) for word in row))
        if header and ind == 0:
            print("")


def sizeof_fmt(num, suffix='B'):
    """Pretty print a size as mutliples of 1024."""
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%3.1f %s%s" % (num, 'Yi', suffix)


def get_git_hash():
    #This assumes that .git and util.py are in the same directory
    dir = os.path.dirname(os.path.realpath(__file__))
    gitdir = os.path.join(dir, ".git")
    githash = subprocess.check_output(
        ['git', '--git-dir={}'.format(gitdir), 'rev-parse', 'HEAD'])
    return githash.decode("utf-8").strip()

def select_except(arr,ind):
    #This function works only on the outer-most layer
    if isinstance(arr,list):
        arr = np.asarray(arr)
    mask = np.ones(len(arr), dtype=bool)
    mask[ind] = False
    return arr[mask]

def multiply_except(arr,ind):
    if len(arr)>1:
        others=select_except(arr,ind)
        return np.prod(others)
    else:
        #It does not make sense to execute this function with only one element
        return arr[0]

# =========== Matrix Evaluation Functions ====================

def is_hermitian(mat):
    """Returns true if the matrix is hermitian."""
    if issparse(mat):
        return np.allclose(mat.todense(), mat.H.todense())
    else:
        return np.allclose(np.conjugate(np.transpose(mat)), mat)


def is_diagonal(mat):
    """Returns true if the matrix is diagonal."""
    if issparse(mat):
        return np.allclose((mat-mat.diagonal()).todense(), np.zeros(mat.shape))
    else:
        return np.allclose(mat-np.diag(np.diag(mat)), np.zeros_like(mat))


def is_symmetric(mat):
    """Returns true if the matrix is symmetric. """
    if issparse(mat):
        return np.allclose(mat.todense(), mat.T.todense())
    else:
        return np.allclose(np.transpose(mat), mat)

def is_permutation(mat):
    """Returns true if the matrix is a permutation matrix. """
    n, m = mat.shape
    if issparse(mat):
        pass
    else:
        square = n == m
        id = np.allclose(np.eye(n), mat@np.transpose(mat))
        sum_rows = np.all(np.sum(mat, axis=0) == 1)
        sum_cols = np.all(np.sum(mat, axis=1) == 1)
        return square and id and sum_rows and sum_cols


def is_antisymmetric(mat):
    """Returns true if the matrix is symmetric. """
    if issparse(mat):
        return np.allclose(mat.todense(), -mat.T.todense())
    else:
        return np.allclose(-np.transpose(mat), mat)


def get_nonzero_fraction(mat):
    """Returns fraction of non-zero elements."""
    return np.count_nonzero(mat)/np.prod(mat.shape)


def herm_conj(mat):
    """Returns the hermitian conjugate of a matrix."""
    return np.conjugate(np.transpose(mat))


def commutator(mat1, mat2):
    """Calculate the commutator of two matrices

    Args:
        mat1 (2d np.ndarray): First argument of commutator
        mat2 (2d np.ndarray): Second argument of commutator

    Returns:
        2d np.ndarray: Commutator
    """
    return mat1@mat2-mat2@mat1


def anticommutator(mat1, mat2):
    """Calculate the anti-commutator of two matrices

    Args:
        mat1 (2d np.ndarray): First argument of anti-commutator
        mat2 (2d np.ndarray): Second argument of anti-commutator

    Returns:
        2d np.ndarray: Anti-commutator
    """
    return mat1@mat2+mat2@mat1

# =========== Covariance Utility Funcitons ===========


def tmat_to_covariance_matrix(tmat):
    m, n = tmat.shape
    id = np.eye(m)
    idinv = np.linalg.inv(id-tmat@np.conjugate(tmat))
    lt = -idinv@tmat
    rt = 0.5*idinv@(id+tmat@np.conjugate(tmat))
    lb = -np.conjugate(rt)
    rb = -np.conjugate(lt)
    return 1.j*np.block([[lt, rt], [lb, rb]])


def generate_smat(n):
    pattern = [[1], [1.j]]
    halfmat = np.kron(np.eye(n//2), pattern)
    return np.block([halfmat, np.conjugate(halfmat)])

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
        #We only save if the file does not exist yet
        if not os.path.isfile(fname):
            with gzip.open(fname, "wb") as outfile:
                pickle.dump(self.store, outfile)

    def list(self):
        print(self.store.keys)

    def __str__(self):
        print("CacheServer: {} Entries".format(len(self.store)))


# =========================== WoodburyInverter ===============================

class WoodburyInverter:
    def __init__(self, mat):
        self.ainv = np.linalg.inv(mat)

    def inv(self):
        return self.ainv

    def update(self, u, c, v):
        # We ware updating the matrix A according to A=A+UCV and recalculate the inverse afterwards
        if not np.allclose(c, 0):
            #We cannot update with C being zero since this matrix has no inverse
            cinv = np.linalg.inv(c)
            self.ainv -= ((self.ainv@u)@np.linalg.inv(cinv +
                                                      v@self.ainv@u))@(v@self.ainv)
        return self.ainv

    def update_index(self, m, indi, indj):
        #Construct two matrices to shift M to the correct position in A
        if not np.allclose(m, 0):
            # We cannot update with C being zero since this matrix has no inverse
            m_m, n_m = m.shape
            m_a, n_a = self.ainv.shape
            idmat = np.eye(m_m, n_m)
            u = np.zeros((m_a, m_m))
            v = np.zeros((n_m, n_a))
            u[indi:indi+m_m, 0:n_m] = idmat
            v[0:m_m, indj:indj+n_m] = idmat
            return self.update(u, m, v)
        else:
            return self.inv()


# =========================== IncDeterminant ===============================
class IncDeterminant:
    def __init__(self, a):
        self.detval = np.linalg.det(a)

    def update(self, ainv, u, c, v, store=True):
        #We ware updating the matrix A according to A=A+UCV and recalculate the inverse afterwards
        dest = self.detval
        if not np.allclose(c, 0):
            cinv = np.linalg.inv(c)
            dest = self.detval * \
                np.linalg.det(cinv + v @ ainv @ u) * np.linalg.det(c)
            if store:
                self.detval = dest
        return dest

    def det(self):
        return self.detval


def update_index(self, ainv, m, indi, indj, store=True):
    #Construct two matrices to shift M to the correct position in A
    if not np.allclose(m, 0):
        m_m, n_m = m.shape
        m_a, n_a = ainv.shape
        idmat = np.eye(m_m, n_m)
        u = np.zeros(m_a, m_m)
        v = np.zeros(n_m, n_a)
        u[indi:indi+m_m, 0:n_m] = idmat
        v[0:m_m, indj:indj+n_m] = idmat
        return self.update(ainv, u, m, v, store)
    else:
        return self.detval

# =========================== IncLogAbsDeterminant ===============================


class IncLogAbsDeterminant:
    def __init__(self, a):
        # We are not using the sign right now.
        # We know that the sign has to be positive
        self.sign, self.detval = np.linalg.slogdet(a)

    def det(self):
        return self.detval

    def update(self, ainv, u, c, v, store=True):
        # We are updating the matrix A according to A=A+UCV and recalculate the inverse afterwards
        dest = self.detval
        converged = True
        if not np.allclose(c, 0):
            # We cannot update if c is zero because we cannot invert it
            # There might also be problems if c is singular !
            sign, cdetval = np.linalg.slogdet(c)
            sign, combined_detval = np.linalg.slogdet(
                np.linalg.inv(c) + v @ ainv @ u)
            if np.isnan(combined_detval) or np.isnan(cdetval):
                converged = False
            if converged:
                dest = self.detval + cdetval + combined_detval
            if store:
                self.detval = dest
        return dest

    def update_index(self, ainv, m, indi, indj, store=True):
        #Construct two matrices to shift M to the correct position in A
        if not np.allclose(m, 0):
            # We cannot update if m is zero because we cannot invert it
            m_m, n_m = m.shape
            m_a, n_a = ainv.shape
            idmat = np.eye(m_m, n_m)
            u = np.zeros((m_a, m_m))
            v = np.zeros((n_m, n_a))
            u[indi:indi+m_m, 0:n_m] = idmat
            v[0:m_m, indj:indj+n_m] = idmat
            return self.update(ainv, u, m, v, store)
        else:
            return self.det()

class BgbTransform():

    def __init__(self,mat_in, pure_gauge=True):
        self.mat_in = mat_in
        self.is_pure_gauge = pure_gauge
        self._mat_out = None
    
    @property
    def mat_out(self):
        if self._mat_out is None:
            wn,s,wp=svd(self.mat_in, full_matrices=True, compute_uv=True)
            wp = herm_conj(wp)
            if not self.is_pure_gauge:
                # We are shuffling the physical mode to the front again
                # It would look like s=perm*s
                #diag = np.ones(wn.shape[0] - 1)
                #perm = np.zeros((wn.shape[0], wn.shape[0]))
                #sub_diag= np.diag(perm,k=-1) 
                #sub_diag= diag
                #perm[0, :- 1] = 1
                # Apply the permutation
                #wn = wn * perm.transpose()
                pass
            un = herm_conj(wn)
            # now we got the transpose of wp
            up = np.transpose(wp)
            un_rows, un_cols = un.shape
            up_rows, up_cols = up.shape
            unitary_transform = np.zeros((un.shape[0] + up.shape[0], un.shape[1] + up.shape[1]),dtype=complex)
            unitary_transform[:un_rows, :un_cols] = un
            unitary_transform[-up_rows:, -up_cols:] = up

            trafo_size = len(s) * 2 if self.is_pure_gauge else len(s) * 2 + 1
            start_ind = 0 if self.is_pure_gauge else 1
            r0_diagonal = np.zeros(trafo_size, dtype=complex)
            if not self.is_pure_gauge:
                r0_diagonal[0] = 1j / 2.
            r0_diagonal[start_ind: start_ind+len(s)] = 1j / 2. * (1 - s**2) / (1 + s**2)
            r0_diagonal[-len(s):] = 1j / 2. * (1 - s**2) / (1 + s**2)
            r0 = np.diag(r0_diagonal)

            q0_offdiagonal = np.zeros(len(s), dtype=complex)
            q0_offdiagonal = 1j * s / (1 + s**2)
            q0_block = np.diag(q0_offdiagonal)
            q0 = np.zeros((trafo_size, trafo_size), dtype=complex)
            if not self.is_pure_gauge:
                q0[0, 0] = 0
            q0[start_ind:start_ind+len(s), start_ind + len(s):start_ind+2*len(s)] = -q0_block
            q0[start_ind + len(s): start_ind+2*len(s), start_ind:start_ind+len(s)] = q0_block

            gamma0 = np.zeros((2 * trafo_size, 2 * trafo_size), dtype=complex)
            gamma0=np.block([[q0,r0],[np.conj(r0),np.conj(q0)]])
            trafo_0 = block_diag(herm_conj(unitary_transform),np.transpose(unitary_transform))
            trafo_1 = block_diag(np.conj(unitary_transform),unitary_transform)
            # This matrix has the following order: psi, r+, u-, l-, d+,t,b, r-, l+,
            # u+, d-,t,b psi_dag, r+_dag, l-_dag, u-_dag, d+_dag,t_dag,b_dag,
            # r-_dag, l+_dag, u+_dag, d-_dag, t_dag, b_dag.
            self._mat_out = trafo_0 @ gamma0 @ trafo_1
        return self._mat_out

# ========= Rebinning Functions ====================

def autocorr_fft(arr):
    arr=arr-np.mean(arr)
    fft_vals=np.fft.fft(arr)
    spectrum=fft_vals*np.conjugate(fft_vals)
    dest=np.fft.ifft(spectrum)
    return dest/dest[0]

def rebin_array(a, R):
    """Rebin an array into bins of length R"""
    if isinstance(a, list):
        a = np.asarray(a)
    R=int(R)
    max_fit = int(len(a) - len(a) % R)
    if a.ndim == 1:
        #Shape (N): N samples of scalars
        dest = np.mean(a[:max_fit].reshape(-1, R), axis=1)
    elif a.ndim==2:
        #Shape (N,n,m): N samples of m-dim vecotrs
        N,m=a.shape
        dest = np.mean(a[:max_fit].reshape(-1, m, R), axis=2)
    elif a.ndim == 3:
        #Shape (N,n,m): N samples of n x m matrices
        N,m,n=a.shape
        dest = np.mean(a[:max_fit].reshape(-1, m, n, R), axis=3)
    else:
        logging.error("rebin_array not implemented for dimensions greater than 3.")
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
    rangevals = [2**i for i in range(max_exp+1)]
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


def rebin_eom(arr):
    """Calculate the error on the mean (EOM) by rebinning.
    As a heuristic for the EOM we use that the biggest bin will give the best estimate.
    We do not rebin to the maximal extent, but use the heuristic of taking the largest binsize of the form 2^i that can fit N/20.

    Args:
        arr (np.ndarray): Timeseries of a measurement

    Returns:
        float or arr: Best estimate of the EOM on the given array. The output shape depends on the input shape of arr.
    """
    N = len(arr)
    #We want to leave a sufficient number of samples to build a reasonable mean
    max_exp = int(np.floor(np.log2(N / 10)))
    if max_exp > 0:
        binsize= 2**(max_exp-1)
        data_rebin = rebin_array(arr, binsize)
    else:
        # We cannot rebin if we have too few data. We will just return the normal EOM
        data_rebin = arr
    eom = np.std(data_rebin, ddof=1, axis=0) / np.sqrt(len(data_rebin))
    return eom


#========== Debugging Functions ====================

def show_vector(vec, title=None):
    """Display a matrix and interrupt the program. """
    f, ax = plt.subplots(1, 1)
    ax.plot(vec)
    if title is not None and len(title) > 0:
        plt.title(title)
    plt.show()

def show_matrix(mat, title=None, **kwargs):
    """Display a matrix and interrupt the program. """
    show_matrixvec([mat],title=[title],**kwargs)

def show_matrixvec(matvec, title=None, log=False):
    """Display a matrix and interrupt the program. """
    f, axvec = plt.subplots(1, len(matvec))
    if len(matvec)==1:
        axvec=[axvec]
    for ind, mat in enumerate(matvec):
        if log:
            minval = np.min(mat)
            if minval == 0:
                #This is a dirty hack to display the 0 in a log plot
                mat += 1e-10
                minval += 1e-10
            matax = axvec[ind].matshow(
                mat, norm=LogNorm(vmin=minval, vmax=np.max(mat)))
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
    print("Min:\t{}".format(np.min(mat)))
    print("Max:\t{}".format(np.max(mat)))
    print("Avg:\t{}".format(np.mean(mat)))
    print("Norm:\t{}".format(np.linalg.norm(mat)))


def show_eigenvalues(mat):
    if is_hermitian(mat):
        #Plot the real eigenvalues
        f, ax = plt.subplots(1, 1)
        eigvals = np.linalg.eigvalsh(mat)
        ax.plot(eigvals, 'o')
    else:
        #Plot the real eigenvalues
        f, ax = plt.subplots(1, 2)
        eigvals = np.linalg.eigvals(mat)
        ax[0].set_title("Real part")
        ax[0].plot(np.real(eigvals), 'o')
        ax[0].set_title("Imaginary part")
        ax[1].plot(np.imag(eigvals), 'o')
    plt.show()
