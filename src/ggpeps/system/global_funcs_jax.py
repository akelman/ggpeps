############## JAX VERSIONS ##############

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, device_put

# The following line ensures that JAX is configured to 64-bit precision.
# Without this line, some of the precision tests do not pass.
jax.config.update("jax_enable_x64", True)

import py_pfaffian.jax

import ggpeps
import ggpeps.utils as utils
from ggpeps.system.backend_base import BackendBase


def derivative_pfaffian_jax(mat, d_mat, pfaval=None):
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
        pfaval = py_pfaffian.jax.pfaffian(mat)

    return 0.5 * pfaval * jnp.trace(jnp.linalg.inv(mat) @ d_mat)


@jit
def calculate_lognormvec_jit(gamma_in_sys: jnp.ndarray, mat_d: jnp.ndarray) -> float:
    # This is still the plain formula, without any update mechanism
    # We are skipping a global factor of 2**(-n) here, to get a reasonable size of the norm
    sign, logval = jnp.linalg.slogdet((jnp.eye(mat_d.shape[0]) - gamma_in_sys @ mat_d))
    return logval


batch_calculate_lognormvec = jax.vmap(calculate_lognormvec_jit)


@partial(jax.jit, static_argnames=["all_factors"])
def calculate_lognormvec_jax(
    gamma_in_sys_vec,
    mat_d_vec,
    all_factors: bool = False,
) -> float:

    dest = batch_calculate_lognormvec(jnp.array(gamma_in_sys_vec), mat_d_vec)

    if all_factors:
        # add back in global factor of 2**(-n)
        dest = dest - mat_d_vec[0].shape[0] * jnp.log(2)

    # The factor 1/2 is the square-root
    return dest / 2


@jit
def compute_grad_over_norm_jax(
    gamma_in_sys: jnp.ndarray,
    diff: jnp.ndarray,
    deriv_d: jnp.ndarray,
    mat_d_inv: jnp.ndarray,
) -> float:
    """Compute the gradient of the norm divided by the norm.
    The expression of deriv_d given to this function decides which derivative is computed

    The gradient of the norm divided by the norm is given by
        -0.5 * np.trace(gamma_in_sys @ deriv_d @ mat_d_inv @ diff)
    which is very expensive to calculate.
    To reduce the number of expensive matrix multiplications, we use the fact that
        Tr(A @ B.T) = sum_ij a_ij b_ij
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
    dest = -0.5 * jnp.trace(jnp.matmul(jnp.matmul(gamma_in_sys, deriv_d), jnp.matmul(mat_d_inv, diff)))
    return dest


class BackendJax_Z2(BackendBase):
    """Backend for Z2 systems using jax."""

    backend_type = "jax"
    gauge_group = "Z2"

    def __init__(self) -> None:
        pass

    @staticmethod
    def array_assign(mat, inds, val):
        mat = mat.at[inds].set(val)
        return mat

    @staticmethod
    def array_add(mat, inds, val):
        mat = mat.at[inds].add(val)
        return mat

    @staticmethod
    def calculate_lognormvec(gamma_in_sys_vec, mat_d_vec, all_factors=False):
        return calculate_lognormvec_jax(gamma_in_sys_vec, mat_d_vec, all_factors=all_factors)
