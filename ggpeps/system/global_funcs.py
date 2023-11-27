from typing import Union, List
import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, device_put
# The following 2 lines ensure that JAX is configured to 64-bit precision.
# Without these 2 lines, some of the precision tests do not pass.
from jax.config import config
config.update("jax_enable_x64", True)

import ggpeps


############## NUMPY CPU VERSIONS ##############

def calculate_lognormvec_numpy(gamma_in_sys_vec: List[np.ndarray], mat_d_vec: np.ndarray, all_factors=False) -> float:
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


def compute_grad_over_norm_numpy(gamma_in_sys: np.ndarray, 
                           diff: np.ndarray,
                           deriv_d: np.ndarray,
                           mat_d_inv: np.ndarray) -> float:
    """Compute the gradient of the norm divided by the norm.
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
    dest = -0.5 * (A*B.T).sum()
    return dest


############## JAX VERSIONS ##############

@jit # Just-In-Time compilation decorator for GPU optimization
def compute_grad_over_norm_jit(gamma_in_sys, diff, deriv_d, mat_d_inv):
    dest = -0.5 * jnp.trace(jnp.matmul(jnp.matmul(gamma_in_sys, deriv_d), jnp.matmul(mat_d_inv, diff)))
    return dest

@jit
def compute_grad_over_norm_jax(gamma_in_sys: np.ndarray, diff: np.ndarray, deriv_d: np.ndarray, mat_d_inv: np.ndarray):

    # Converts the input NumPy arrays into JAX arrays and moves them to the selected device (GPU or CPU).
    # This step ensures that the computation utilizes the appropriate hardware (GPU acceleration if possible).
    gamma_in_sys_jax = device_put(jnp.array(gamma_in_sys), device=ggpeps.PREFERRED_DEVICE)
    diff_jax = device_put(jnp.array(diff), device=ggpeps.PREFERRED_DEVICE)
    deriv_d_jax = device_put(jnp.array(deriv_d), device=ggpeps.PREFERRED_DEVICE)
    mat_d_inv_jax = device_put(jnp.array(mat_d_inv), device=ggpeps.PREFERRED_DEVICE)

    # Calls the JIT-compiled function to perform the computation. The JIT (Just-In-Time) compilation
    # is used to optimize the function for faster execution on the selected device.
    # This step performs the actual gradient-over-norm computation.
    result = compute_grad_over_norm_jit(gamma_in_sys_jax, diff_jax, deriv_d_jax, mat_d_inv_jax)

    # Transfers the result back to the CPU. This is necessary because the JIT-compiled function
    # may return a result on the GPU, and further CPU-based processing or analysis might be required.
    result_cpu = jax.device_get(result)

    # Converts the result from a JAX array (which may still be an array even for scalar results)
    # to a standard Python scalar (float). This conversion simplifies further usage of the result
    # in Python code that expects standard scalar types.
    scalar_result_cpu = result_cpu.item()

    return scalar_result_cpu


############## SELECT APPROPRIATE VERSION ##############
if ggpeps.GPU_AVAILABLE:
    calculate_lognormvec = calculate_lognormvec_jax
    compute_grad_over_norm = compute_grad_over_norm_jax
else:
    calculate_lognormvec = calculate_lognormvec_numpy
    compute_grad_over_norm = compute_grad_over_norm_numpy