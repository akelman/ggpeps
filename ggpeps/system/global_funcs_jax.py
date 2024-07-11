############## JAX VERSIONS ##############

from typing import List

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, device_put
# The following line ensures that JAX is configured to 64-bit precision.
# Without this line, some of the precision tests do not pass.
jax.config.update("jax_enable_x64", True)

import ggpeps

@jit
def calculate_lognormvec_jit(gamma_in_sys: jnp.ndarray, mat_d: jnp.ndarray) -> float:
    # This is still the plain formula, without any update mechanism    
    # We are skipping a global factor of 2**(-n) here, to get a reasonable size of the norm
    sign, logval = jnp.linalg.slogdet(
        (jnp.eye(mat_d.shape[0]) - gamma_in_sys @ mat_d))
    return logval

batch_calculate_lognormvec = jax.vmap(calculate_lognormvec_jit)

def calculate_lognormvec_jax(gamma_in_sys_vec: List[np.ndarray], mat_d_vec: List[np.ndarray], all_factors=False) -> float:
    
    #gamma_in_sys_vec_jax = device_put(jnp.array(gamma_in_sys_vec), device=ggpeps.PREFERRED_DEVICE)
    #mat_d_vec_jax = device_put(jnp.array(mat_d_vec), device=ggpeps.PREFERRED_DEVICE)
    dest = batch_calculate_lognormvec(jnp.array(gamma_in_sys_vec), mat_d_vec)
    #dest = jax.device_get(dest_jax)
    
    if all_factors:
        dest = dest - mat_d_vec[0].shape[0] * np.log(2) # add back in global factor of 2**(-n)

    # The factor 1/2 is the square-root
    return dest / 2

@jit # Just-In-Time compilation decorator for GPU optimization
def compute_grad_over_norm_jit(gamma_in_sys, diff, deriv_d, mat_d_inv):
    dest = -0.5 * jnp.trace(jnp.matmul(jnp.matmul(gamma_in_sys, deriv_d), jnp.matmul(mat_d_inv, diff)))
    return dest

@jit
def compute_grad_over_norm_jax(gamma_in_sys: np.ndarray, diff: np.ndarray, deriv_d: np.ndarray, mat_d_inv: np.ndarray):

    '''
    # Converts the input NumPy arrays into JAX arrays and moves them to the selected device (GPU or CPU).
    # This step ensures that the computation utilizes the appropriate hardware (GPU acceleration if possible).
    gamma_in_sys_jax = device_put(jnp.array(gamma_in_sys), device=ggpeps.PREFERRED_DEVICE)
    diff_jax = device_put(jnp.array(diff), device=ggpeps.PREFERRED_DEVICE)
    deriv_d_jax = device_put(jnp.array(deriv_d), device=ggpeps.PREFERRED_DEVICE)
    mat_d_inv_jax = device_put(jnp.array(mat_d_inv), device=ggpeps.PREFERRED_DEVICE)
    '''

    # Calls the JIT-compiled function to perform the computation. The JIT (Just-In-Time) compilation
    # is used to optimize the function for faster execution on the selected device.
    # This step performs the actual gradient-over-norm computation.
    result = compute_grad_over_norm_jit(gamma_in_sys, diff, deriv_d, mat_d_inv)

    '''
    # Transfers the result back to the CPU. This is necessary because the JIT-compiled function
    # may return a result on the GPU, and further CPU-based processing or analysis might be required.
    result_cpu = jax.device_get(result)

    # Converts the result from a JAX array (which may still be an array even for scalar results)
    # to a standard Python scalar (float). This conversion simplifies further usage of the result
    # in Python code that expects standard scalar types.
    scalar_result_cpu = result_cpu.item()

    return scalar_result_cpu
    '''
    return result

@jit
def derivative_pfaffian_jax(pfaval, mat, d_mat):
    """Compute the derivative of a Pfaffian of a matrix A.
    The numpy version of this function is in ggpeps.utils.
    """
    #pfaval = pfaffian_LTL_jax(mat)
    #rtol=1.e-5
    #atol=1.e-8
    #return jax.lax.select(not abs(pfaval) <= atol + rtol * jnp.abs(pfaval), 0.5 * pfaval * jnp.trace(jnp.linalg.inv(mat) @ d_mat), 0.0 )
    #if not abs(pfaval) <= atol + rtol * jnp.abs(pfaval): # replacement for utils.isclose
    return 0.5 * pfaval * jnp.trace(jnp.linalg.inv(mat) @ d_mat)

def compute_el_grad_vec_jax(system):

    dest_grad = []
    el_energy_vec = device_put(system.el_energy_op_vec, device=ggpeps.PREFERRED_DEVICE) #this gets the electric energy, and ensures that the intermediate steps are calculated

    layers = [k for k in range(system.cfg.nlayer)]

    # TODO: cleanup comments
    mat_b_vec = system.mat_b_mod_vec # device_put(jnp.asarray(system.mat_b_mod_vec), device=ggpeps.PREFERRED_DEVICE)
    diff_d_gamma_inv_vec = device_put(jnp.asarray([system.wi_gamma_out_mod_vec[layerind].inv() for layerind in layers]), device=ggpeps.PREFERRED_DEVICE) # this does not actually do a computation, just a retrieval
    single_link_offset = 2 * system.cfg.nvirtmodes_link
    offset = 2 * system.cfg.lattice.size + single_link_offset
    overall_factors = jnp.asarray(system.el_overall_factors) # device_put(jnp.asarray(system.el_overall_factors), device=ggpeps.PREFERRED_DEVICE)
    idxarrs_prefactors = jnp.asarray([[t[0] for t in system.idxarr_vec[layerind]] for layerind in layers]) # device_put(jnp.asarray([[t[0] for t in system.idxarr_vec[layerind]] for layerind in layers]), device=ggpeps.PREFERRED_DEVICE)
    idxarrs_indices = jnp.asarray([[t[1] for t in system.idxarr_vec[layerind]] for layerind in layers]) # device_put(jnp.asarray([[t[1] for t in system.idxarr_vec[layerind]] for layerind in layers]), device=ggpeps.PREFERRED_DEVICE)
    nlinks = system.cfg.lattice.nlinks
    gamma_in_sys_mod_vec = jnp.asarray(system.gamma_in_sys_mod_vec) # device_put(jnp.asarray(system.gamma_in_sys_mod_vec), device=ggpeps.PREFERRED_DEVICE)
    diff_d_inv_gamma_inv_vec = jnp.asarray([system.wi_gamma_in_mod_vec[layerind].inv() for layerind in layers]) # device_put(jnp.asarray([system.wi_gamma_in_mod_vec[layerind].inv() for layerind in layers]), device=ggpeps.PREFERRED_DEVICE)
    mat_d_mod_inv_vec = jnp.asarray(system.mat_d_mod_inv_vec) # device_put(jnp.asarray(system.mat_d_mod_inv_vec), device=ggpeps.PREFERRED_DEVICE)

    # get saved intermediate results from electric energy calculation
    intermediate = system._electric_energy_intermediate_vals 
    covmat_out_virt_vec = jnp.asarray(intermediate.covmat_out_virt_vec) # device_put(jnp.asarray(intermediate.covmat_out_virt_vec), device=ggpeps.PREFERRED_DEVICE)
    norm_mod_vec = jnp.asarray(intermediate.norm_mod_vec) # device_put(jnp.asarray(intermediate.norm_mod_vec), device=ggpeps.PREFERRED_DEVICE)
    lognorm_default_vec = jnp.asarray(intermediate.lognorm_default_vec) # device_put(jnp.asarray(intermediate.lognorm_default_vec), device=ggpeps.PREFERRED_DEVICE)
    pfaffian_vec = jnp.asarray(intermediate.pfaffian_vec) # device_put(jnp.asarray(intermediate.pfaffian_vec), device=ggpeps.PREFERRED_DEVICE)

    # these depend on the symbol
    deriv_gamma_maj_sys_vec = jnp.asarray([[system.gamma_maj_sys_deriv_vec(symbol)[layerind] for symbol in system.symbolvec] for layerind in layers]) # device_put(, device=ggpeps.PREFERRED_DEVICE)
    trace_def_vec = jnp.asarray([[system.compute_grad_over_norm(symbol, layerind) for symbol in system.symbolvec] for layerind in layers]) # device_put(, device=ggpeps.PREFERRED_DEVICE)

    dest_jax = batch_calculate_el_grads_all_layers(el_energy_vec, mat_b_vec, diff_d_gamma_inv_vec, single_link_offset, offset, pfaffian_vec, idxarrs_prefactors, idxarrs_indices, overall_factors, nlinks, gamma_in_sys_mod_vec, diff_d_inv_gamma_inv_vec, covmat_out_virt_vec, norm_mod_vec, lognorm_default_vec, deriv_gamma_maj_sys_vec, mat_d_mod_inv_vec, trace_def_vec)
    dest_grad = dest_jax # np.asarray(dest_jax).copy()

    # We have to weigh the different layers with the electric energy operator expectation of the other layers.
    # They act as a prefactor in the derivative
    if system.cfg.nlayer > 1:
        for i in range(system.cfg.nlayer):
            prod_other_layers = ggpeps.utils.multiply_except(el_energy_vec, i)
            dest_grad = dest_grad.at[i].multiply(prod_other_layers)
    
    system.cfg.enforce_parameter_conditions(dest_grad)
    return dest_grad

@jit
def deriv_pfarr_jax(covmat_out_virt, d_covmat_out_virt, pfaval, prefactor, ind):
    res = prefactor * derivative_pfaffian_jax(pfaval, covmat_out_virt[jnp.ix_(ind,ind)], d_covmat_out_virt[jnp.ix_(ind,ind)]) 
    return res
in_axes_pfarr = (None, None, 0, 0, 0)
batch_pfarr = jax.vmap(deriv_pfarr_jax, in_axes=in_axes_pfarr)

def gamma_in_sys_mod_jax(gamma_in_sys, single_link_offset):
    """Get function to return the gauged gamma_in_sys with a single link modification (to compute the electric energy), 
    the covariance matrix of the links for the whole system.

    Returns:
        np.ndarray: Gauged, modified covariance matrix of the system
    """
    N = gamma_in_sys.shape[0]
    return jax.lax.slice(gamma_in_sys, (single_link_offset, single_link_offset), (N, N)) # TODO: should this be dynamic_slice?

def extract_partial_covmats_jax(mat, corner):
    """Extract the partial covariance matrices from a gaussian mapping

    Args:
        mat (np.ndarray): Full covariance matrix
        corner (int): Index of the top left element of the bottom right matrix

    Returns:
        tuple: Matrices (A,B,D)
    """
    N = mat.shape[0]
    mat_a = jax.lax.slice(mat, (0,0), (corner, corner)) # TODO: should this be dynamic_slice?
    mat_b = jax.lax.slice(mat, (0,corner), (corner, N))
    mat_d = jax.lax.slice(mat, (corner,corner), (N, N))
    return mat_a, mat_b, mat_d

def slice_matrix_jax(mat, a, b, c, d):
    return jax.lax.slice(mat, (a,c), (b,d))

def compute_el_grad_onelayer_onesymbol(
        el_energy,
        mat_b,
        diff_d_gamma_inv,
        single_link_offset,
        offset,
        pfaffian_vals,
        idxarrs_prefactors, 
        idxarrs_indices,
        overall_factor,
        nlinks,
        gamma_in_sys_mod,
        diff_d_inv_gamma_inv,
        covmat_out_virt,
        norm_mod,
        lognorm_default, 
        deriv_gamma_maj_sys,
        mat_d_mod_inv,
        trace_def):

    ###################### Calculation of the derivative ########################
    d_mat_a, d_mat_b, d_mat_d = extract_partial_covmats_jax(deriv_gamma_maj_sys, offset)
    d_gamma_out = d_mat_a + \
            d_mat_b @ diff_d_gamma_inv @ jnp.transpose(mat_b) \
            + mat_b @ diff_d_gamma_inv @ jnp.transpose(d_mat_b) \
            - mat_b @ diff_d_gamma_inv @ d_mat_d @ diff_d_gamma_inv @ jnp.transpose(mat_b)
    # The virtual mode is the last link on the bottom right of the covariance matrix
    d_covmat_out_virt = d_gamma_out[-single_link_offset:, -single_link_offset:]
    # Summand with derivative of the covariance matrix
    # We re-use the list comprehension from above to use the indices
    deriv_pfarr = batch_pfarr(covmat_out_virt, d_covmat_out_virt, pfaffian_vals, idxarrs_prefactors, idxarrs_indices)
    d_el_energy = jnp.real(overall_factor * jnp.sum(deriv_pfarr)) * jnp.exp(norm_mod - lognorm_default)
    
    # Summand with derivative of norms
    trace_mod = compute_grad_over_norm_jit(gamma_in_sys_mod, diff_d_inv_gamma_inv, d_mat_d, mat_d_mod_inv)
    # This is the second contribution of the elctric energy gradient F_{el} (\tilde(v) - v)
    d_el_energy += el_energy * (trace_mod - trace_def)
    # Scale to system size
    d_el_energy *= nlinks

    return jnp.real(d_el_energy)

in_axes_symbols = (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0, None, 0)
batch_calculate_el_grads_all_symbols = jax.vmap(compute_el_grad_onelayer_onesymbol, in_axes=in_axes_symbols) # for one layer, all symbols
in_axes_layers = (0, 0, 0, None, None, 0, 0, 0, 0, None, 0, 0, 0, 0, 0, 0, 0, 0)
batch_calculate_el_grads_all_layers = jax.vmap(batch_calculate_el_grads_all_symbols, in_axes=in_axes_layers) # for all layers, all symbols

