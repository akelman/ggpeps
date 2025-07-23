############## JAX VERSIONS ##############

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, device_put

# The following line ensures that JAX is configured to 64-bit precision.
# Without this line, some of the precision tests do not pass.
jax.config.update("jax_enable_x64", True)

import ggpeps
from ggpeps.system.backend_base import BackendBase


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


# Two issues:
# (a) cannot jit, because this function calls pfaffian code, which is not built for jax
# (b) TODO: should accept required matrices, which will not need to be static
# @partial(jax.jit, static_argnames=["system"])
def compute_el_grad_vec_jax(system):
    """Computation of the electric energy gradients.
    We start by calculating the electric energies, since these are needed for evaluating the gradients.
    Since several operations needed for the computation of the gradient and the energy are similar, we can reuse many intermediate steps.

    This method overwrites an abstract method in System2DBase.

    Args:
        use_trans_inv (bool, optional): Use the translationally invariant implementation. Defaults to True.

    Returns:
        list: list of gradients for the full system
    """

    dest_grad = jnp.zeros(system.cfg.param_shape(), dtype=jnp.float64)
    overall_factors = system.cfg.el_overall_factors
    idxarrs = system.cfg.idxarr_vec
    el_energy_vec = system.el_energy_op_vec

    for layerind in range(system.cfg.nlayer):

        # Abbreviations for more readable code
        mat_b = system.mat_b_mod_vec[layerind]
        diff_d_gamma_inv = system.wi_gamma_out_mod_vec[
            layerind
        ].inv()  # this does not actually do a computation, just a retrieval
        single_link_offset = 2 * system.cfg.nvirtmodes_link
        offset = 2 * system.cfg.lattice.size * system.cfg.nphysmodes_site + single_link_offset
        idxarr = idxarrs[layerind]
        overall_factor = overall_factors[layerind]
        nlinks = system.cfg.lattice.nlinks
        gamma_in_sys_mod = system.gamma_in_sys_mod_vec[layerind]
        diff_d_inv_gamma_inv = system.wi_gamma_in_mod_vec[layerind].inv()

        covmat_out_virt = system.covmat_out_virt_vec[layerind]
        norm_mod = system.norm_mod_vec[layerind]
        lognorm_default = jnp.sum(system.lognorm_default_vec)

        ###################### Calculation of the derivative ########################
        for uc_ind in range(system.cfg.unitcell_size):
            for symbol_ind, symbol in enumerate(system.symbolvec):
                if (layerind, symbol_ind) in system.cfg.zeroed_params:
                    # the derivative calculation is compuationally expensive
                    # we can skip it for parameters that are forced by the ansatz to be zero
                    dest_grad.at[layerind, uc_ind, symbol_ind].set(0)
                else:
                    deriv_gamma_maj_sys = system.gamma_maj_sys_deriv_vec(symbol)[layerind, uc_ind]
                    d_mat_a, d_mat_b, d_mat_d = extract_partial_covmats_jax(deriv_gamma_maj_sys, offset)
                    d_gamma_out = (
                        d_mat_a
                        + d_mat_b @ diff_d_gamma_inv @ jnp.transpose(mat_b)
                        + mat_b @ diff_d_gamma_inv @ jnp.transpose(d_mat_b)
                        - mat_b @ diff_d_gamma_inv @ d_mat_d @ diff_d_gamma_inv @ jnp.transpose(mat_b)
                    )
                    # The virtual mode is the last link on the bottom right of the covariance matrix
                    d_covmat_out_virt = d_gamma_out[-single_link_offset:, -single_link_offset:]
                    # Summand with derivative of the covariance matrix
                    # We re-use the list comprehension from above to use the indices
                    deriv_pfarr = [
                        prefactor
                        * ggpeps.utils.derivative_pfaffian(
                            covmat_out_virt[jnp.ix_(jnp.array(ind), jnp.array(ind))],
                            d_covmat_out_virt[jnp.ix_(jnp.array(ind), jnp.array(ind))],
                            backend="jax",
                        )
                        for prefactor, ind in idxarr
                    ]
                    d_el_energy = jnp.real(overall_factor * jnp.sum(jnp.array(deriv_pfarr))) * jnp.exp(
                        norm_mod - lognorm_default
                    )

                    # Summand with derivative of norms
                    trace_def = system.compute_grad_over_norm(symbol, layerind, uc_ind)
                    trace_mod = compute_grad_over_norm_jax(
                        gamma_in_sys_mod,
                        diff_d_inv_gamma_inv,
                        d_mat_d,
                        system.mat_d_mod_inv_vec[layerind],
                    )
                    # This is the second contribution of the elctric energy gradient F_{el} (\tilde(v) - v)
                    d_el_energy += el_energy_vec[layerind] * (trace_mod - trace_def)
                    # Scale to system size
                    d_el_energy *= nlinks
                    dest_grad.at[layerind, uc_ind, symbol_ind].set(jnp.real(d_el_energy))

    # We have to weigh the different layers with the electric energy operator expectation of the other layers.
    # They act as a prefactor in the derivative
    if system.cfg.nlayer > 1:
        for i in range(system.cfg.nlayer):
            prod_other_layers = ggpeps.utils.multiply_except(el_energy_vec, i)
            dest_grad = dest_grad.at[i].multiply(prod_other_layers)

    system.cfg.enforce_parameter_conditions(dest_grad)
    return dest_grad


def extract_partial_covmats_jax(mat, corner):
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


def slice_matrix_jax(mat, a, b, c, d):
    return mat[a:b, c:d]


def gamma_in_sys_mod_jax(gamma_in_sys, single_link_offset):
    """Get function to return the gauged gamma_in_sys with a single link modification (to compute the electric energy),
    the covariance matrix of the links for the whole system.

    Returns:
        np.ndarray: Gauged, modified covariance matrix of the system
    """
    return gamma_in_sys[single_link_offset:, single_link_offset:]


class BackendJax_Z2(BackendBase):
    """Backend for Z2 systems using jax."""

    backend_type = "jax"
    gauge_group = "Z2"

    def __init__(self) -> None:
        pass

    @staticmethod
    def slice_matrix(mat, a, b, c, d):
        return slice_matrix_jax(mat, a, b, c, d)

    @staticmethod
    def extract_partial_covmats(mat, corner):
        return extract_partial_covmats_jax(mat, corner)

    @staticmethod
    def calculate_lognormvec(gamma_in_sys_vec, mat_d_vec, all_factors=False):
        return calculate_lognormvec_jax(gamma_in_sys_vec, mat_d_vec, all_factors=all_factors)

    @staticmethod
    def gamma_in_sys_mod(gamma_in_sys, single_link_offset):
        return gamma_in_sys_mod_jax(gamma_in_sys, single_link_offset)

    @staticmethod
    def compute_grad_over_norm(gamma_in_sys, diff, deriv_d, mat_d_inv):
        return compute_grad_over_norm_jax(gamma_in_sys, diff, deriv_d, mat_d_inv)

    @staticmethod
    def compute_el_grad_vec(system):
        return compute_el_grad_vec_jax(system)
