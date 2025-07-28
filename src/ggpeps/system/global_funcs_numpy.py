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


def compute_el_grad_vec_numpy(
    lattice_size: int,
    num_pg_layer: int,
    num_fermionic_layer: int,
    unitcell_size: int,
    nvirtmodes_link: int,
    nphysmodes_site: int,
    symbolvec: tuple,
    overall_factors,
    idxarr_vec,
    el_energy_vec,
    mat_b_mod_vec,
    gamma_in_sys_mod_vec,
    covmat_out_virt_vec,
    norm_mod_vec,
    lognorm_default_vec,
    wi_gamma_in_mod_inv_vec,
    wi_gamma_out_mod_inv_vec,
    mat_d_mod_inv_vec,
    gamma_maj_sys_deriv_layvec_ucvec_symbvec,
    grad_over_norm_vec,
    zeroed_params,
) -> np.ndarray:
    """Computation of the electric energy gradients.
    We start by calculating the electric energies, since these are needed for evaluating the gradients.
    Since several operations needed for the computation of the gradient and the energy are similar, we can reuse many intermediate steps.

    This method overwrites an abstract method in System2DBase.

    Args:
        use_trans_inv (bool, optional): Use the translationally invariant implementation. Defaults to True.

    Returns:
        list: list of gradients for the full system
    """

    nlayer = num_pg_layer + num_fermionic_layer
    param_shape = (nlayer, unitcell_size, len(symbolvec))
    dest_grad = np.zeros(param_shape, dtype=np.float64)

    for layerind in range(nlayer):

        # Abbreviations for more readable code
        mat_b = mat_b_mod_vec[layerind]
        diff_d_gamma_inv = wi_gamma_out_mod_inv_vec[layerind]
        single_link_offset = 2 * nvirtmodes_link
        offset = 2 * lattice_size * nphysmodes_site + single_link_offset
        idxarr = idxarr_vec[layerind]
        overall_factor = overall_factors[layerind]
        nlinks = 2 * lattice_size  # valid for 2D with periodic boundary conditions
        gamma_in_sys_mod = gamma_in_sys_mod_vec[layerind]
        diff_d_inv_gamma_inv = wi_gamma_in_mod_inv_vec[layerind]

        covmat_out_virt = covmat_out_virt_vec[layerind]
        norm_mod = norm_mod_vec[layerind]
        lognorm_default = np.sum(lognorm_default_vec)

        ###################### Calculation of the derivative ########################
        for uc_ind in range(unitcell_size):
            for symbol_ind, symbol in enumerate(symbolvec):
                if (layerind, uc_ind, symbol_ind) in zeroed_params:
                    # the derivative calculation is compuationally expensive
                    # we can skip it for parameters that are forced by the ansatz to be zero
                    dest_grad[layerind, uc_ind, symbol_ind] = 0
                else:
                    deriv_gamma_maj_sys = gamma_maj_sys_deriv_layvec_ucvec_symbvec[layerind, uc_ind, symbol_ind]
                    d_mat_a, d_mat_b, d_mat_d = utils.extract_partial_covmats(deriv_gamma_maj_sys, offset)
                    d_gamma_out = (
                        d_mat_a
                        + d_mat_b @ diff_d_gamma_inv @ np.transpose(mat_b)
                        + mat_b @ diff_d_gamma_inv @ np.transpose(d_mat_b)
                        - mat_b @ diff_d_gamma_inv @ d_mat_d @ diff_d_gamma_inv @ np.transpose(mat_b)
                    )
                    # The virtual mode is the last link on the bottom right of the covariance matrix
                    d_covmat_out_virt = d_gamma_out[-single_link_offset:, -single_link_offset:]
                    # Summand with derivative of the covariance matrix
                    # We re-use the list comprehension from above to use the indices
                    deriv_pfarr = [
                        prefactor
                        * utils.derivative_pfaffian(
                            covmat_out_virt[np.ix_(ind, ind)],
                            d_covmat_out_virt[np.ix_(ind, ind)],
                        )
                        for prefactor, ind in idxarr
                    ]
                    d_el_energy = np.real(overall_factor * np.sum(deriv_pfarr)) * np.exp(norm_mod - lognorm_default)

                    # Summand with derivative of norms
                    trace_def = grad_over_norm_vec[layerind, uc_ind, symbol_ind]
                    # TODO: use the system compute_grad_over_norm
                    trace_mod = compute_grad_over_norm_numpy(
                        gamma_in_sys_mod,
                        diff_d_inv_gamma_inv,
                        d_mat_d,
                        mat_d_mod_inv_vec[layerind],
                    )
                    # This is the second contribution of the elctric energy gradient F_{el} (\tilde(v) - v)
                    d_el_energy += el_energy_vec[layerind] * (trace_mod - trace_def)
                    # Scale to system size
                    d_el_energy *= nlinks
                    dest_grad[layerind, uc_ind, symbol_ind] = d_el_energy

    dest_grad = np.asarray(dest_grad)

    # We have to weigh the different layers with the electric energy operator expectation of the other layers.
    # They act as a prefactor in the derivative
    if nlayer > 1:
        for i in range(nlayer):
            prod_other_layers = ggpeps.utils.multiply_except(el_energy_vec, i)
            dest_grad[i] *= prod_other_layers

    return dest_grad


class BackendNumpy_Z2(BackendBase):
    """Backend for Z2 systems using numpy."""

    backend_type = "numpy"
    gauge_group = "Z2"

    def __init__(self) -> None:
        pass

    @staticmethod
    def calculate_lognormvec(gamma_in_sys_vec, mat_d_vec, all_factors=False):
        return calculate_lognormvec_numpy(gamma_in_sys_vec, mat_d_vec, all_factors=all_factors)

    @staticmethod
    def compute_el_grad_vec(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        unitcell_size: int,
        nvirtmodes_link: int,
        nphysmodes_site: int,
        symbolvec: tuple,
        overall_factors,
        idxarr_vec,
        el_energy_vec,
        mat_b_mod_vec,
        gamma_in_sys_mod_vec,
        covmat_out_virt_vec,
        norm_mod_vec,
        lognorm_default_vec,
        wi_gamma_in_mod_inv_vec,
        wi_gamma_out_mod_inv_vec,
        mat_d_mod_inv_vec,
        gamma_maj_sys_deriv_layvec_ucvec_symbvec,
        grad_over_norm_vec,
        zeroed_params,
    ):
        return compute_el_grad_vec_numpy(
            lattice_size,
            num_pg_layer,
            num_fermionic_layer,
            unitcell_size,
            nvirtmodes_link,
            nphysmodes_site,
            symbolvec,
            overall_factors,
            idxarr_vec,
            el_energy_vec,
            mat_b_mod_vec,
            gamma_in_sys_mod_vec,
            covmat_out_virt_vec,
            norm_mod_vec,
            lognorm_default_vec,
            wi_gamma_in_mod_inv_vec,
            wi_gamma_out_mod_inv_vec,
            mat_d_mod_inv_vec,
            gamma_maj_sys_deriv_layvec_ucvec_symbvec,
            grad_over_norm_vec,
            zeroed_params,
        )
