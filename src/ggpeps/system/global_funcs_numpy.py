############## NUMPY CPU VERSIONS ##############

from typing import List
import numpy as np

import ggpeps


def calculate_lognormvec_numpy(
    gamma_in_sys_vec: List[np.ndarray],
    mat_d_vec: List[np.ndarray],
    all_factors: bool = False,
) -> float:
    # This is still the plain formula, without any update mechanism
    nlayer = len(mat_d_vec)
    dest = np.zeros(nlayer)

    for ind in range(nlayer):
        gamma_in_sys = gamma_in_sys_vec[ind]
        mat_d = mat_d_vec[ind]

        sign, logval = np.linalg.slogdet(
            (np.eye(mat_d.shape[0]) - gamma_in_sys @ mat_d)
        )

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
    dest = -0.5 * (A * B.T).sum()
    return dest


def compute_el_grad_vec_numpy(system):
    """Computation of the electric energy gradients.
    We start by calculating the electric energies, since these are needed for evaluating the gradients.
    Since several operations needed for the computation of the gradient and the energy are similar, we can reuse many intermediate steps.

    This method overwrites an abstract method in System2DBase.

    Args:
        use_trans_inv (bool, optional): Use the translationally invariant implementation. Defaults to True.

    Returns:
        list: list of gradients for the full system
    """

    dest_grad = np.zeros(system.cfg.param_shape(), dtype=np.float64)
    overall_factors = system.cfg.el_overall_factors
    idxarrs = system.cfg.idxarr_vec
    el_energy_vec = (
        system.el_energy_op_vec
    )  # this gets the electric energy, and ensures that the intermediate steps are calculated

    for layerind in range(system.cfg.nlayer):

        # Abbreviations for more readable code
        mat_b = system.mat_b_mod_vec[layerind]
        diff_d_gamma_inv = system.wi_gamma_out_mod_vec[
            layerind
        ].inv()  # this does not actually do a computation, just a retrieval
        single_link_offset = 2 * system.cfg.nvirtmodes_link
        offset = 2 * system.cfg.lattice.size + single_link_offset
        idxarr = idxarrs[layerind]
        overall_factor = overall_factors[layerind]
        nlinks = system.cfg.lattice.nlinks
        gamma_in_sys_mod = system.gamma_in_sys_mod_vec[layerind]
        diff_d_inv_gamma_inv = system.wi_gamma_in_mod_vec[layerind].inv()

        # get saved intermediate results from electric energy calculation
        intermediate = system._electric_energy_intermediate_vals
        covmat_out_virt = intermediate.covmat_out_virt_vec[layerind]
        norm_mod = intermediate.norm_mod_vec[layerind]
        lognorm_default = intermediate.lognorm_default_vec[layerind]

        ###################### Calculation of the derivative ########################
        for uc_ind in range(system.cfg.unitcell_size):
            for symbol_ind, symbol in enumerate(system.symbolvec):
                if (layerind, uc_ind, symbol_ind) in system.cfg.zeroed_params:
                    # the derivative calculation is compuationally expensive
                    # we can skip it for parameters that are forced by the ansatz to be zero
                    dest_grad[layerind, uc_ind, symbol_ind] = 0
                else:
                    deriv_gamma_maj_sys = system.gamma_maj_sys_deriv_vec(symbol)[
                        layerind, uc_ind
                    ]
                    d_mat_a, d_mat_b, d_mat_d = (
                        ggpeps.system.system_base.extract_partial_covmats(
                            deriv_gamma_maj_sys, offset
                        )
                    )
                    d_gamma_out = (
                        d_mat_a
                        + d_mat_b @ diff_d_gamma_inv @ np.transpose(mat_b)
                        + mat_b @ diff_d_gamma_inv @ np.transpose(d_mat_b)
                        - mat_b
                        @ diff_d_gamma_inv
                        @ d_mat_d
                        @ diff_d_gamma_inv
                        @ np.transpose(mat_b)
                    )
                    # The virtual mode is the last link on the bottom right of the covariance matrix
                    d_covmat_out_virt = d_gamma_out[
                        -single_link_offset:, -single_link_offset:
                    ]
                    # Summand with derivative of the covariance matrix
                    # We re-use the list comprehension from above to use the indices
                    deriv_pfarr = [
                        prefactor
                        * ggpeps.utils.derivative_pfaffian(
                            covmat_out_virt[np.ix_(ind, ind)],
                            d_covmat_out_virt[np.ix_(ind, ind)],
                        )
                        for prefactor, ind in idxarr
                    ]
                    d_el_energy = np.real(
                        overall_factor * np.sum(deriv_pfarr)
                    ) * np.exp(norm_mod - lognorm_default)

                    # Summand with derivative of norms
                    trace_def = system.compute_grad_over_norm(symbol, layerind, uc_ind)
                    trace_mod = compute_grad_over_norm_numpy(
                        gamma_in_sys_mod,
                        diff_d_inv_gamma_inv,
                        d_mat_d,
                        system.mat_d_mod_inv_vec[layerind],
                    )
                    # This is the second contribution of the elctric energy gradient F_{el} (\tilde(v) - v)
                    d_el_energy += el_energy_vec[layerind] * (trace_mod - trace_def)
                    # Scale to system size
                    d_el_energy *= nlinks
                    dest_grad[layerind, uc_ind, symbol_ind] = d_el_energy

    dest_grad = np.asarray(dest_grad)

    # We have to weigh the different layers with the electric energy operator expectation of the other layers.
    # They act as a prefactor in the derivative
    if system.cfg.nlayer > 1:
        for i in range(system.cfg.nlayer):
            prod_other_layers = ggpeps.utils.multiply_except(el_energy_vec, i)
            dest_grad[i] *= prod_other_layers

    system.cfg.enforce_parameter_conditions(dest_grad)
    return dest_grad


# def update_gauge_ind_numpy(z2_system, link_ind, theta):
#     """Update method that is called upon changing a gauge field.
#     This method is central to the algorithm since it changes the gauged projectors and updates all incremental trackers
#     of determinants and inverses.
#     The re-calculation of determinants and inverses for the norm would be prohibitively expensive.

#     This method overwrites an abstract method in System2DBase.

#     Args:
#         link_ind (int): Link index to be updated
#         theta (float): New gauge field value
#     """
#     # Update the gaugefield
#     z2_system._gaugefieldvec[link_ind] = theta
#     # There are two directions per vertex
#     ind_mat = 2 * z2_system.cfg.nvirtmodes_link * link_ind
#     coord, dir = z2_system.cfg.lattice.ind2coord_dir(link_ind)
#     rotmat = z2_system.generate_rotmat(theta, coord, dir)
#     gamma_neutral_gauge = z2_system.gamma_gauge_neutral[0][dir]
#     gamma_in_subst = rotmat @ gamma_neutral_gauge @ np.transpose(rotmat)
#     update = z2_system.calculate_update_gamma_in(ind_mat, gamma_in_subst)

#     # Update the determinant
#     mat_inv_vec = [wi_gamma_in.inv() for wi_gamma_in in z2_system.wi_gamma_in_vec]
#     detval_vec = [
#         incdet.update_index(mat_inv, update, ind_mat, ind_mat)
#         for mat_inv, incdet in zip(mat_inv_vec, z2_system.incdet_vec)
#     ]

#     # Update the modified determinant
#     offset = 2 * z2_system.cfg.nvirtmodes_link
#     if ind_mat - offset >= 0:
#         for wi, incdet in zip(z2_system.wi_gamma_in_mod_vec, z2_system.incdet_mod_vec):
#             mat_inv = wi.inv()
#             incdet.update_index(mat_inv, update, ind_mat - offset, ind_mat - offset)

#     # Update the weight
#     z2_system.weight = 0.5 * np.sum(detval_vec)

#     # Update the matrix inversion
#     for wi_gamma_in in z2_system.wi_gamma_in_vec:
#         wi_gamma_in.update_index(update, ind_mat, ind_mat)
#     for wi_gamma_out in z2_system.wi_gamma_out_vec:
#         wi_gamma_out.update_index(update, ind_mat, ind_mat)

#     if ind_mat - offset >= 0:
#         for wi_gamma_in_mod in z2_system.wi_gamma_in_mod_vec:
#             wi_gamma_in_mod.update_index(update, ind_mat - offset, ind_mat - offset)
#         for wi_gamma_out_mod in z2_system.wi_gamma_out_mod_vec:
#             wi_gamma_out_mod.update_index(update, ind_mat - offset, ind_mat - offset)

#     # Substitute in the array
#     z2_system.gamma_in_sys[ind_mat:ind_mat + rotmat.shape[0], ind_mat:ind_mat + rotmat.shape[1]] = gamma_in_subst

#     # Invalidate gauge dependent quantities
#     z2_system.invalidate_gauge_update()


def extract_partial_covmats_numpy(mat, corner):
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


def slice_matrix_numpy(mat, a, b, c, d):
    return mat[a:b, c:d]


def gamma_in_sys_mod_numpy(gamma_in_sys, single_link_offset):
    """Get function to return the gauged gamma_in_sys with a single link modification (to compute the electric energy),
    the covariance matrix of the links for the whole system.

    Returns:
        np.ndarray: Gauged, modified covariance matrix of the system
    """
    return gamma_in_sys[single_link_offset:, single_link_offset:]
