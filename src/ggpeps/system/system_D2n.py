import logging

import numpy as np
from ggpeps import xnp as xnp
from ggpeps import xscipy as xscipy

import ggpeps
from ggpeps.lattice import Direction
from ggpeps.system.backend import backend
from ggpeps import modearray
from ggpeps import utils

from .system_base import System2DBase
from .config_D6_2d import D6System2D_Config
from .config_base import IdxGroup, CoeffsVec, ConstantsVec


from .system_base import maybe_jit

# from ggpeps.system.global_funcs import update_gauge_ind

logger = logging.getLogger(ggpeps.LOGGER_NAME)


###################### D2nSystem2D ##########################


class D2nSystem2D(System2DBase):
    """2D Z2 system GGPEPS ansatz with physical fermions.

    Some general notes about conventions:

    Order of the paramvec: see the functions that create the symbolvec in the configs.
        We split the real and the imaginary part of the parameters into independent variables.
    Mode order of tmat: {p,l1,r1,d1,u1,l2,r2,d2,u2,l3,r3...}.
    Mode order of gamma_dirac:
        {p,l1,r1,d1,u1,l2,r2,d2,u2,p_dag,l1_dag,r1_dag,u1_dag,d1_dag,l2_dat,r2_dag,u2_dag,d2_dag,l3,r3...}.
    Mode order of gamma_maj:
        {p_1,p_2,l1_1,l1_2,r1_1,r1_2,d1_1,d1_2,u1_1,u1_2,l2_1,l2_2,r2_1,r2_2,d2_1,d2_2,u2_1,u2_2,l3_1,l3_2...}.
    """

    def __init__(self, cfg: D6System2D_Config):
        self.cfg: D6System2D_Config
        super().__init__(cfg)

    # Gauging
    @classmethod
    def generate_rotmat(cls, ncopy: int, group_element: xnp.ndarray, coord: tuple, dir: Direction) -> xnp.ndarray:
        """Generate the matrix to rotate gamma_in_neutral according to a given gauge field value.
        We work in the convention where for majorana c_i: U_g c_i U_g^dagger = sum_{j} rotmat_{i,j} c_j
        In this convention gamma_in_neutral, is gauged with gamma_in = rotmat @ gamma_neutral rotnat^T.
        Where gamma_in is the covariance matrix of the state U_g^dagger w |Omega>.

        The mode order is (as for gamma_in_neutral):
            1 copy:
                {l_1_1, l_2_1, r_1_1, r_2_1,l_1_2,l_2_2,r_1_2,r_2_2}
                or (for vertical links)
                {d_1_1, d_2_1, u_1_1, u_2_1,d_1_2,d_2_2,u_1_2,u_2_2},
            2 copies:
                {l1_1_1,l1_2_1,r1_1_1,r1_2_1,l2_1_1,l2_2_1,r2_1_1,r2_2_1,l1_1_2,l1_2_2,r1_1_2,r1_2_2,l2_1_2,l2_2_2,r2_1_2,r2_2_2}
                or (for vertical links)
                {d1_1_1,d1_2_1,u1_1_1,u1_2_1,d2_1_1,d2_2_1,u2_1_1,u2_2_1,d1_1_2,d1_2_2,u1_1_2,u1_2_2,d2_1_2,d2_2_2,u2_1_2,u2_2_2},
        The naming convention here is <mode letter><number of copy>_<majorana mode>_<color>.
        We order first by link and then by copy then by color.

        For both fermionic and pure gauge layers, the projectors don't mix copies.
        This ensures the U(1) symmetry is obeyed for the fermionic layers, and is a convention for the pure gauge ones.

        This method overwrites an abstract method in System2DBase.
        See this method in System2DBase for further documentation.
        """
        g = group_element
        # We are only rotating the right modes.
        # Thus, we leave an identity matrix for the left modes.
        real_g = xnp.real(g)
        imag_g = xnp.imag(g)
        if xnp.sum(xnp.asarray(coord)) % 2 == 0:  # gauging is different for different sublattices
            # Note that this gauging is true only for b modes and c virtual modes
            # (in the conventions of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.110.054511).
            rot_right = xnp.block(
                # TODO: Generalize this to fermionic layers as well.
                [
                    [real_g, -imag_g],
                    [imag_g, real_g],
                ],
            )  # This is the rot_right for the mode order of {r_1_1, r_1_2,r_2_1,r_2_2}
        else:
            # Note that this gauging is true only for b modes and c virtual modes
            # (in the conventions of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.110.054511).
            # TODO: Generalizze this to fermionic layers as well.
            rot_right = xnp.block(
                [
                    [real_g, imag_g],
                    [-imag_g, real_g],
                ],
            )

        # We have dim(representation) left mode => 2*dim(representation) Majorana modes
        dim_rep = len(g)  # dimension of the representation
        rot_left = xnp.eye(2 * dim_rep)

        # The mode order is lr (horizontally) or du (vertically).
        # We rotate the different copies in the SAME way.
        dest = xscipy.linalg.block_diag(
            rot_left, rot_right
        )  # This is the rot for the mode order of {l_1_1, l_1_2,l_2_1,l_2_2, r_1_1, r_1_2,r_2_1,r_2_2}

        rotmat = xnp.kron(xnp.eye(ncopy), dest)

        # TODO: we should rather just order correctly from the start
        wrong_order = cls.get_wrong_single_link_majorana_mode_order_by_copy_then_color(ncopy)
        rep_dim = 2  # for this system, the representation dimension is always 2
        correct_order_first_color_then_copy = cls.get_single_link_majorana_mode_order(ncopy, rep_dim)
        # Generate permutation matrix to change the modes's order to
        # {l1_1_1,l1_2_1,r1_1_1,r1_2_1,l2_1_1,l2_2_1,r2_1_1,r2_2_1,l1_1_2,l1_2_2,r1_1_2,r1_2_2,l2_1_2,l2_2_2,r2_1_2,r2_2_2}.
        perm_mat = xnp.array(
            modearray.generate_permutation_matrix(
                wrong_order,
                correct_order_first_color_then_copy,
            )
        )
        rotmat = xnp.transpose(perm_mat) @ rotmat @ perm_mat

        return rotmat

    @staticmethod
    def get_wrong_single_link_majorana_mode_order_by_copy_then_color(num_copies: int) -> list:
        """Generate the link-based majorana mode order for a single link. We first order by copy and then by color.
        This is not the order we use in the code. This is just to change the generate_rotmat ordering.

        Returns:
            list: List of strings of the form <mode_letter:majorana mode>_<copy>_<color>
        """

        mode_order = []
        num_colors = 2  # always 2 for this is system, in general: self.cfg.gaugemgr.rep_dim
        # We demonstrate the order for a single horizontal link -
        for copy in range(1, num_copies + 1):
            for color in range(1, num_colors + 1):
                mode1 = ("l1", copy, color)  # majorana mode l1
                mode_order += [mode1]
            for color in range(1, num_colors + 1):
                mode2 = ("l2", copy, color)  # majorana mode l2
                mode_order += [mode2]
            for color in range(1, num_colors + 1):
                mode1 = ("r1", copy, color)
                mode_order += [mode1]
            for color in range(1, num_colors + 1):
                mode2 = ("r2", copy, color)
                mode_order += [mode2]

        # Convert to a list of strings
        # This was left as a tuple above in case there was ever any use for that format
        mode_order_str = []
        for mode in mode_order:
            mode_str = mode[0] + "_" + str(mode[1]) + "_" + str(mode[2])
            mode_order_str.append(mode_str)

        return mode_order_str

    def _update_gauge_ind(self, link_ind: int, theta: xnp.ndarray) -> None:
        """Substitute the gauge field on a single link and incrementally update the trackers.

        Exactly substitutes ``gamma_in_sys`` for the changed link, then incrementally updates both
        the closed (``wi_gamma_in/out_vec``, ``incdet_vec``) and modified (``wi_gamma_*_mod_vec``,
        ``incdet_mod_vec``) Woodbury/IncDet trackers. The ``gamma_out`` trackers invert
        ``mat_d + gamma_in``, so their Woodbury step uses the negated update. Records
        ``self._last_step_max_inv_mag`` (largest entry of any updated inverse) as the global
        near-singularity signal the public ``update_gauge_ind`` wrapper uses for the magnitude guard.

        Args:
            link_ind (int): Link index to be updated
            theta (xnp.ndarray): New gauge field value

        Returns:
            None
        """

        # Update the gaugefield
        self._gaugefieldvec = backend.array_assign(self._gaugefieldvec, link_ind, theta)

        # There are two directions per vertex
        ind_mat = 2 * self.cfg.nvirtmodes_link * link_ind
        coord, dir = self.cfg.lattice.ind2coord_dir(link_ind)
        rotmat = self.generate_rotmat(self.cfg.ncopy, theta, coord, dir)

        update_vec = []
        for layer in range(self.cfg.nlayer):
            gamma_neutral_gauge = self.gamma_gauge_neutral_vec[layer][dir]
            gamma_in_subst = rotmat @ gamma_neutral_gauge @ xnp.transpose(rotmat)
            update_vec.append(
                self.calculate_update_gamma_in(ind_mat, gamma_in_subst, gamma_in_sys=self.gamma_in_sys_vec[layer])
            )

            # Substitute in the array
            inds = (layer, slice(ind_mat, ind_mat + rotmat.shape[0]), slice(ind_mat, ind_mat + rotmat.shape[1]))
            self._gamma_in_sys_vec = backend.array_assign(self._gamma_in_sys_vec, inds, gamma_in_subst)

        update_arr = xnp.array(update_vec)

        # --- Incrementally update the closed (full-system) trackers via Woodbury / IncDet.
        # gamma_out now inverts (mat_d + gamma_in), so a +Delta change in gamma_in enters its
        # inverted matrix with the opposite sign -> the out-tracker Woodbury step uses -update.
        update_arr_out = -update_arr
        self._incdet_vec = utils.IncLogAbsDeterminant.update_index(
            self.incdet_vec, self.wi_gamma_in_vec, update_arr, ind_mat, ind_mat
        )
        self.weight = 0.5 * xnp.sum(self.incdet_vec)
        self._wi_gamma_in_vec = utils.WoodburyInverter.update_index(self.wi_gamma_in_vec, update_arr, ind_mat, ind_mat)
        self._wi_gamma_out_vec = utils.WoodburyInverter.update_index(
            self.wi_gamma_out_vec, update_arr_out, ind_mat, ind_mat
        )
        # Largest entry of any updated inverse (the GLOBAL near-singularity signal: large |inverse|
        # <=> small smallest singular value). The public update_gauge_ind wrapper uses it to trigger
        # an out-of-schedule from-scratch refresh. Accumulate the per-tracker maxes as on-device
        # scalars and defer the single device->host read to the end (one sync/step, not one each).
        inv_mags = [xnp.max(xnp.abs(self._wi_gamma_in_vec)), xnp.max(xnp.abs(self._wi_gamma_out_vec))]

        # --- Incrementally update the modified (open-link) trackers. The link excluded from the
        # modified objects is skipped; for the others the local update is shifted by the carved-out
        # link when it sits below the changed link. The vectorized index update supports neither
        # skipping a link nor a variable offset, so we loop explicitly.
        assert self._wi_gamma_in_mod_vec is not None  # for mypy
        assert self._wi_gamma_out_mod_vec is not None
        for lay in range(self.cfg.nlayer):
            for ind, mod_link_ind in enumerate(self.cfg.mod_link_inds):
                if mod_link_ind == link_ind:
                    continue
                offset = 2 * self.cfg.nvirtmodes_link if link_ind > mod_link_ind else 0
                pos = ind_mat - offset

                mat_inv = self.wi_gamma_in_mod_vec[lay][ind]
                update_out = -update_vec[lay]
                new_det = utils.IncLogAbsDeterminant.update_index(
                    self.incdet_mod_vec[lay][ind], mat_inv, update_vec[lay], pos, pos
                )
                self._incdet_mod_vec = backend.array_assign(self._incdet_mod_vec, (lay, ind), new_det)
                new_in = utils.WoodburyInverter.update_index(
                    self._wi_gamma_in_mod_vec[lay][ind], update_vec[lay], pos, pos
                )
                self._wi_gamma_in_mod_vec = backend.array_assign(self._wi_gamma_in_mod_vec, (lay, ind), new_in)
                new_out = utils.WoodburyInverter.update_index(
                    self._wi_gamma_out_mod_vec[lay][ind], update_out, pos, pos
                )
                self._wi_gamma_out_mod_vec = backend.array_assign(self._wi_gamma_out_mod_vec, (lay, ind), new_out)
                inv_mags.append(xnp.max(xnp.abs(new_in)))
                inv_mags.append(xnp.max(xnp.abs(new_out)))

        # Single device->host read for the whole step (max over all per-tracker maxes).
        self._last_step_max_inv_mag = float(xnp.max(xnp.asarray(inv_mags)))

        # Invalidate gauge dependent quantities
        self.invalidate_gauge_update()

    # Observables
    def _compute_mag_energy_op(self, use_trans_inv: bool = True) -> float:
        if use_trans_inv:
            # Evaluate one plaquette and multiply by number of plaquettes
            wilson_plaquette = self.cfg.lattice.generate_wilson_loop((0, 0), (1, 1))
            nplaq = self.cfg.lattice.nplaquettes
            mag_energy_bare = nplaq * xnp.real(self.compute_path(wilson_plaquette))
        else:
            # Evaluate every plaquette of the system
            mag_energy_bare = 0
            for x in range(self.cfg.lattice.nx):
                for y in range(self.cfg.lattice.ny):
                    wilson_plaquette = self.cfg.lattice.generate_wilson_loop((x, y), (1, 1))
                    mag_energy_bare += xnp.real(self.compute_path(wilson_plaquette))
        return mag_energy_bare

    @staticmethod
    @maybe_jit(static_argnames=["mod_link_inds", "nlayer", "coeffs_vec", "constants_vec"])
    def _compute_el_energy_op_vec(
        lognormvec_default: xnp.ndarray,
        mod_link_inds: tuple[int, ...],
        nlayer: int,
        el_pfaffians: xnp.ndarray,
        norm_mod_vec: xnp.ndarray,
        group_elements_for_el_energy: tuple[xnp.ndarray, ...],
        coeffs_vec: CoeffsVec,
        constants_vec: ConstantsVec,
    ) -> xnp.ndarray:

        lognorm_default = xnp.sum(lognormvec_default)

        num_el_links = len(mod_link_inds)  # number of links on which the electric energy is computed
        num_group_elements = len(group_elements_for_el_energy)
        dest = xnp.zeros((num_group_elements, nlayer, num_el_links), dtype=complex)

        # TODO: vectorize!
        for group_element_idx in range(num_group_elements):
            # coeffs for the specific group element, in the unique-index basis (cfg.uniq_coeffs_vec):
            # the pfaffians are computed once per unique index tuple (no group-element axis) and
            # each group element applies its own coefficient dot product.
            coeffs_group_element = coeffs_vec[group_element_idx]
            for layerind in range(nlayer):
                layer_coeffs = coeffs_group_element[layerind]  # tuple of tuples of coeffs
                norm_mod_linkvec = norm_mod_vec[layerind]

                # Iterate over the links
                for link_pos, norm_mod in enumerate(norm_mod_linkvec):
                    ###################### Calculation of sum f_j |jmn><jmn| ########################
                    link_coeffs = layer_coeffs[link_pos]
                    pf_tot = constants_vec[group_element_idx][layerind][link_pos]
                    # this is the constant term in the sum, which does not come with a Pfaffian,
                    # for Z2 it should be 0
                    for size_ind, size_term in enumerate(link_coeffs):
                        array_size_term = xnp.asarray(
                            size_term
                        )  # TODO: We should convert this to array outside the loop
                        current_pfaffians = el_pfaffians[layerind, link_pos, size_ind, : len(size_term)]
                        pf_tot += xnp.dot(array_size_term, current_pfaffians)

                    # Keep el_energy_link COMPLEX. The gauged projector is not hermitian for
                    # non-Abelian reflections, so the imaginary part is physical here; the real part
                    # is taken only after the product over layers and the sum over group elements
                    # (prod(Re) != Re(prod)). See el_energy_op in system_base.
                    el_energy_link = pf_tot * xnp.exp(norm_mod - lognorm_default)

                    dest = backend.array_assign(dest, (group_element_idx, layerind, link_pos), el_energy_link)
        return dest

    @staticmethod
    @maybe_jit(
        static_argnames=[
            "lattice_size",
            "num_pg_layer",
            "num_fermionic_layer",
            "unitcell_size",
            "nvirtmodes_link",
            "nphysmodes_site",
            "mod_link_inds",
            "symbolvec",
            "idxarr_vec",
            "coeffs_vec",
        ]
    )
    def _compute_el_grad_vec(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        unitcell_size: int,
        nvirtmodes_link: int,
        nphysmodes_site: int,
        mod_link_inds: tuple[int, ...],
        symbolvec: tuple,
        el_energy_vec: xnp.ndarray,
        mat_b_mod_vec: xnp.ndarray,
        gamma_in_sys_mod_vec: xnp.ndarray,
        covmat_out_mod_vec: xnp.ndarray,
        el_pfaffians: xnp.ndarray,
        norm_mod_vec: xnp.ndarray,
        lognorm_default_vec: xnp.ndarray,
        gamma_in_mod_inv_vec: xnp.ndarray,
        gamma_out_mod_inv_vec: xnp.ndarray,
        mat_d_mod_inv_vec: xnp.ndarray,
        d_mat_a_vec: xnp.ndarray,
        d_mat_b_vec: xnp.ndarray,
        d_mat_d_vec: xnp.ndarray,
        grad_over_norm_vec: xnp.ndarray,
        inds: tuple,
        group_elements_for_el_energy: tuple[xnp.ndarray, ...],
        idxarr_vec: IdxGroup,
        coeffs_vec: CoeffsVec,
        rotmat_vec: xnp.ndarray,
        sp_ii: xnp.ndarray,
        sp_jj: xnp.ndarray,
        sp_vals: xnp.ndarray,
    ) -> xnp.ndarray:
        """In early 2026, this function was significantly optimized.
        This was done after it was generalized in various ways over the previous months:
            compute on multiple links, horizontal and vertical links, on different sublattices, for non-Abelian groups.
        As a result, it is somewhat harder to read.
        It may be easier to read the (slower and less general) versions at (in reverse chronological order)
            commit 6cbabbd: before significant vectorization
            commit 1d63a6b: after generalization to multiple hor/vert links, but before many optimizations,
        or even earlier versions.
        """
        num_group_elements = len(group_elements_for_el_energy)

        nlayer = num_pg_layer + num_fermionic_layer
        grad_shape = (num_group_elements, nlayer, len(mod_link_inds), unitcell_size, len(symbolvec))
        # Kept COMPLEX: the gauged projector is non-hermitian for non-Abelian reflections, so the
        # real part is taken only after the product over layers (and sum over group elements).
        dest_grad = xnp.zeros(grad_shape, dtype=complex)

        nlinks = 2 * lattice_size  # valid for 2D with periodic boundary conditions
        k = 2 * nvirtmodes_link  # single link offset
        lognorm_default = xnp.sum(lognorm_default_vec)

        # Calculate the derivatives (wrt all non-zero parameters) of the modified covmat_out
        shape = (nlayer, len(mod_link_inds), unitcell_size, len(symbolvec), k, k)
        d_covmat_out_virt_vec = xnp.zeros(shape)

        l, m, u, s = inds
        # NOTE: from limited testing, it appears that masking these is not worth it, as that creates extra copies.
        # But it could be worth it if some of the matmuls were moved out of here, to happen once per eval.
        # (nlayer, nmodlinks, mod_virt_dim, mod_virt_dim)
        prod_mod_norm_vec = mat_d_mod_inv_vec @ gamma_in_mod_inv_vec @ gamma_in_sys_mod_vec
        # (nlayer, nmodlinks, mod_virt_dim, link_dim), take only the last k columns
        diff_times_b_vec = gamma_out_mod_inv_vec @ xnp.swapaxes(mat_b_mod_vec, -1, -2)[:, :, :, -k:]
        # (nlayer, nmodlinks, link_dim, mod_virt_dim), take only the last k rows
        b_times_diff_vec = mat_b_mod_vec[:, :, -k:, :] @ gamma_out_mod_inv_vec

        shape = (nlayer, len(mod_link_inds), unitcell_size, len(symbolvec), k, k)
        d_covmat_out_virt_vec = xnp.zeros(shape)

        l, m, u, s = inds

        R_active = rotmat_vec[m]
        R_active_T = xnp.swapaxes(R_active, -1, -2)

        diffB = diff_times_b_vec[l, m]
        Bdiff = b_times_diff_vec[l, m]

        # TODO: The middle part here (what's multiplied by R_active and R_active_T)
        # does not depend on the gauge configuration,
        # so it could be be computed once per eval, and not here for every gauge configuration.
        vals = (
            R_active_T
            @ (
                d_mat_a_vec[..., -k:, -k:]
                + d_mat_b_vec[..., -k:, :] @ diffB
                + Bdiff @ xnp.swapaxes(d_mat_b_vec, -1, -2)[..., :, -k:]
                - Bdiff @ d_mat_d_vec @ diffB
            )
            @ R_active
        )

        d_covmat_out_virt_vec = backend.array_assign(d_covmat_out_virt_vec, (l, m, u, s), vals)

        # Calculate the modified norms: trace(d_mat_d_a @ prod_mod_norm_a) for every active param a.
        # d_mat_d is a parameter-derivative -> config-independent and ~99% zero, so instead of the
        # dense form utils.trace_of_product((d_mat_d_vec, prod_mod_norm_vec[l, m])) - whose fancy-index
        # copy + full O(D^2) einsum dominated this function (>40% on numpy) - we sum only over the
        # nonzero entries of d_mat_d: Tr(dD_a @ P_a) = sum_k vals[a,k] * P_a[jj[a,k], ii[a,k]], with
        # P_a = prod_mod_norm_vec[l[a], m[a]]. The (ii, jj, vals) are precomputed once per eval
        # (dmatd_trace_sparse). This gathers only the nonzeros -> ~30x faster on numpy, ~7x on jax.
        norm_shape = (nlayer, len(mod_link_inds), unitcell_size, len(symbolvec))
        prod_vec = xnp.zeros(norm_shape)
        # (num_active, kmax): prod at the nonzero (row=ii, col=jj) positions; trace uses P[jj, ii].
        p_gather = prod_mod_norm_vec[l[:, None], m[:, None], sp_jj, sp_ii]
        vals = xnp.sum(sp_vals * p_gather, axis=-1)
        prod_vec = backend.array_assign(prod_vec, (l, m, u, s), vals)

        # The pfaffian derivatives depend only on (layer, link, index tuple) -- NOT on the group
        # element: the pfaffians live in the unique-index basis (cfg.uniq_idx_vec), so each
        # derivative is computed once and the per-group-element coefficients (cfg.uniq_coeffs_vec,
        # same unique basis and length for every group element) are applied as a stacked dot below.
        deriv_pf_tot_vec_vec = xnp.zeros(
            (num_group_elements, nlayer, len(mod_link_inds), unitcell_size, len(symbolvec)), dtype=complex
        )

        for layerind in range(nlayer):

            for link_pos, _ in enumerate(mod_link_inds):

                for lens_ind in range(len(idxarr_vec[layerind][link_pos])):
                    # (# pfafs, pfaf submat dim)
                    inds_arr = xnp.asarray(idxarr_vec[layerind][link_pos][lens_ind])
                    # (num_group_elements, num_pfafs) coefficients in the unique basis
                    prefactors = xnp.asarray(
                        [coeffs_vec[ge][layerind][link_pos][lens_ind] for ge in range(num_group_elements)]
                    )

                    # We slice the last dimension because the el_pfaffians array is padded with zeros.
                    pfafs = el_pfaffians[layerind, link_pos, lens_ind, : len(inds_arr)]

                    virts = covmat_out_mod_vec[layerind][link_pos][
                        None, None, inds_arr[:, :, None], inds_arr[:, None, :]
                    ]
                    d_virts = d_covmat_out_virt_vec[layerind, link_pos][
                        :, :, inds_arr[:, :, None], inds_arr[:, None, :]
                    ]

                    # (unitcell_size, len(symbolvec), num_pfafs)
                    deriv_pf_tot_vectorized = utils.derivative_pfaffian_vectorized(virts, d_virts, pfafs)
                    deriv_pf_tot_vec_vec = backend.array_add(
                        deriv_pf_tot_vec_vec,
                        (slice(None), layerind, link_pos),
                        xnp.einsum("usn,gn->gus", deriv_pf_tot_vectorized, prefactors),
                    )

        for group_element_idx in range(num_group_elements):
            for layerind in range(nlayer):
                for link_pos, _ in enumerate(mod_link_inds):

                    d_el_energy_vec = deriv_pf_tot_vec_vec[group_element_idx, layerind, link_pos] * xnp.exp(
                        norm_mod_vec[layerind][link_pos] - lognorm_default
                    )

                    # Summand with derivative of norms
                    trace_def = grad_over_norm_vec[layerind]

                    # Instead of computing the modified grad over the norm as:
                    # compute_grad_over_norm(gamma_in_sys_mod, d_mat_d, mat_d_mod_inv, diff_d_inv_gamma_inv)
                    #    = -0.5 * trace(gamma_in_sys @ deriv_d @ mat_d_inv @ diff)
                    # we have saved the product of several mats above
                    # (since they don't change in inner loops), and use it here
                    trace_mod = -0.5 * prod_vec[layerind, link_pos]

                    # This is the second contribution of the elctric energy gradient F_{el} (\tilde(v) - v)
                    d_el_energy_vec += el_energy_vec[group_element_idx][layerind][link_pos] * (trace_mod - trace_def)

                    dest_grad = backend.array_add(dest_grad, (group_element_idx, layerind, link_pos), d_el_energy_vec)

        # scale to system size - currently only valid when all links should be weighed equally
        dest_grad *= nlinks / len(mod_link_inds)

        # We have to weigh the different layers with the electric energy operator expectation of the other layers.
        # They act as a prefactor in the derivative.
        # This must be done separately over all links.
        if nlayer > 1:
            for group_idx in range(num_group_elements):
                for lay in range(nlayer):
                    for linkind in range(len(mod_link_inds)):
                        prod_other_layers = utils.multiply_except(el_energy_vec[group_idx, :, linkind], lay)
                        dest_grad = backend.array_mult(dest_grad, (group_idx, lay, linkind), prod_other_layers)
        dest_grad = xnp.sum(dest_grad, axis=0)  # sum over group elements
        dest_grad = xnp.sum(dest_grad, axis=1)  # sum over the links

        # Take the real part only now, after the layer product and the group-element sum.
        return xnp.real(dest_grad)

    @staticmethod
    def _compute_mass_energy_op_vec(
        occupations_after_ph: xnp.ndarray,
        use_trans_inv: bool = True,
    ) -> xnp.ndarray:
        mass_energy_op = xnp.sum(occupations_after_ph, axis=1)  # currently, occupations do not support color!!
        return mass_energy_op

    @staticmethod
    def _compute_mass_energy_grad(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        unitcell_size: int,
        symbolvec: tuple,
        d_gamma_out_symbolvec: xnp.ndarray,
        zeroed_params: tuple,
        use_trans_inv: bool = True,
    ) -> xnp.ndarray:
        nlayer = num_pg_layer + num_fermionic_layer
        param_shape = (nlayer, unitcell_size, len(symbolvec))

        gradients = xnp.zeros(param_shape)
        return gradients

    @staticmethod
    def _compute_int_energy_op_vec(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        gaugefieldvec: xnp.ndarray,
        ferm_covmat_vec: xnp.ndarray,
        horizontal_neighbor_data: tuple,
        vertical_neighbor_data: tuple,
    ) -> xnp.ndarray:

        int_energy_op = xnp.zeros(num_pg_layer + num_fermionic_layer)
        return int_energy_op

    @staticmethod
    def _compute_int_energy_grad(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        unitcell_size: int,
        nparams: int,
        gaugefieldvec: xnp.ndarray,
        d_gamma_out_symbolvec: xnp.ndarray,
        horizontal_neighbor_data: tuple,
        vertical_neighbor_data: tuple,
        zeroed_params: tuple,
    ) -> xnp.ndarray:
        nlayer = num_pg_layer + num_fermionic_layer
        param_shape = (nlayer, unitcell_size, nparams)
        gradients = xnp.zeros(param_shape)
        return gradients

    @staticmethod
    def _compute_chem_energy_op_vec(
        occupations_before_ph: xnp.ndarray,
    ) -> xnp.ndarray:
        chem_energy_op = xnp.sum(occupations_before_ph, axis=1)  # currently, occupations do not support color!!
        return chem_energy_op

    @staticmethod
    def _compute_chem_energy_grad(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        unitcell_size: int,
        symbolvec: tuple,
        sublattice_factors: tuple,
        zeroed_params: tuple,
        d_gamma_out_vec: xnp.ndarray,
    ) -> xnp.ndarray:
        nlayer = num_pg_layer + num_fermionic_layer
        param_shape = (nlayer, unitcell_size, len(symbolvec))
        gradients = xnp.zeros(param_shape)
        return gradients

    def _meson_string_vec(self, path: tuple[tuple[int, bool], ...]) -> xnp.ndarray:
        meson_op_vec = xnp.zeros(self.cfg.nlayer)
        return xnp.array(meson_op_vec)

    @staticmethod
    @maybe_jit(static_argnames=["after_ph"])
    def occupation(covmat: xnp.ndarray, site: int, site_coord: tuple[int, int], after_ph: bool = False) -> float:
        return 0.0
