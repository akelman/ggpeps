import logging

from ggpeps import xnp as xnp
from ggpeps import xscipy as xscipy

import ggpeps
from ggpeps import utils
from ggpeps.lattice import Direction
from ggpeps.system.backend import backend

from .config_base import IdxGroup, CoeffsVec, ConstantsVec

from .system_base import System2DBase
from .system_base import maybe_jit

logger = logging.getLogger(ggpeps.LOGGER_NAME)


###################### Z2System2D ##########################


class Z2System2D(System2DBase):
    """2D Z2 system GGPEPS ansatz with physical fermions.

    Some general notes about conventions:

    Order of the paramvec: see the functions that create the symbolvec in the configs.
    We split the real and the imaginary part of the parameters into independent variables.
    Mode order of tmat:
        {p,l1,r1,d1,u1,l2,r2,d2,u2,l3,r3... and so on}
    Mode order of gamma_dirac:
        {p,l1,r1,d1,u1,l2,r2,d2,u2,p_dag,l1_dag,r1_dag,u1_dag,d1_dag,l2_dat,r2_dag,u2_dag,d2_dag,l3,r3...and so on}
    Mode order of gamma_maj:
        {p_1,p_2,l1_1,l1_2,r1_1,r1_2,d1_1,d1_2,u1_1,u1_2,l2_1,l2_2,r2_1,r2_2,d2_1,d2_2,u2_1,u2_2,l3_1,l3_2...and so on}
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    ################## Gauging ##################
    @classmethod
    @maybe_jit(static_argnames=["cls", "ncopy"])
    def generate_rotmat(cls, ncopy: int, group_element: xnp.ndarray, coord: tuple, dir: Direction) -> xnp.ndarray:
        """Generate the matrix to rotate gamma_in_neutral according to a given gauge field value.

        The mode order is (as for gamma_in_neutral):
            1 copy: {l_1, l_2, r_1, r_2}/{d_1, d_2, u_1, u_2},
            2 copies: {l1_1, l1_2, r1_1, r1_2, l2_1, l2_2, r2_1, r2_2}/{d1_1, d1_2, u1_1, u1_2,d2_1, d2_2, u2_1, u2_2},
        depending on whether the link is vertical or horizontal.
        The naming convention here is <mode letter><number of copy>_<majorana mode>.
        We order first by link and then by copy.

        For this system, the rotmat does not depend on the coord or dir.

        This method overwrites an abstract method in System2DBase.
        See this method in System2DBase for further documentation.
        """
        theta = xnp.angle(group_element[0][0])  # equivalent to gaugemgr.get_angle(group_element)

        # We are only rotating the right modes.
        # Thus, we leave an identity matrix for the left modes.
        rot_right = xnp.array([[xnp.cos(theta), xnp.sin(theta)], [-xnp.sin(theta), xnp.cos(theta)]])
        # We have only one left mode => 2 Majorana modes
        rot_left = xnp.eye(2)
        # The mode order is lr (horizontally) or du (vertically).
        # We rotate the different copies in the SAME way.
        dest = xscipy.linalg.block_diag(rot_left, rot_right)
        rotmat = xnp.kron(xnp.eye(ncopy), dest)
        return rotmat

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

        # The mod family (gamma_in_sys_mod + mod trackers) is measurement-only; skip it during warmup.
        if not self.defer_mod_trackers:
            self._gamma_in_sys_mod_vec = self._patch_gamma_in_sys_mod(self._gamma_in_sys_mod_vec, link_ind)

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
        # Largest entry of any updated inverse. The public update_gauge_ind wrapper uses it to trigger
        # an out-of-schedule from-scratch refresh.
        inv_mags = [xnp.max(xnp.abs(self._wi_gamma_in_vec)), xnp.max(xnp.abs(self._wi_gamma_out_vec))]

        # --- Incrementally update the modified (open-link) trackers. The link excluded from the
        # modified objects is skipped; for the others the local update is shifted by the carved-out
        # link when it sits below the changed link. The vectorized index update supports neither
        # skipping a link nor a variable offset, so we loop explicitly.
        # Skipped entirely during warmup (defer_mod_trackers): these are measurement-only and are
        # recomputed from scratch (recompute_mod_trackers) when warmup ends.
        if not self.defer_mod_trackers:
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

    ################## Observables ##################
    def _compute_mag_energy_op(self, use_trans_inv: bool = False) -> float:
        if use_trans_inv:
            if self.cfg.unitcell_size > 1:
                raise ValueError("Cannot rely on translation invariance if unitcell size is >1.")

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
            # each group element applies its own coefficient dot product. For Z2 there is only 1
            # stored group element anyway.
            coeffs_group_element = coeffs_vec[group_element_idx]
            for layerind in range(nlayer):
                layer_coeffs = coeffs_group_element[layerind]  # tuple of tuples of coeffs
                norm_mod_linkvec = norm_mod_vec[layerind]

                # Iterate over the links
                for link_pos, norm_mod in enumerate(norm_mod_linkvec):
                    ###################### Calculation of <P+P^dagger> ########################
                    link_coeffs = layer_coeffs[link_pos]
                    pf_tot: complex = constants_vec[group_element_idx][layerind][link_pos]
                    # this is the constant term in the sum, which does not come with a Pfaffian,
                    # for Z2 it should be 0
                    for size_ind, size_term in enumerate(link_coeffs):
                        array_size_term = xnp.asarray(size_term)
                        current_pfaffians = el_pfaffians[layerind, link_pos, size_ind, : len(size_term)]
                        pf_tot += xnp.dot(array_size_term, current_pfaffians)

                    # Keep el_energy_link COMPLEX. The real part is taken only after the product over
                    # layers and the sum over group elements (prod(Re) != Re(prod)).
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
        covmat_out_mod_vec: xnp.ndarray,
        el_pfaffians: xnp.ndarray,
        norm_mod_vec: xnp.ndarray,
        lognorm_default_vec: xnp.ndarray,
        gamma_out_mod_inv_vec: xnp.ndarray,
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
        dest_grad = xnp.zeros(grad_shape, dtype=complex)  # real part taken after the layer product

        nlinks = 2 * lattice_size  # valid for 2D with periodic boundary conditions
        k = 2 * nvirtmodes_link  # single link offset
        lognorm_default = xnp.sum(lognorm_default_vec)

        # Calculate the derivatives (wrt all non-zero parameters) of the modified covmat_out
        shape = (nlayer, len(mod_link_inds), unitcell_size, len(symbolvec), k, k)
        d_covmat_out_virt_vec = xnp.zeros(shape)

        l, m, u, s = inds
        # (nlayer, nmodlinks, mod_virt_dim, mod_virt_dim)
        # prod = mat_d_mod_inv @ wi_gamma_in_mod @ gamma_in_sys_mod = -(Dmod + gamma_in_mod)^-1:
        # gamma_in_sys_mod is a pure-state covariance (Gamma^2 = -1), so
        # (1 - Gamma D)^-1 Gamma = -(D + Gamma)^-1, which is the tracked wi_gamma_out_mod.
        prod_mod_norm_vec = -gamma_out_mod_inv_vec
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
        # per-active-param loop over utils.trace_of_product((d_mat_d_vec[idx], prod_mod_norm_vec[l,m]))
        # (a full O(D^2) einsum per param),
        # we sum only over the nonzero entries of d_mat_d: Tr(dD_a @ P_a) = sum_k vals[a,k] * P_a[jj,ii],
        # with P_a = prod_mod_norm_vec[l[a], m[a]]. The (ii, jj, vals) are precomputed once per eval
        # (dmatd_trace_sparse). Gathers only the nonzeros -> ~2.7x faster numpy, ~11x jax (vs the loop).
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
        # For Z2 there is only 1 stored group element anyway.
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

                    # In previous versions of the code, Pfaffians with complex/imaginary coefficients
                    # were included, but dropped here. Since operators of interest (electric energy + grad)
                    # are Hermitian, we can just take the real part here.
                    # At present, we drop these complex/imaginary terms higher in the stack to save on
                    # computation. We leave the xnp.real() for testing purposes.
                    # Keep COMPLEX; the real part is taken after the layer product + group sum below.
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
    @maybe_jit(static_argnames=["use_trans_inv"])
    def _compute_mass_energy_op_vec(
        occupations_after_ph: xnp.ndarray,
        use_trans_inv: bool = False,
    ) -> xnp.ndarray:

        if use_trans_inv:
            logger.warning(
                "Translation invariance need not be assumed for the mass energy operator."
                "Doing so gives negligible speedup, so we proceed without it."
            )

        mass_energy_op = xnp.sum(occupations_after_ph, axis=1)  # sum over sites

        return mass_energy_op

    @staticmethod
    @maybe_jit(
        static_argnames=[
            "lattice_size",
            "symbolvec",
            "unitcell_size",
            "use_trans_inv",
            "num_pg_layer",
            "num_fermionic_layer",
            "zeroed_params",
        ],
    )
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

        if not use_trans_inv:
            raise NotImplementedError("Translation invariance must be set to True.")

        nlayer = num_pg_layer + num_fermionic_layer
        param_shape = (nlayer, unitcell_size, len(symbolvec))
        gradients = xnp.zeros(param_shape)

        for layer_ind in range(num_pg_layer, nlayer):
            # only the fermionic layers directly contribute to the mass

            for site_ind in range(0, 2 * lattice_size, 2):

                for uc_ind in range(unitcell_size):
                    for symbol_ind, symbol in enumerate(symbolvec):
                        # the derivative calculation is relatively compuationally expensive
                        # (though less than for electric energy)
                        # we can skip it for parameters that are forced by the ansatz to be zero
                        if (layer_ind, uc_ind, symbol_ind) not in zeroed_params:

                            d_gamma_out = d_gamma_out_symbolvec[layer_ind, uc_ind, symbol_ind]
                            grad = 0.5 * d_gamma_out[site_ind + 1, site_ind]
                            gradients = backend.array_add(gradients, (layer_ind, uc_ind, symbol_ind), grad)

                    # further terms of the derivative are included higher up in the computation stack
                    # because computing them requires knowing various expectation values, which are not available here

        return gradients

    @staticmethod
    @maybe_jit(
        static_argnames=[
            "lattice_size",
            "num_pg_layer",
            "num_fermionic_layer",
            "horizontal_neighbor_data",
            "vertical_neighbor_data",
        ],
    )
    def _compute_int_energy_op_vec(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        gaugefieldvec: xnp.ndarray,
        ferm_covmat_vec: xnp.ndarray,
        horizontal_neighbor_data: tuple,
        vertical_neighbor_data: tuple,
    ) -> xnp.ndarray:
        """
        Note: this function assumes that U = U^dagger, which is only valid for Z2.
        For other groups, the calculation will not be as simple.
        """

        nlayer = num_pg_layer + num_fermionic_layer
        int_energy_op = xnp.zeros(nlayer)

        for layer_ind in range(num_pg_layer, nlayer):
            layer_int_energy = 0.0
            covmat = ferm_covmat_vec[layer_ind]

            for site_ind in range(lattice_size):
                site_ind_cov = 2 * site_ind  # index into covariance matrix, factor of 2 for Majorana modes per site

                # Horizontal link
                hor_link_ind = horizontal_neighbor_data[site_ind][0]
                neighborX_ind = 2 * horizontal_neighbor_data[site_ind][1]  # 2 * index of neighboring site

                gaugefield_hor = gaugefieldvec[hor_link_ind]  # a matrix representation of the group element
                cos_factor_hor = xnp.real(gaugefield_hor[0][0]).astype(
                    float
                )  # get U from gauge representation, this handles cosine
                hor_energy = 0.5 * (covmat[site_ind_cov, neighborX_ind] - covmat[site_ind_cov + 1, neighborX_ind + 1])
                layer_int_energy += hor_energy * cos_factor_hor

                # Vertical link
                vert_link_ind = vertical_neighbor_data[site_ind][0]
                neighborY_ind = 2 * vertical_neighbor_data[site_ind][1]

                gaugefield_vert = gaugefieldvec[vert_link_ind]
                cos_factor_vert = xnp.real(gaugefield_vert[0][0]).astype(float)
                vert_energy = 0.5 * (covmat[site_ind_cov, neighborY_ind + 1] + covmat[site_ind_cov + 1, neighborY_ind])
                layer_int_energy -= vert_energy * cos_factor_vert

            int_energy_op = backend.array_assign(int_energy_op, layer_ind, layer_int_energy)

        return int_energy_op

    @staticmethod
    @maybe_jit(
        static_argnames=[
            "lattice_size",
            "num_pg_layer",
            "num_fermionic_layer",
            "unitcell_size",
            "nparams",
            "horizontal_neighbor_data",
            "vertical_neighbor_data",
            "zeroed_params",
        ],
    )
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
        """
        Note: this function assumes that U = U^dagger, which is only valid for Z2.
        For other groups, the calculation will not be as simple.
        """

        nlayer = num_pg_layer + num_fermionic_layer
        param_shape = (nlayer, unitcell_size, nparams)
        gradients = xnp.zeros(param_shape)

        for layer_ind in range(num_pg_layer, nlayer):

            for site_ind in range(lattice_size):
                site_ind_cov = 2 * site_ind  # index into covariance matrix, factor of 2 for Majorana modes per site

                # Horizontal link
                hor_link_ind = horizontal_neighbor_data[site_ind][0]
                neighborX_ind = 2 * horizontal_neighbor_data[site_ind][1]  # 2 * index of neighboring site

                gaugefield_hor = gaugefieldvec[hor_link_ind]  # a matrix representation of the group element
                cos_factor_hor = xnp.real(gaugefield_hor[0][0]).astype(
                    float
                )  # get U from gauge representation, this handles cosine

                # Vertical link
                vert_link_ind = vertical_neighbor_data[site_ind][0]
                neighborY_ind = 2 * vertical_neighbor_data[site_ind][1]

                gaugefield_vert = gaugefieldvec[vert_link_ind]
                cos_factor_vert = xnp.real(gaugefield_vert[0][0]).astype(float)

                # Calculate derivatives
                for uc_ind in range(unitcell_size):
                    for symbol_ind in range(nparams):
                        # the derivative calculation is relatively compuationally expensive
                        # (though less than for electric energy)
                        # we can skip it for parameters that are forced by the ansatz to be zero
                        if (layer_ind, uc_ind, symbol_ind) not in zeroed_params:

                            d_gamma_out = d_gamma_out_symbolvec[layer_ind, uc_ind, symbol_ind]
                            grad = (
                                0.5
                                * cos_factor_hor
                                * (
                                    d_gamma_out[site_ind_cov, neighborX_ind]
                                    - d_gamma_out[site_ind_cov + 1, neighborX_ind + 1]
                                )
                            )
                            grad += (
                                -0.5
                                * cos_factor_vert
                                * (
                                    d_gamma_out[site_ind_cov, neighborY_ind + 1]
                                    + d_gamma_out[site_ind_cov + 1, neighborY_ind]
                                )
                            )
                            gradients = backend.array_add(gradients, (layer_ind, uc_ind, symbol_ind), grad)

        return gradients

    @staticmethod
    @maybe_jit(static_argnames=[])
    def _compute_chem_energy_op_vec(
        occupations_before_ph: xnp.ndarray,
    ) -> xnp.ndarray:

        chem_energy_op = xnp.sum(occupations_before_ph, axis=1)

        return chem_energy_op

    @staticmethod
    @maybe_jit(
        static_argnames=[
            "lattice_size",
            "num_pg_layer",
            "num_fermionic_layer",
            "unitcell_size",
            "symbolvec",
            "sublattice_factors",
            "zeroed_params",
        ],
    )
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

        for layer_ind in range(num_pg_layer, nlayer):
            # only the fermionic layers directly contribute to the chemical potential

            # Calculate chem term
            # Since we set the system to have different parameters on the even and odd sites when using a non-zero
            # chemical potential (i.e. the system is translationally invariant by two sites),
            # we could just calculate it for one even and one odd site and multiply by the size of the system
            for site in range(lattice_size):
                site_ind = 2 * site  # index into covariance matrix
                site_factor = sublattice_factors[site]  # even or odd sublattice

                for uc_ind in range(unitcell_size):
                    for symbol_ind, symbol in enumerate(symbolvec):
                        # the derivative calculation is relatively compuationally expensive
                        # (though less than for electric energy)
                        # we can skip it for parameters that are forced by the ansatz to be zero
                        if (layer_ind, uc_ind, symbol_ind) not in zeroed_params:

                            d_gamma_out = d_gamma_out_vec[layer_ind, uc_ind, symbol_ind]
                            grad = 0.5 * site_factor * d_gamma_out[site_ind + 1, site_ind]
                            gradients = backend.array_add(gradients, (layer_ind, uc_ind, symbol_ind), grad)

                    # further terms of the derivative are included higher up in the computation stack
                    # because computing them requires knowing various expectation values, which are not available here

        return gradients

    def _meson_string_vec(self, path: tuple[tuple[int, bool], ...]) -> xnp.ndarray:

        meson_op_vec = xnp.zeros(self.cfg.nlayer)

        # value of the fields
        path_factor = self.compute_path(path)

        # indices into the covariance matrices at the start and end of the path
        # TODO: it is a waste to calculate this for every gauge config - instead, this function should accept
        #       as input the start and end site indices
        start_site_ind, end_site_ind = self.cfg.lattice.get_path_endpoints(path)
        site_ind_cov_in = 2 * start_site_ind
        site_ind_cov_fin = 2 * end_site_ind

        for layer_ind in range(self.cfg.num_pg_layer, self.cfg.nlayer):
            covmat = self.ferm_covmat_vec[layer_ind]

            # Since for the L-shaped strings considered here the endpoints are always on the same sublattice,
            # we still have \psi^\dagger \psi after the PH transformation
            layer_val = (
                0.25
                * path_factor
                * (
                    -1j * covmat[site_ind_cov_in, site_ind_cov_fin]
                    - 1j * covmat[site_ind_cov_in + 1, site_ind_cov_fin + 1]
                    + covmat[site_ind_cov_in + 1, site_ind_cov_fin]
                    - covmat[site_ind_cov_in, site_ind_cov_fin + 1]
                )
            )

            # TODO: is the absolute value necessary? why?
            meson_op_vec = backend.array_assign(meson_op_vec, layer_ind, xnp.abs(layer_val))
        return meson_op_vec

    @staticmethod
    @maybe_jit(static_argnames=["after_ph"])
    def occupation(covmat: xnp.ndarray, site: int, site_coord: tuple[int, int], after_ph: bool = False) -> float:

        site_ind = 2 * site  # index into covariance matrix

        x, y = site_coord
        if after_ph:
            site_factor = 1
        else:
            site_factor = (-1) ** (x + y)  # even or odd sublattice

        mass_site = 0.5 * (1 + site_factor * covmat[site_ind + 1, site_ind])

        return mass_site
