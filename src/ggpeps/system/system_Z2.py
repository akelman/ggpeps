import logging
from operator import index

import numpy as np
from ggpeps import xnp as xnp
from ggpeps import xscipy as xscipy

import ggpeps
from ggpeps import utils
from ggpeps.lattice import Direction
from ggpeps.system.backend import backend

from .config_base import IdxArrVec

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
            # TODO: should not modify "private" variable - make a setter?
            """
            equivalent to:
                self._gamma_in_sys_vec[layer][
                    ind_mat : ind_mat + rotmat.shape[0],
                    ind_mat : ind_mat + rotmat.shape[1],
                ] = gamma_in_subst
            """

        # Update the determinant
        mat_inv_vec = [wi_gamma_in.inv() for wi_gamma_in in self.wi_gamma_in_vec]
        detval_vec = np.array(
            [
                incdet.update_index(mat_inv, update, ind_mat, ind_mat)
                for mat_inv, update, incdet in zip(mat_inv_vec, update_vec, self.incdet_vec)
            ]
        )

        # Update the weight
        self.weight = 0.5 * np.sum(detval_vec)

        # Update the matrix inversion
        for wi_gamma_in, update in zip(self.wi_gamma_in_vec, update_vec):
            wi_gamma_in.update_index(update, ind_mat, ind_mat)
        for wi_gamma_out, update in zip(self.wi_gamma_out_vec, update_vec):
            wi_gamma_out.update_index(update, ind_mat, ind_mat)

        # Update the modified determinant & matrices
        for lay in range(self.cfg.nlayer):
            for ind, mod_link_ind in enumerate(self.cfg.mod_link_inds):
                if mod_link_ind != link_ind:
                    # We do not update if the link is the one that is excluded in the modified objects

                    offset = 0  # no offset if link_ind < mod_link_ind
                    if link_ind > mod_link_ind:
                        offset = 2 * self.cfg.nvirtmodes_link

                    mat_inv = self.wi_gamma_in_mod_vec[lay][ind].inv()
                    self.incdet_mod_vec[lay][ind].update_index(
                        mat_inv, update_vec[lay], ind_mat - offset, ind_mat - offset
                    )

                    self.wi_gamma_in_mod_vec[lay][ind].update_index(
                        update_vec[lay], ind_mat - offset, ind_mat - offset
                    )

                    self.wi_gamma_out_mod_vec[lay][ind].update_index(
                        update_vec[lay], ind_mat - offset, ind_mat - offset
                    )

        # Invalidate gauge dependent quantities
        self.invalidate_gauge_update()

    ################## Observables ##################
    def _compute_mag_energy_op(self, use_trans_inv: bool = True):
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
    @maybe_jit(static_argnames=["idxarrs", "nlayer", "mod_link_inds", "nlinks"])
    def _compute_el_energy_op_vec(
        lognormvec_default: xnp.ndarray,
        idxarrs: IdxArrVec,
        mod_link_inds: tuple[int, ...],
        nlinks: int,
        nlayer: int,
        el_pfaffians: xnp.ndarray,
        norm_mod_vec: xnp.ndarray,
    ) -> xnp.ndarray:

        lognorm_default = xnp.sum(lognormvec_default)

        num_el_links = len(mod_link_inds)  # number of links on which the electric energy is computed
        dest = xnp.zeros((nlayer, num_el_links))

        # TODO: vectorize!
        for layerind in range(nlayer):
            layer_pairs = idxarrs[layerind]  # tuple of pairs: ((H,V), (H,V), ...)
            norm_mod_linkvec = norm_mod_vec[layerind]

            # Iterate over the links
            for link_pos, norm_mod in enumerate(norm_mod_linkvec):
                ###################### Calculation of <P> ########################

                # The matrix elements yield only the real part of <P>
                # If we use the log formulation, we can calculate the log of single terms.

                # Instead of writing down all the terms explicitly, we build tuples of the prefactors
                # and the indices of the covariance matrix.

                is_vertical = mod_link_inds[link_pos] >= (nlinks // 2)

                pf_tot = 0.0
                for term_ind, (term_h, term_v) in enumerate(layer_pairs):
                    # each term_* is (prefactor, indices); pfaffians already computed per term_ind
                    prefactor = term_v[0] if is_vertical else term_h[0]
                    pfaval = el_pfaffians[layerind, link_pos, term_ind]
                    pf_tot += prefactor * pfaval

                el_energy_link = xnp.real(pf_tot) * xnp.exp(norm_mod - lognorm_default)
                dest = backend.array_assign(dest, (layerind, link_pos), el_energy_link)

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
            "zeroed_params",
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
        idxarr_vec: IdxArrVec,
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
        gamma_maj_sys_deriv_layvec_ucvec_symbvec: xnp.ndarray,
        grad_over_norm_vec: xnp.ndarray,
        zeroed_params: tuple,
    ) -> xnp.ndarray:

        nlayer = num_pg_layer + num_fermionic_layer
        shape = (nlayer, len(mod_link_inds), unitcell_size, len(symbolvec))
        dest_grad = xnp.zeros(shape)

        nlinks = 2 * lattice_size  # valid for 2D with periodic boundary conditions
        single_link_offset = 2 * nvirtmodes_link
        lognorm_default = xnp.sum(lognorm_default_vec)

        for layerind in range(nlayer):

            # Abbreviations for more readable code
            layer_pairs = idxarr_vec[layerind]  # tuple of pairs: ((H,V), (H,V), ...)

            for link_pos, mod_link_ind in enumerate(mod_link_inds):
                mat_b = mat_b_mod_vec[layerind][link_pos]
                diff_d_gamma_inv = gamma_out_mod_inv_vec[layerind][link_pos]
                gamma_in_sys_mod = gamma_in_sys_mod_vec[layerind][link_pos]
                diff_d_inv_gamma_inv = gamma_in_mod_inv_vec[layerind][link_pos]

                covmat_out_virt = covmat_out_mod_vec[layerind][link_pos]
                norm_mod = norm_mod_vec[layerind][link_pos]
                mat_d_mod_inv = mat_d_mod_inv_vec[layerind][link_pos]

                ###################### Calculation of the derivative ########################

                # choose H/V per term based on link direction
                is_vertical = mod_link_ind >= (nlinks // 2)

                for uc_ind in range(unitcell_size):
                    for symbol_ind, _ in enumerate(symbolvec):
                        if (layerind, uc_ind, symbol_ind) not in zeroed_params:
                            # the derivative calculation is computationally expensive
                            # we can skip it for parameters that are forced by the ansatz to be zero

                            deriv_gamma_maj_sys = gamma_maj_sys_deriv_layvec_ucvec_symbvec[
                                layerind, uc_ind, symbol_ind
                            ]
                            mod_covmats = utils.extract_mod_covmats(
                                deriv_gamma_maj_sys, (mod_link_ind,), lattice_size, nphysmodes_site, nvirtmodes_link
                            )
                            d_mat_a, d_mat_b, d_mat_d = mod_covmats[0][0], mod_covmats[1][0], mod_covmats[2][0]
                            d_gamma_out = (
                                d_mat_a
                                + d_mat_b @ diff_d_gamma_inv @ xnp.transpose(mat_b)
                                + mat_b @ diff_d_gamma_inv @ xnp.transpose(d_mat_b)
                                - mat_b @ diff_d_gamma_inv @ d_mat_d @ diff_d_gamma_inv @ xnp.transpose(mat_b)
                            )
                            # The virtual mode is the last link on the bottom right of the covariance matrix
                            d_covmat_out_virt = d_gamma_out[-single_link_offset:, -single_link_offset:]
                            # Summand with derivative of the covariance matrix

                            deriv_pf_tot = 0.0
                            # for ind, (prefactor, inds) in enumerate(idxarr):
                            for term_ind, (term_h, term_v) in enumerate(layer_pairs):
                                prefactor, inds = term_v if is_vertical else term_h
                                inds_arr = xnp.asarray(inds)
                                deriv_pf_tot += prefactor * utils.derivative_pfaffian(
                                    covmat_out_virt[xnp.ix_(inds_arr, inds_arr)],
                                    d_covmat_out_virt[xnp.ix_(inds_arr, inds_arr)],
                                    el_pfaffians[layerind, link_pos, term_ind],
                                )

                            d_el_energy = xnp.real(deriv_pf_tot) * xnp.exp(norm_mod - lognorm_default)

                            # Summand with derivative of norms
                            trace_def = grad_over_norm_vec[layerind, uc_ind, symbol_ind]
                            trace_mod = utils.compute_grad_over_norm(
                                gamma_in_sys_mod,
                                diff_d_inv_gamma_inv,
                                d_mat_d,
                                mat_d_mod_inv,
                            )
                            # This is the second contribution of the elctric energy gradient F_{el} (\tilde(v) - v)
                            d_el_energy += el_energy_vec[layerind][link_pos] * (trace_mod - trace_def)

                            dest_grad = backend.array_add(
                                dest_grad, (layerind, link_pos, uc_ind, symbol_ind), d_el_energy
                            )

        # scale to system size - currently only valid when all links should be weighed equally
        dest_grad *= nlinks / len(mod_link_inds)

        # We have to weigh the different layers with the electric energy operator expectation of the other layers.
        # They act as a prefactor in the derivative.
        # This must be done separately over all links.
        if nlayer > 1:
            for lay in range(nlayer):
                for linkind in range(len(mod_link_inds)):
                    prod_other_layers = utils.multiply_except(el_energy_vec[:, linkind], lay)
                    dest_grad = backend.array_mult(dest_grad, (lay, linkind), prod_other_layers)
        dest_grad = xnp.sum(dest_grad, axis=1)  # sum over the links

        return dest_grad

    @staticmethod
    @maybe_jit(static_argnames=["lattice_size", "use_trans_inv", "num_pg_layer", "num_fermionic_layer"])
    def _compute_mass_energy_op_vec(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        ferm_cov_vec: xnp.ndarray,
        use_trans_inv: bool = True,
    ) -> xnp.ndarray:

        if not use_trans_inv:
            raise NotImplementedError("Translation invariance must be set to True.")

        nlayer = num_pg_layer + num_fermionic_layer
        mass_energy_op = xnp.zeros(nlayer)

        for layer_ind in range(num_pg_layer, nlayer):
            # only the fermionic layers directly contribute to the mass

            covmat = ferm_cov_vec[layer_ind]
            layer_mass_energy = 0.0

            # Calculate mass term
            # Since the system is translationally invariant, we could just calculate it
            # for one site and multiply by nsites instead
            for site_ind in range(0, 2 * lattice_size, 2):
                layer_mass_energy += 0.5 * (1 + covmat[site_ind + 1, site_ind])

            mass_energy_op = backend.array_assign(mass_energy_op, layer_ind, layer_mass_energy)

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
    @staticmethod
    @maybe_jit(
        static_argnames=[
            "lattice_size",
            "num_pg_layer",
            "num_fermionic_layer",
            "sublattice_factors",
        ],
    )
    def _compute_chem_energy_op_vec(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        sublattice_factors: tuple,
        ferm_covmat_vec: xnp.ndarray,
    ) -> xnp.ndarray:

        nlayer = num_pg_layer + num_fermionic_layer
        chem_energy_op = xnp.zeros(nlayer)

        for layer_ind in range(num_pg_layer, nlayer):
            # only the fermionic layers directly contribute to the chemical potential

            # Calculation prelimaries
            covmat = ferm_covmat_vec[layer_ind]
            layer_chem_energy = 0.0

            # Calculate chem term
            # Since we set the system to have different parameters on the even and odd sites when using a non-zero
            # chemical potential (i.e. the system is translationally invariant by two sites),
            # we could just calculate it for one even and one odd site and multiply by the size of the system
            for site in range(lattice_size):
                site_ind = 2 * site  # index into covariance matrix
                site_factor = sublattice_factors[site]  # even or odd sublattice
                mass_site = 0.5 * (1 + covmat[site_ind + 1, site_ind])
                layer_chem_energy += site_factor * mass_site
                layer_chem_energy += 0.5  # constant offset which arises from particle-hole transformation

            chem_energy_op = backend.array_assign(chem_energy_op, layer_ind, layer_chem_energy)

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

    def _meson_string_vec(self, path: list[tuple[int, bool]]) -> xnp.ndarray:

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

    def occupation(self, lay: int, site: int, after_ph: bool = False) -> float:

        covmat = self.ferm_covmat_vec[lay]
        site_ind = 2 * site  # index into covariance matrix

        x, y = self.cfg.lattice.ind2coord(site)
        site_factor = (-1) ** (x + y)  # even or odd sublattice
        site_even = True if site_factor == 1 else False

        if site_even or after_ph:
            mass_site = 0.5 * (1 + covmat[site_ind + 1, site_ind])
        else:
            mass_site = 0.5 * (1 - covmat[site_ind + 1, site_ind])

        return mass_site
