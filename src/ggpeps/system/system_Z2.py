import logging
from pfapack import pfaffian as pf

import numpy as np
from ggpeps import xnp as xnp
from ggpeps import xscipy as xscipy

import ggpeps
from ggpeps import utils
from ggpeps.lattice import Direction
from ggpeps.system.backend import backend

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
        """Constructor of a Z2System2D system, with any number of virtual fermions per site per link
        (provided a valid config is given).

        Args:
            cfg (Config2DBase): Configuration containing all system-related parameters
        """
        super().__init__(cfg)

    ################## Gauging ##################

    def generate_rotmat(self, group_element: xnp.array, coord: tuple, dir: Direction):
        """Generate the matrix to rotate gamma_in_neutral according to a given gauge field value.

        The mode order is (as for gamma_in_neutral):
            1 copy: {l_1, l_2, r_1, r_2}/{d_1, d_2, u_1, u_2},
            2 copies: {l1_1, l1_2, r1_1, r1_2, l2_1, l2_2, r2_1, r2_2}/{d1_1, d1_2, u1_1, u1_2,d2_1, d2_2, u2_1, u2_2},
        depending on whether the link is vertical or horizontal.
        The naming convention here is <mode letter><number of copy>_<majorana mode>.
        We order first by link and then by copy.

        For pure gauge layers, modes of copy one are coupled to modes of copy 2. The projectors mix copies.
        For fermionic layers, the projectors don't mix copies to ensure the U(1) symmetry is obeyed.

        The sites are picked such that the left mode is right of the right modes,
        i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        This method overwrites an abstract method in System2DBase.

        Args:
            g (fxnp.array): representation of group element
            coord (tuple): (x,y) coordinate on the lattice
            dir (lattice.Direction): direction of the link

        Returns:
            xnp.ndarray: Rotation matrix for gamma_in_neutral
        """
        theta = self.cfg.gaugemgr.get_angle(group_element)
        # Gauging might be different depending on sublattice or link direction, but for this system it is the same
        if dir == Direction.X and (-1) ** (coord[0] + coord[1]) == -1:
            pass

        # We are only rotating the right modes.
        # Thus, we leave an identity matrix for the left modes.
        rot_right = xnp.array([[xnp.cos(theta), xnp.sin(theta)], [-xnp.sin(theta), xnp.cos(theta)]])
        # We have only one left mode => 2 Majorana modes
        rot_left = xnp.eye(2)
        # The mode order is lr (horizontally) or du (vertically).
        # We rotate the different copies in the SAME way.
        dest = xscipy.linalg.block_diag(rot_left, rot_right)
        rotmat = xnp.kron(xnp.eye(self.cfg.ncopy), dest)
        return rotmat

    def update_gauge_ind(self, link_ind, theta):
        """Update method that is called upon changing a gauge field.
        This method is central to the algorithm since it changes the gauged projectors
        and updates all incremental trackers of determinants and inverses.
        The re-calculation of determinants and inverses for the norm would be
        prohibitively expensive.

        This method overwrites an abstract method in System2DBase.

        Args:
            link_ind (int): Link index to be updated
            theta (xnp.array): New gauge field value
        """
        # Update the gaugefield
        self._gaugefieldvec = backend.array_assign(self._gaugefieldvec, link_ind, theta)

        # There are two directions per vertex
        ind_mat = 2 * self.cfg.nvirtmodes_link * link_ind
        coord, dir = self.cfg.lattice.ind2coord_dir(link_ind)
        rotmat = self.generate_rotmat(theta, coord, dir)

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
        # Update the modified determinant
        offset = 2 * self.cfg.nvirtmodes_link
        if ind_mat - offset >= 0:
            for wi, update, incdet in zip(self.wi_gamma_in_mod_vec, update_vec, self.incdet_mod_vec):
                mat_inv = wi.inv()
                incdet.update_index(mat_inv, update, ind_mat - offset, ind_mat - offset)
        # Update the weight
        self.weight = 0.5 * np.sum(detval_vec)
        # Update the matrix inversion
        [
            wi_gamma_in.update_index(update, ind_mat, ind_mat)
            for wi_gamma_in, update in zip(self.wi_gamma_in_vec, update_vec)
        ]
        [
            wi_gamma_out.update_index(update, ind_mat, ind_mat)
            for wi_gamma_out, update in zip(self.wi_gamma_out_vec, update_vec)
        ]

        if ind_mat - offset >= 0:
            # We do not update the matrix if the first link is updated (it is just not there)
            [
                wi_gamma_in_mod.update_index(update, ind_mat - offset, ind_mat - offset)
                for wi_gamma_in_mod, update in zip(self.wi_gamma_in_mod_vec, update_vec)
            ]
            [
                wi_gamma_out_mod.update_index(update, ind_mat - offset, ind_mat - offset)
                for wi_gamma_out_mod, update in zip(self.wi_gamma_out_mod_vec, update_vec)
            ]

        # Invalidate gauge dependent quantities
        self.invalidate_gauge_update()

    ################## Observables ##################
    def _compute_mag_energy_op(self, use_trans_inv: bool = True):
        """Computation of the magnetic energy operator (w/o shift).
        This operator is diagonal in the gauge field (group element) basis and can thus
        be computed easily.

        This method overwrites an abstract method in System2DBase.

        Args:
            use_trans_inv (bool, optional): Use the translationally invariant computation method. Defaults to True.

        Returns:
            float: magnetic energy w/o shift for a single plaquette
        """
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
    @maybe_jit(static_argnames=["overall_factors", "idxarrs", "use_trans_inv", "nlayer"])
    def _compute_el_energy_op_vec(
        lognormvec_default,
        overall_factors,
        idxarrs,
        nlayer: int,
        covmat_out_virt_vec,
        norm_mod_vec,
        use_trans_inv: bool = True,
    ):
        """Computation of the electric energy.

        This method overwrites an abstract method in System2DBase.

        Args:
            lognormvec_default: the usual norm without any modifications
            overall_factors: prefactors for building the required Pfaffians
            idxarrs: indices for building the required Pfaffians
            nlayer (int): total number of layers (pure gauge + fermionic)
            covmat_out_virt_vec:
            norm_mod_vec:
            use_trans_inv (bool, optional): Use the translationally invariant implementation. Defaults to True.

        Returns:
            list: list of electric energies for a single link
        """
        if not use_trans_inv:
            # Evaluate every link of the system
            logger.error("compute_el_energy: The non-translational invariant case is not implemented yet.")
            raise NotImplementedError("The non-translational invariant case is not implemented yet.")

        lognorm_default = xnp.sum(lognormvec_default)

        dest = []
        # TODO: vectorize!
        for layerind in range(nlayer):

            idxarr = idxarrs[layerind]
            overall_factor = overall_factors[layerind]

            ###################### Calculation of <P> ########################

            covmat_out_virt = covmat_out_virt_vec[layerind]

            norm_mod = norm_mod_vec[layerind]
            # The matrix elements yield only the real part of <P>
            # If we use the log formulation, we can calculate the log of single terms.

            # Instead of writing down all the terms explicitly, we build tuples of the prefactors
            # and the indices of the covariance matrix.
            # Then, we compute all terms in a list comprehension.
            pfarr = []
            pfvals = []  # without the prefactor
            for prefactor, ind in idxarr:
                ind = xnp.asarray(ind)
                pfaval = backend.pfaffian(covmat_out_virt[xnp.ix_(ind, ind)])
                pfarr.append(prefactor * pfaval)
                pfvals.append(pfaval)
            el_energy_full = overall_factor * xnp.sum(xnp.array(pfarr))

            el_energy_layer = xnp.real(el_energy_full) * xnp.exp(norm_mod - lognorm_default)
            dest.append(el_energy_layer)

        return xnp.asarray(dest)

    @staticmethod
    @maybe_jit(
        static_argnames=[
            "lattice_size",
            "num_pg_layer",
            "num_fermionic_layer",
            "unitcell_size",
            "nvirtmodes_link",
            "nphysmodes_site",
            "symbolvec",
            "overall_factors",
            "idxarr_vec",
            "zeroed_params",
            "use_trans_inv",
        ]
    )
    def _compute_el_grad_vec(
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
        use_trans_inv: bool = True,
    ):
        """Computation of the electric energy gradients.
        We start by calculating the electric energies, since these are needed for evaluating the gradients.
        Since several operations needed for the computation of the gradient and the energy are similar,
        we can reuse many intermediate steps.

        This method overwrites an abstract method in System2DBase.

        Args:
            use_trans_inv (bool, optional): Use the translationally invariant implementation. Defaults to True.

        Returns:
            list: list of gradients for the full system
        """

        if not use_trans_inv:
            # Evaluate every link of the system
            logger.error("compute_el_energy: The non-translational invariant case is not implemented yet.")
            raise NotImplementedError("The non-translational invariant case is not implemented yet.")

        nlayer = num_pg_layer + num_fermionic_layer
        param_shape = (nlayer, unitcell_size, len(symbolvec))
        dest_grad = xnp.zeros(param_shape, dtype=xnp.float64)

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
            lognorm_default = xnp.sum(lognorm_default_vec)

            ###################### Calculation of the derivative ########################
            for uc_ind in range(unitcell_size):
                for symbol_ind, symbol in enumerate(symbolvec):
                    if (layerind, uc_ind, symbol_ind) not in zeroed_params:
                        # the derivative calculation is compuationally expensive
                        # we can skip it for parameters that are forced by the ansatz to be zero

                        deriv_gamma_maj_sys = gamma_maj_sys_deriv_layvec_ucvec_symbvec[layerind, uc_ind, symbol_ind]
                        d_mat_a, d_mat_b, d_mat_d = utils.extract_partial_covmats(deriv_gamma_maj_sys, offset)
                        d_gamma_out = (
                            d_mat_a
                            + d_mat_b @ diff_d_gamma_inv @ xnp.transpose(mat_b)
                            + mat_b @ diff_d_gamma_inv @ xnp.transpose(d_mat_b)
                            - mat_b @ diff_d_gamma_inv @ d_mat_d @ diff_d_gamma_inv @ np.transpose(mat_b)
                        )
                        # The virtual mode is the last link on the bottom right of the covariance matrix
                        d_covmat_out_virt = d_gamma_out[-single_link_offset:, -single_link_offset:]
                        # Summand with derivative of the covariance matrix
                        # We re-use the list comprehension from above to use the indices
                        deriv_pfarr = xnp.array(
                            [
                                prefactor
                                * utils.derivative_pfaffian(
                                    covmat_out_virt[xnp.ix_(xnp.asarray(ind), xnp.asarray(ind))],
                                    d_covmat_out_virt[xnp.ix_(xnp.asarray(ind), xnp.asarray(ind))],
                                )
                                for prefactor, ind in idxarr
                            ]
                        )
                        d_el_energy = xnp.real(overall_factor * xnp.sum(deriv_pfarr)) * xnp.exp(
                            norm_mod - lognorm_default
                        )

                        # Summand with derivative of norms
                        trace_def = grad_over_norm_vec[layerind, uc_ind, symbol_ind]
                        trace_mod = utils.compute_grad_over_norm(
                            gamma_in_sys_mod,
                            diff_d_inv_gamma_inv,
                            d_mat_d,
                            mat_d_mod_inv_vec[layerind],
                        )
                        # This is the second contribution of the elctric energy gradient F_{el} (\tilde(v) - v)
                        d_el_energy += el_energy_vec[layerind] * (trace_mod - trace_def)
                        # Scale to system size
                        d_el_energy *= nlinks
                        dest_grad = backend.array_assign(dest_grad, (layerind, uc_ind, symbol_ind), d_el_energy)

        dest_grad = xnp.asarray(dest_grad)

        # We have to weigh the different layers with the electric energy operator expectation of the other layers.
        # They act as a prefactor in the derivative
        if nlayer > 1:
            for i in range(nlayer):
                prod_other_layers = utils.multiply_except(el_energy_vec, i)
                dest_grad = backend.array_mult(dest_grad, i, prod_other_layers)

        return dest_grad

    @staticmethod
    @maybe_jit(static_argnames=["lattice_size", "use_trans_inv", "num_pg_layer", "num_fermionic_layer"])
    def _compute_mass_energy_op_vec(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        ferm_cov_vec: xnp.ndarray,
        use_trans_inv: bool = True,
    ):
        """Compute the mass term of the Hamiltonian for a single site.

        Args:
            use_trans_inv (bool, optional): Use translationally invariant implementation. Defaults to True.

        Returns:
            array: mass energy as a vec over layers
        """
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

        mass_energy_op = xnp.asarray(mass_energy_op)

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
        d_gamma_out_symbolvec: xnp.array,
        zeroed_params: tuple,
        use_trans_inv: bool = True,
    ):
        """Compute the mass term of the Hamiltonian for a single site.

        Args:
            use_trans_inv (bool, optional): Use translationally invariant implementation. Defaults to True.

        Returns:
            array: gradients of the mass energy
        """
        if not use_trans_inv:
            raise NotImplementedError("Translation invariance must be set to True.")

        nlayer = num_pg_layer + num_fermionic_layer
        param_shape = (nlayer, unitcell_size, len(symbolvec))
        gradients = xnp.zeros(param_shape, dtype=xnp.float64)

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

        return xnp.array(gradients)

    def _compute_int_energy_op_vec(self):
        """Calculate the energy due to the interaction of the
        physical fermions with the gauge fields.
        Note: this function assumes that U = U^dagger, which is valid only for Z2.
        For other groups, the calculation will not be as simple.

        Returns:
            array: interaction energy for a single link
        """

        int_energy_op = [0] * self.cfg.num_pg_layer

        for layer_ind in range(self.cfg.num_pg_layer, self.cfg.nlayer):
            layer_int_energy = 0.0
            covmat = self.ferm_covmat_vec[layer_ind]

            for site_ind in range(self.cfg.lattice.size):
                coord = self.cfg.lattice.ind2coord(site_ind)

                # this is the index to use when accessing elements of the covariance matrix,
                # which has 2 Majorana modes per site
                site_ind_cov = 2 * site_ind

                # Horizontal link
                ind_field_hor = self.cfg.lattice.coord2ind_dir(coord, Direction.X)  # index of the horizontal link
                neighborX_coord = self.cfg.lattice.get_neighbor(coord, Direction.X)  # coordinates of neighboring site
                neighborX_ind = 2 * self.cfg.lattice.coord2ind(
                    neighborX_coord
                )  # index of neighboring site, factor of 2 is due to Majorana modes (2 per site)
                gaugefield_hor = self.gaugefieldvec[
                    ind_field_hor
                ]  # gaugefield_hor is a matrix representation of a group element
                theta_hor = self.cfg.gaugemgr.get_angle(gaugefield_hor)  # convert it to an angle
                cos_factor_hor = xnp.cos(theta_hor)  # simple way to get U from gauge value
                hor_link_energy = 0.5 * (
                    covmat[site_ind_cov, neighborX_ind] - covmat[site_ind_cov + 1, neighborX_ind + 1]
                )
                layer_int_energy += hor_link_energy * cos_factor_hor

                # Vertical link
                ind_field_vert = self.cfg.lattice.coord2ind_dir(coord, Direction.Y)
                neighborY_coord = self.cfg.lattice.get_neighbor(coord, Direction.Y)
                neighborY_ind = 2 * self.cfg.lattice.coord2ind(neighborY_coord)
                gaugefield_vert = self.gaugefieldvec[ind_field_vert]
                theta_vert = self.cfg.gaugemgr.get_angle(
                    gaugefield_vert
                )  # gaugefield_vert is a matrix represntation of a group element
                cos_factor_vert = xnp.cos(theta_vert)
                vert_link_energy = 0.5 * (
                    covmat[site_ind_cov, neighborY_ind + 1] + covmat[site_ind_cov + 1, neighborY_ind]
                )
                layer_int_energy -= vert_link_energy * cos_factor_vert

            int_energy_op.append(layer_int_energy)

        int_energy_op = xnp.asarray(int_energy_op)

        return int_energy_op

    def _compute_int_energy_grad(self):
        """Calculate the energy gradient due to the interaction of the
        physical fermions with the gauge fields.
        Note: this function assumes that U = U^dagger, which is valid only for Z2.
        For other groups, the calculation will not be as simple.

        Returns:
            array: gradients
        """

        gradients = xnp.zeros(self.cfg.param_shape(), dtype=xnp.float64)

        for layer_ind in range(self.cfg.num_pg_layer, self.cfg.nlayer):

            for site_ind in range(self.cfg.lattice.size):
                coord = self.cfg.lattice.ind2coord(site_ind)

                # this is the index to use when accessing elements of the covariance matrix,
                # which has 2 Majorana modes per site
                site_ind_cov = 2 * site_ind

                # Horizontal link
                ind_field_hor = self.cfg.lattice.coord2ind_dir(coord, Direction.X)  # index of the horizontal link
                neighborX_coord = self.cfg.lattice.get_neighbor(coord, Direction.X)  # coordinates of neighboring site
                neighborX_ind = 2 * self.cfg.lattice.coord2ind(
                    neighborX_coord
                )  # index of neighboring site, factor of 2 is due to Majorana modes (2 per site)
                gaugefield_hor = self.gaugefieldvec[
                    ind_field_hor
                ]  # gaugefield_hor is a matrix representation of a group element
                theta_hor = self.cfg.gaugemgr.get_angle(gaugefield_hor)  # convert it to an angle
                cos_factor_hor = xnp.cos(theta_hor)  # simple way to get U from gauge value

                # Vertical link
                ind_field_vert = self.cfg.lattice.coord2ind_dir(coord, Direction.Y)
                neighborY_coord = self.cfg.lattice.get_neighbor(coord, Direction.Y)
                neighborY_ind = 2 * self.cfg.lattice.coord2ind(neighborY_coord)
                gaugefield_vert = self.gaugefieldvec[ind_field_vert]
                theta_vert = self.cfg.gaugemgr.get_angle(
                    gaugefield_vert
                )  # gaugefield_vert is a matrix represntation of a group element
                cos_factor_vert = xnp.cos(theta_vert)

                # Calculate derivatives
                for uc_ind in range(self.cfg.unitcell_size):
                    for symbol_ind, symbol in enumerate(self.symbolvec):
                        # the derivative calculation is relatively compuationally expensive
                        # (though less than for electric energy)
                        # we can skip it for parameters that are forced by the ansatz to be zero
                        if (layer_ind, uc_ind, symbol_ind) not in self.cfg.zeroed_params:

                            d_gamma_out = self.d_gamma_out_symbolvec[layer_ind, uc_ind, symbol_ind]
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

        return xnp.array(gradients)

    @staticmethod
    def _compute_chem_energy_op_vec(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        sublattice_factors: tuple,
        ferm_covmat_vec: xnp.ndarray,
    ):
        """Calculate the chemical potential energy operator."""

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

    def _compute_chem_energy_grad(self):
        """Calculate the chemical potential energy operator gradient."""

        gradients = xnp.zeros(self.cfg.param_shape(), dtype=xnp.float64)

        for layer_ind in range(self.cfg.num_pg_layer, self.cfg.nlayer):
            # only the fermionic layers directly contribute to the chemical potential

            # Calculate chem term
            # Since we set the system to have different parameters on the even and odd sites when using a non-zero
            # chemical potential (i.e. the system is translationally invariant by two sites),
            # we could just calculate it for one even and one odd site and multiply by the size of the system
            for site in range(self.cfg.lattice.size):
                site_ind = 2 * site  # index into covariance matrix
                site_factor = self.cfg.lattice.sublattice_factors[site]  # even or odd sublattice

                for uc_ind in range(self.cfg.unitcell_size):
                    for symbol_ind, symbol in enumerate(self.symbolvec):
                        # the derivative calculation is relatively compuationally expensive
                        # (though less than for electric energy)
                        # we can skip it for parameters that are forced by the ansatz to be zero
                        if (layer_ind, uc_ind, symbol_ind) not in self.cfg.zeroed_params:

                            d_gamma_out = self.d_gamma_out_symbolvec[layer_ind, uc_ind, symbol_ind]
                            grad = 0.5 * site_factor * d_gamma_out[site_ind + 1, site_ind]
                            gradients = backend.array_add(gradients, (layer_ind, uc_ind, symbol_ind), grad)

                    # further terms of the derivative are included higher up in the computation stack
                    # because computing them requires knowing various expectation values, which are not available here

        return gradients

    def _meson_string_vec(self, path):
        r"""Compute a layer resolved meson string for the given path.
        This is \psi^dagger (start) * String * \psi(end) before particle-hole,
        and assumes that start and end are on the same sublattice.

        Args:
            path (list): List of tuples [(index,conj),....]. conj indicates whether the argument should be conjugated.

        Returns:
            array: meson_str_vec
        """

        meson_op_vec = [0] * self.cfg.num_pg_layer

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

            meson_op_vec.append(xnp.abs(layer_val))  # TODO: is the absolute value necessary? why?
        return xnp.array(meson_op_vec)

    def occupation(self, lay: int, site: int, after_ph: bool = False) -> float:
        """Compute the occupation number for the given layer and site.

        Args:
            lay (int): Layer index
            site (int): Site index
            after_ph (bool, optional): If True, compute the occupation number using the operators
                                       defined after the particle-hole transformation. Defaults to False.

        Returns:
            float: the occupation number for the given layer and site
        """

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
