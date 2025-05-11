import logging
from pfapack import pfaffian as pf
from typing import List

import numpy as np
from ggpeps import xnp as xnp
from ggpeps import xscipy as xscipy

import ggpeps
from ggpeps import utils
from ggpeps.lattice import Direction
from ggpeps.system.global_funcs import *
from ggpeps import modearray

from .system_base import System2DBase
from .system_base import calculate_lognorm_inc

# from ggpeps.system.global_funcs import update_gauge_ind

logger = logging.getLogger(ggpeps.LOGGER_NAME)


###################### D2nSystem2D ##########################


class D2nSystem2D(System2DBase):
    """2D Z2 system GGPEPS ansatz with physical fermions.

    Some general notes about conventions:

    Order of the paramvec: see the functions that create the symbolvec in the configs.
        We split the real and the imaginary part of the parameters into independent variables.
    Mode order of tmat: {p,l1,r1,d1,u1,l2,r2,d2,u2,l3,r3... and so on}.
    Mode order of gamma_dirac: {p,l1,r1,d1,u1,l2,r2,d2,u2,p_dag,l1_dag,r1_dag,u1_dag,d1_dag,l2_dat,r2_dag,u2_dag,d2_dag,l3,r3... and so on}.
    Mode order of gamma_maj: {p_1,p_2,l1_1,l1_2,r1_1,r1_2,d1_1,d1_2,u1_1,u1_2,l2_1,l2_2,r2_1,r2_2,d2_1,d2_2,u2_1,u2_2,l3_1,l3_2... and so on}.
    """

    def __init__(self, cfg):
        """Constructor of a Z2System2D system, with any number of virtual fermions per site per link
        (provided a valid config is given).

        Args:
            cfg (Config2DBase): Configuration containing all system-related parameters
        """
        super().__init__(cfg)

    def initialize_gamma_in_sys(self):
        """
        The mode-order in gamma_in_sys is dictated by the numbering of the links on the lattice.
        The numbering guarantees that we split the vertical from the horizontal links for easier gauging.

            |         |
            "5"       "7"
            |         |
            2 --"2"-- 3 --"3"--
            |         |
            "4"       "6"
            |         |
            0 --"0"-- 1 --"1"--

        The vertex indices are written as <number>, the link indices are written as "<number>".

        For a 2x2 system with 1 copy (one virtual fermions per site per link), gamma_in has the order
        {l_1, r_0, l_0, r_1, l_3, r_2, l_2, r_3, d_2, u_0, d_0, u_2, d_3, u_1, d_1, d_3}.

        For a 2x2 system with 2 copies (two virtual fermions per site per link), gamma_in has the order
        { l1_1, r2_0, l1_1, r2_0, l1_0, r2_1, l1_0, r2_1,
          l1_3, r2_2, l1_3, r2_2, l1_2, r2_3, l1_2, r2_3,
          d1_2, u2_0, d1_2, u2_0, d1_0, u2_2, d1_0, u2_2,
          d1_3, u2_1, d1_3, u2_1, d1_1, d2_3, d1_1, d2_3 }.

        The naming convention here is <mode letter><number of copy>_<vertex index>.
        (<number of copy> is ommitted for the 1 copy case).
        Each constituent in the lists above refers to two Majorana modes.

        This method overwrites an abstract method in System2DBase.
        """

        # Initialize empty lists
        gamma_in_sys_vec = []
        wi_gamma_in_vec, wi_gamma_out_vec, incdet_vec = [], [], []
        wi_gamma_in_mod_vec, wi_gamma_out_mod_vec, incdet_mod_vec = [], [], []

        # Initialize gamma_in_sys for the full system (and trackers)
        size = self.cfg.lattice.size  # number of sites
        id = xnp.eye(size)

        # TODO: vectorize!
        for layer in range(self.cfg.nlayer):
            neutral_gauge_X = xnp.kron(
                id, self.gamma_gauge_neutral_vec[layer][Direction.X]
            )
            neutral_gauge_Y = xnp.kron(
                id, self.gamma_gauge_neutral_vec[layer][Direction.Y]
            )
            gamma_in_sys = xscipy.linalg.block_diag(neutral_gauge_X, neutral_gauge_Y)
            gamma_in_sys_vec.append(gamma_in_sys)

            wi_gamma_in_vec.append(
                utils.WoodburyInverter(self.mat_d_inv_vec[layer] - gamma_in_sys)
            )
            wi_gamma_out_vec.append(
                utils.WoodburyInverter(self.mat_d_vec[layer] - gamma_in_sys)
            )
            incdet_vec.append(
                utils.IncLogAbsDeterminant(self.mat_d_inv_vec[layer] - gamma_in_sys)
            )

            # Initialize the modified gamma_in_sys for the full system (and trackers)
            single_link_offset = 2 * self.cfg.nvirtmodes_link
            gamma_in_sys_mod = gamma_in_sys[single_link_offset:, single_link_offset:]
            wi_gamma_in_mod_vec.append(
                utils.WoodburyInverter(self.mat_d_mod_inv_vec[layer] - gamma_in_sys_mod)
            )
            wi_gamma_out_mod_vec.append(
                utils.WoodburyInverter(self.mat_d_mod_vec[layer] - gamma_in_sys_mod)
            )
            incdet_mod_vec.append(
                utils.IncLogAbsDeterminant(
                    self.mat_d_mod_inv_vec[layer] - gamma_in_sys_mod
                )
            )

        return (
            xnp.array(gamma_in_sys_vec, dtype=xnp.complex64),
            (wi_gamma_in_vec, wi_gamma_out_vec, incdet_vec),
            (wi_gamma_in_mod_vec, wi_gamma_out_mod_vec, incdet_mod_vec),
        )

    # Gauging

    def generate_rotmat(self, group_element: xnp.ndarray, coord: tuple, dir: Direction):
        """Generate the matrix to rotate gamma_in_neutral according to a given gauge field value.

        The mode order is (as for gamma_in_neutral):
            1 copy: {l_1_1, l_2_1, r_1_1, r_2_1,l_1_2,l_2_2,r_1_2,r_2_2}/{d_1_1, d_2_1, u_1_1, u_2_1,d_1_2,d_2_2,u_1_2,u_2_2},
            2 copies: {l1_1_1, l1_2_1, r1_1_1, r1_2_1,l1_1_2,l1_2_2,r1_1_2,r1_2_2,l2_1_1, l2_2_1, r2_1_1, r2_2_1,l2_1_2,l2_2_2,r2_1_2,r2_2_2}/{d1_1_1, d1_2_1, u1_1_1, u1_2_1,d1_1_2,d1_2_2,u1_1_2,u1_2_2,d2_1_1, d2_2_1, u2_1_1, u2_2_1,d2_1_2,d2_2_2,u2_1_2,u2_2_2},
        depending on whether the link is vertical or horizontal.
        The naming convention here is <mode letter><number of copy>_<majorana mode>_<color>.
        We order first by link and then by copy.

        For fermionic and pure gauge layers, the projectors don't mix copies to ensure the U(1) symmetry is obeyed.

        The sites are picked such that the left mode is right of the right modes, i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        This method overwrites an abstract method in System2DBase.

        Args:
            group_element (xnp.ndarray): Representation of group elemnt
            coord (tuple): (x,y) coordinate on the lattice
            dir (lattice.Direction): direction of the link

        Returns:
            xnp.ndarray: Rotation matrix for gamma_in_neutral
        """
        g = group_element
        # We are only rotating the right modes.
        # Thus, we leave an identity matrix for the left modes.
        g_transpose = xnp.transpose(g)
        g_dagger = xnp.conj(g_transpose)
        sum_of_g_matrices = g_transpose + g_dagger
        dif_of_g_matrices = g_transpose - g_dagger

        if xnp.sum(coord) % 2 == 0:  # gauging is different for different sublattices
            rot_right = (
                0.5
                * xnp.block(  # Note that this gauging is true only for b modes and c virtual modes (in the conventions of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.110.054511).
                    # TODO: Generalize this to fermionic layers as well.
                    [
                        [sum_of_g_matrices, -1.0j * dif_of_g_matrices],
                        [1.0j * dif_of_g_matrices, sum_of_g_matrices],
                    ],
                )
            )  # This is the rot_right for the mode order of {l_1_1, l_1_2,l_2_1,l_2_2, r_1_1, r_1_2,r_2_1,r_2_2}
        else:
            rot_right = (
                0.5
                * xnp.block(  # Note that this gauging is true only for b modes and c virtual modes (in the conventions of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.110.054511).
                    # TODO: Generalizze this to fermionic layers as well.
                    [
                        [sum_of_g_matrices, 1.0j * dif_of_g_matrices],
                        [-1.0j * dif_of_g_matrices, sum_of_g_matrices],
                    ],
                )
            )

        # We have dim(representaion) left mode => 2*dim(representation) Majorana modes
        dim_rep = len(g)  # dimension of the representation
        rot_left = xnp.eye(2 * dim_rep)

        # The mode order is lr (horizontally) or du (vertically).
        # We rotate the different copies in the SAME way.
        dest = xscipy.linalg.block_diag(
            rot_left, rot_right
        )  # This is the rot for the mode order of {l_1_1, l_1_2,l_2_1,l_2_2, r_1_1, r_1_2,r_2_1,r_2_2}
        perm_mat = xnp.array(
            modearray.generate_permutation_matrix(
                [1, 2, 3, 4, 5, 6, 7, 8],
                [1, 3, 5, 7, 2, 4, 6, 8],  # Here we assume a 2D representation
            )
        )  # Generate permutation matrix to change the modes's order to {l_1_1, l_2_1, r_1_1, r_2_1,l_1_2,l_2_2,r_1_2,r_2_2} - i.e., colors are treated similarly to copies.
        dest = xnp.transpose(perm_mat) @ dest @ perm_mat

        rotmat = xnp.kron(xnp.eye(self.cfg.ncopy), dest)
        return rotmat

    def update_gauge_ind(self, link_ind, theta):
        """Update method that is called upon changing a gauge field.
        This method is central to the algorithm since it changes the gauged projectors
        and updates all incremental trackers of determinants and inverses.
        The re-calculation of determinants and inverses for the norm would be prohibitively expensive.

        Unlike the update_non_singular_gauge_ind method, this method checks whether the transition is singular
        (i.e., the update matrix is singular and therfore can't be inverted)
        if not, it calls the update_non_singular_gauge_ind method directly. Else, it computes a non singular
        path and then calls the update_non_singular_gauge_ind method.

        This method overwrites an abstract method in System2DBase.

        Args:
            link_ind (int): Link index to be updated
            theta (xnp.array): New gauge field value
        """
        if (
            set(self._gaugefieldvec[link_ind], theta)
            in self.cfg.gaugemgr.forbidden_transitions
        ):  # if the update matrix is singular
            path = self.cfg.gaugemgr.get_nonsingular_path(
                self._gaugefieldvec[link_ind], theta
            )  # get a non singular path between the two gauge values
            for g in path:
                self.update_non_singular_gauge_ind(link_ind, g)
            self.update_non_singular_gauge_ind(
                link_ind, theta
            )  # update the gauge field to the final value
        else:  # the update matrix is not singular and we fan update the gauge straightforwardly
            self.update_non_singular_gauge_ind(link_ind, theta)

    # TODO: fix for JAX - DONE, except for stuff in utils
    def update_non_singular_gauge_ind(self, link_ind, theta):
        """Update method that is called upon changing a gauge field.
        This method is central to the algorithm since it changes the gauged projectors
        and updates all incremental trackers of determinants and inverses.
        The re-calculation of determinants and inverses for the norm would be prohibitively expensive.

        This method assumes that the two gauge values don't yield a singular update matrix.
        It is called by the update_gauge_ind method which takes care of not allowing singular updates.

        This method overwrites an abstract method in System2DBase.

        Args:
            link_ind (int): Link index to be updated
            theta (xnp.array): New gauge field value
        """
        # Update the gaugefield
        if ggpeps.PREFERRED_BACKEND == "jax":
            self._gaugefieldvec = self._gaugefieldvec.at[link_ind].set(theta)
        else:
            self._gaugefieldvec[link_ind] = theta
        # There are two directions per vertex
        ind_mat = 2 * self.cfg.nvirtmodes_link * link_ind
        coord, dir = self.cfg.lattice.ind2coord_dir(link_ind)
        rotmat = self.generate_rotmat(theta, coord, dir)

        update_vec = []
        for layer in range(self.cfg.nlayer):
            gamma_neutral_gauge = self.gamma_gauge_neutral_vec[layer][dir]
            gamma_in_subst = rotmat @ gamma_neutral_gauge @ xnp.transpose(rotmat)
            update_vec.append(
                self.calculate_update_gamma_in(
                    ind_mat, gamma_in_subst, gamma_in_sys=self.gamma_in_sys_vec[layer]
                )
            )

            # Substitute in the array
            if ggpeps.PREFERRED_BACKEND == "jax":
                # TODO: should not modify "private" variable - make a setter?
                self._gamma_in_sys_vec = self.gamma_in_sys_vec.at[
                    layer,
                    ind_mat : ind_mat + rotmat.shape[0],
                    ind_mat : ind_mat + rotmat.shape[1],
                ].set(gamma_in_subst)
            else:
                self.gamma_in_sys_vec[layer][
                    ind_mat : ind_mat + rotmat.shape[0],
                    ind_mat : ind_mat + rotmat.shape[1],
                ] = gamma_in_subst

        # Update the determinant
        mat_inv_vec = [wi_gamma_in.inv() for wi_gamma_in in self.wi_gamma_in_vec]
        detval_vec = np.array(
            [
                incdet.update_index(mat_inv, update, ind_mat, ind_mat)
                for mat_inv, update, incdet in zip(
                    mat_inv_vec, update_vec, self.incdet_vec
                )
            ]
        )
        # Update the modified determinant
        offset = 2 * self.cfg.nvirtmodes_link
        if ind_mat - offset >= 0:
            for wi, update, incdet in zip(
                self.wi_gamma_in_mod_vec, update_vec, self.incdet_mod_vec
            ):
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
                wi_gamma_out_mod.update_index(
                    update, ind_mat - offset, ind_mat - offset
                )
                for wi_gamma_out_mod, update in zip(
                    self.wi_gamma_out_mod_vec, update_vec
                )
            ]

        # Invalidate gauge dependent quantities
        self.invalidate_gauge_update()

    # def update_gauge_ind(self, link_ind, theta):
    #    update_gauge_ind(self, link_ind, theta)

    # Observables
    def _compute_mass_energy_op_vec_and_grad(self, use_trans_inv: bool = True):
        """Compute the mass term of the Hamiltonian for a single site.

        Args:
            use_trans_inv (bool, optional): Use translationally invariant implementation. Defaults to True.

        Returns:
            tuple: Tuple of (mass energy for a single site, gradients)
        """
        if not use_trans_inv:
            raise NotImplementedError("Translation invariance must be set to True.")

        mass_energy_op = [0] * self.cfg.num_pg_layer
        gradients = xnp.zeros(self.cfg.param_shape(), dtype=xnp.float64)

        for layer_ind in range(self.cfg.num_pg_layer, self.cfg.nlayer):
            # only the fermionic layers directly contribute to the mass

            # Calculation prelimaries
            covmat = self.compute_ferm_cov(layer_ind)
            layer_mass_energy = 0.0

            # Calculate mass term
            # Since the system is translationally invariant, we could just calculate it for one site and multiply by nsites instead
            for site_ind in range(0, 2 * self.cfg.lattice.size, 2):
                layer_mass_energy += 0.5 * (
                    1 + covmat[site_ind + 1, site_ind]
                )  # TODO: fix for JAX - NOT NEEDED

                for uc_ind in range(self.cfg.unitcell_size):
                    for symbol_ind, symbol in enumerate(self.symbolvec):
                        if (
                            layer_ind,
                            uc_ind,
                            symbol_ind,
                        ) not in self.cfg.zeroed_params:
                            # the derivative calculation is relatively compuationally expensive (though less than for electric energy)
                            # we can skip it for parameters that are forced by the ansatz to be zero

                            d_gamma_out = self.d_gamma_out_symbolvec(layer_ind, uc_ind)[
                                symbol_ind
                            ]
                            if ggpeps.PREFERRED_BACKEND == "numpy":
                                gradients[layer_ind, uc_ind, symbol_ind] += (
                                    0.5 * d_gamma_out[site_ind + 1, site_ind]
                                )
                            elif ggpeps.PREFERRED_BACKEND == "jax":
                                gradients = gradients.at[
                                    layer_ind, uc_ind, symbol_ind
                                ].add(0.5 * d_gamma_out[site_ind + 1, site_ind])

                    # further terms of the derivative are included higher up in the computation stack
                    # because computing them requires knowing various expectation values, which are not available here

            mass_energy_op.append(xnp.asarray(layer_mass_energy))

        mass_energy_op = xnp.asarray(mass_energy_op)

        self.cfg.enforce_parameter_conditions(gradients)

        # When computing the electric energy, we have to weigh the gradients of each layer with the electric energy operator expectation of the other layers.
        # They act as a prefactor in the derivative.
        # However, here, because the mass term only acts on the fermionic layers, we simply multiply the mass_energy and grads by the norm of the first layer
        # (this is handled higher up in the computation stack).

        return mass_energy_op, xnp.array(gradients)

    def _compute_el_energy_op_vec(self, use_trans_inv: bool = True):
        """Computation of the electric energy.
        Since several operations needed for the computation of the gradient and the energy are similar, we can reuse many intermediate steps.
        These are saved at the end of the function.

        This method overwrites an abstract method in System2DBase.

        Args:
            use_trans_inv (bool, optional): Use the translationally invariant implementation. Defaults to True.

        Returns:
            list: list of electric energies for a single link
        """
        if not use_trans_inv:
            # Evaluate every link of the system
            logger.error(
                "compute_el_energy: The non-translational invariant case is not implemented yet."
            )
            raise NotImplementedError(
                "The non-translational invariant case is not implemented yet."
            )

        lognormvec_default = self.calculate_lognormvec_inc(all_factors=True)
        # This is the usual norm without any modifications
        lognorm_default = xnp.sum(lognormvec_default)
        # Number of fermions = # of sites
        # Since we have 2 copies, we get 8 virtual fermions per site
        single_link_offset = 2 * self.cfg.nvirtmodes_link
        # We have to cut one link from gamma_in_sys as well
        gamma_in_sys_mod_vec = self.gamma_in_sys_mod_vec
        dest = []

        # Indices and prefactors for building the required Pfaffians
        overall_factors = self.cfg.el_overall_factors
        idxarrs = self.cfg.idxarr_vec

        # TODO: vectorize!
        for layerind in range(self.cfg.nlayer):

            # We shift the first virtual link (0,0,X) towards the physical modes to trace out everything else
            mat_a = self.mat_a_mod_vec[
                layerind
            ]  # dim: 2*nsites (for majorana) + 8 (= 4 virtual modes per link x2 for majorana)
            mat_b = self.mat_b_mod_vec[layerind]
            diff_d_gamma_inv = self.wi_gamma_out_mod_vec[layerind].inv()

            gamma_in_sys_mod = gamma_in_sys_mod_vec[layerind]

            idxarr = idxarrs[layerind]
            overall_factor = overall_factors[layerind]

            ###################### Calculation of <P> ########################
            covmat_out = mat_a + mat_b @ diff_d_gamma_inv @ xnp.transpose(mat_b)
            size = covmat_out.shape[1]
            covmat_out_virt = slice_matrix(
                covmat_out,
                size - single_link_offset,
                size,
                size - single_link_offset,
                size,
            )
            # covmat_out[-single_link_offset:, -single_link_offset:] # TODO: fix for JAX - DONE

            # The library pfapack is rather picky about the anti-symmetrization (to 1e-14)
            covmat_out_virt = utils.anti_symmetrize(covmat_out_virt)
            # For the modified norm, we still have to take into account the other contributions from the unmodified parts
            norm_mod = calculate_lognorm_inc(
                [self.incdet_mod_vec[layerind]],
                [self.det_mat_d_mod_vec[layerind]],
                gamma_in_sys_mod.shape[0],
                all_factors=True,
            )
            norm_mod += xnp.sum(utils.select_except(lognormvec_default, layerind))
            # The matrix elements yield only the real part of <P>
            # If we use the log formulation, we can calculate the log of single terms.

            # Instead of writing down all the terms explicitly, we build tuples of the prefactors and the indices of the covariance matrix.
            # Then, we compute all terms in a list comprehension.
            pfarr = []
            pfvals = []  # without the prefactor
            for prefactor, ind in idxarr:
                ind = xnp.asarray(ind)
                pfaval = pf.pfaffian(
                    covmat_out_virt[xnp.ix_(ind, ind)]
                )  # TODO: fix for JAX - NOT NEEDED, jxnp.ix_ should work
                pfarr.append(prefactor * pfaval)
                pfvals.append(pfaval)
            el_energy_full = overall_factor * xnp.sum(xnp.array(pfarr))

            el_energy_layer = xnp.real(el_energy_full) * xnp.exp(
                norm_mod - lognorm_default
            )
            dest.append(el_energy_layer)

            # Save intermediate calculations for use in gradient calculation
            intermediate = self._electric_energy_intermediate_vals
            intermediate.covmat_out_virt_vec.append(covmat_out_virt)
            intermediate.norm_mod_vec.append(norm_mod)
            intermediate.lognorm_default_vec.append(lognorm_default)
            intermediate.pfaffian_vec.append(pfvals)

        return xnp.asarray(dest)

    def _compute_el_grad_vec(self, use_trans_inv: bool = True):
        """Computation of the electric energy gradients.
        We start by calculating the electric energies, since these are needed for evaluating the gradients.
        Since several operations needed for the computation of the gradient and the energy are similar, we can reuse many intermediate steps.

        This method overwrites an abstract method in System2DBase.

        Args:
            use_trans_inv (bool, optional): Use the translationally invariant implementation. Defaults to True.

        Returns:
            list: list of gradients for the full system
        """

        if not use_trans_inv:
            # Evaluate every link of the system
            logger.error(
                "compute_el_energy: The non-translational invariant case is not implemented yet."
            )
            raise NotImplementedError(
                "The non-translational invariant case is not implemented yet."
            )

        gradients = compute_el_grad_vec(self)
        return gradients

    def _compute_mag_energy_op(self, use_trans_inv: bool = True):
        """Computation of the magnetic energy operator (w/o shift).
        This operator is diagonal in the gauge field (group element) basis and can thus be computed easily.

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
                    wilson_plaquette = self.cfg.lattice.generate_wilson_loop(
                        (x, y), (1, 1)
                    )
                    mag_energy_bare += xnp.real(self.compute_path(wilson_plaquette))
        return mag_energy_bare

    def _compute_int_energy_op_vec_and_grad(self):
        """Calculate the energy and energy gradient due to the interaction of the physical fermions with the gauge fields.
        Note: this function assumes that U = U^dagger, which is valid only for Z2. For other groups, the calculation will not be as simple.

        Returns:
            tuple: Tuple of (interaction energy for a single link, gradients)
        """

        int_energy_op = [0] * self.cfg.num_pg_layer
        gradients = xnp.zeros(self.cfg.param_shape(), dtype=xnp.float64)

        for layer_ind in range(self.cfg.num_pg_layer, self.cfg.nlayer):
            layer_int_energy = 0.0
            covmat = self.compute_ferm_cov(layer_ind)

            for site_ind in range(self.cfg.lattice.size):
                coord = self.cfg.lattice.ind2coord(site_ind)
                site_ind_cov = (
                    2 * site_ind
                )  # this is the index to use when accessing elements of the covariance matrix, which has 2 Majorana modes per site

                # Horizontal link
                ind_field_hor = self.cfg.lattice.coord2ind_dir(
                    coord, Direction.X
                )  # index of the horizontal link
                neighborX_coord = self.cfg.lattice.get_neighbor(
                    coord, Direction.X
                )  # coordinates of neighboring site
                neighborX_ind = 2 * self.cfg.lattice.coord2ind(
                    neighborX_coord
                )  # index of neighboring site, factor of 2 is due to Majorana modes (2 per site)
                gaugefield_hor = self.gaugefieldvec[ind_field_hor]
                cos_factor_hor = xnp.cos(
                    gaugefield_hor
                )  # simple way to get U from gauge value
                hor_link_energy = 0.5 * (
                    covmat[site_ind_cov, neighborX_ind]
                    - covmat[site_ind_cov + 1, neighborX_ind + 1]
                )  # TODO: fix for JAX - NOT NEEDED
                layer_int_energy += hor_link_energy * cos_factor_hor

                # Vertical link
                ind_field_vert = self.cfg.lattice.coord2ind_dir(coord, Direction.Y)
                neighborY_coord = self.cfg.lattice.get_neighbor(coord, Direction.Y)
                neighborY_ind = 2 * self.cfg.lattice.coord2ind(neighborY_coord)
                gaugefield_vert = self.gaugefieldvec[ind_field_vert]
                cos_factor_vert = xnp.cos(gaugefield_vert)
                vert_link_energy = 0.5 * (
                    covmat[site_ind_cov, neighborY_ind + 1]
                    + covmat[site_ind_cov + 1, neighborY_ind]
                )
                layer_int_energy -= vert_link_energy * cos_factor_vert

                # Calculate derivatives
                for uc_ind in range(self.cfg.unitcell_size):
                    for symbol_ind, symbol in enumerate(self.symbolvec):
                        if (
                            layer_ind,
                            uc_ind,
                            symbol_ind,
                        ) not in self.cfg.zeroed_params:
                            # the derivative calculation is relatively compuationally expensive (though less than for electric energy)
                            # we can skip it for parameters that are forced by the ansatz to be zero

                            d_gamma_out = self.d_gamma_out_symbolvec(layer_ind, uc_ind)[
                                symbol_ind
                            ]
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
                            if ggpeps.PREFERRED_BACKEND == "numpy":
                                gradients[layer_ind, uc_ind, symbol_ind] += grad
                            elif ggpeps.PREFERRED_BACKEND == "jax":
                                gradients = gradients.at[
                                    layer_ind, uc_ind, symbol_ind
                                ].add(grad)

            int_energy_op.append(layer_int_energy)

        int_energy_op = xnp.asarray(int_energy_op)

        self.cfg.enforce_parameter_conditions(gradients)

        # When computing the electric energy, we have to weigh the gradients of each layer with the electric energy operator expectation of the other layers.
        # They act as a prefactor in the derivative.
        # However, here (just as in the mass case), because the interaction term only acts on the fermionic layers, we simply multiply the int_energy and grads by the norm of the first layer
        # (this is handled higher up in the computation stack).

        return int_energy_op, xnp.array(gradients)

    def _compute_chem_energy_op_vec_and_grad(self):
        """Calculate the chemical potential energy operator and its gradient."""

        chem_energy_op = [0] * self.cfg.num_pg_layer
        gradients = xnp.zeros(self.cfg.param_shape(), dtype=xnp.float64)

        for layer_ind in range(self.cfg.num_pg_layer, self.cfg.nlayer):
            # only the fermionic layers directly contribute to the chemical potential

            # Calculation prelimaries
            covmat = self.compute_ferm_cov(layer_ind)
            layer_chem_energy = 0.0

            # Calculate chem term
            # Since we set the system to have different parameters on the even and odd sites when using a non-zero
            # chemical potential (i.e. the system is translationally invariant by two sites),
            # we could just calculate it for one even and one odd site and multiply by the size of the system
            for site in range(self.cfg.lattice.size):
                site_ind = 2 * site  # index into covariance matrix
                x, y = self.cfg.lattice.ind2coord(site)
                site_factor = (-1) ** (x + y)  # even or odd sublattice
                mass_site = 0.5 * (1 + covmat[site_ind + 1, site_ind])
                layer_chem_energy += site_factor * mass_site
                layer_chem_energy += 0.5  # constant offset which arises from particle-hole transformation

                for uc_ind in range(self.cfg.unitcell_size):
                    for symbol_ind, symbol in enumerate(self.symbolvec):
                        if (
                            layer_ind,
                            uc_ind,
                            symbol_ind,
                        ) not in self.cfg.zeroed_params:
                            # the derivative calculation is relatively compuationally expensive (though less than for electric energy)
                            # we can skip it for parameters that are forced by the ansatz to be zero

                            d_gamma_out = self.d_gamma_out_symbolvec(layer_ind, uc_ind)[
                                symbol_ind
                            ]
                            if ggpeps.PREFERRED_BACKEND == "numpy":
                                gradients[layer_ind, uc_ind, symbol_ind] += (
                                    0.5
                                    * site_factor
                                    * d_gamma_out[site_ind + 1, site_ind]
                                )
                            elif ggpeps.PREFERRED_BACKEND == "jax":
                                gradients = gradients.at[
                                    layer_ind, uc_ind, symbol_ind
                                ].add(
                                    0.5
                                    * site_factor
                                    * d_gamma_out[site_ind + 1, site_ind]
                                )

                    # further terms of the derivative are included higher up in the computation stack
                    # because computing them requires knowing various expectation values, which are not available here

            chem_energy_op.append(np.asarray(layer_chem_energy))

        chem_energy_op = np.asarray(chem_energy_op)

        self.cfg.enforce_parameter_conditions(gradients)

        return chem_energy_op, gradients

    def _meson_string_vec(self, path):
        """Compute a layer resolved meson string for the given path.
        This is \psi^dagger (start) * String * \psi(end) before particle-hole, and assumes that start and end are on the same sublattice.

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
            covmat = self.compute_ferm_cov(layer_ind)

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

            meson_op_vec.append(
                xnp.abs(layer_val)
            )  # Is the absolute value necessary? why?
        return xnp.array(meson_op_vec)

    def occupation(self, lay: int, site: int) -> float:
        """Compute the occupation number for the given layer and site.

        Returns:
            float: the occupation number for the given layer and site
        """

        covmat = self.compute_ferm_cov(lay)
        site_ind = 2 * site  # index into covariance matrix
        mass_site = 0.5 * (1 + covmat[site_ind + 1, site_ind])
        return mass_site
