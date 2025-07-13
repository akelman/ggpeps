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
            xnp.array(gamma_in_sys_vec),
            (wi_gamma_in_vec, wi_gamma_out_vec, incdet_vec),
            (wi_gamma_in_mod_vec, wi_gamma_out_mod_vec, incdet_mod_vec),
        )

    # Calculating weight attempt
    def calculate_weight_attempt_non_singular(
        self, link_ind: int, theta: xnp.array, all_factors=False, color_to_check=None
    ):
        """
        Compute the weight of an update attempt in which the link index link_ind is substituted for theta
        The inclusion of all constant pre-factors can be switched on and off.

        This method assumes that the two gauge values don't yield a singular update matrix.
        It is called by the calculate_weight_attempt method which takes care of not allowing singular updates.

        Args:
            link_ind (int): Link index
            theta (xnp.array): New gauge field value
            all_factors (bool, optional): Include all constant factors. Defaults to False.

        Returns:
            float: Logarithm of the weight of the proposed configuration
        """
        # There are two directions per vertex and two Majoranas per link
        coord, dir = self.cfg.lattice.ind2coord_dir(link_ind)
        rotmat = self.generate_rotmat(theta, coord, dir)
        gamma_neutral_gauge_vec = self.gamma_gauge_neutral_vec
        if color_to_check is not None:
            ind_mat = (
                2 * self.cfg.nvirtmodes_link * link_ind
                + 2 * color_to_check * self.cfg.nvirtmodes_link_per_color
            )
            rotmat = slice_matrix(
                rotmat,
                2 * self.cfg.nvirtmodes_link_per_color * color_to_check,
                2 * self.cfg.nvirtmodes_link_per_color * (color_to_check + 1),
                2 * self.cfg.nvirtmodes_link_per_color * color_to_check,
                2 * self.cfg.nvirtmodes_link_per_color * (color_to_check + 1),
            )
            gamma_in_subst_layers = []
            for gamma_neutral_gauge in gamma_neutral_gauge_vec:
                gamma_neutral_gauge_sliced = slice_matrix(
                    gamma_neutral_gauge[dir],
                    2 * self.cfg.nvirtmodes_link_per_color * color_to_check,
                    2 * self.cfg.nvirtmodes_link_per_color * (color_to_check + 1),
                    2 * self.cfg.nvirtmodes_link_per_color * color_to_check,
                    2 * self.cfg.nvirtmodes_link_per_color * (color_to_check + 1),
                )
                gamma_in_subst_layers.append(
                    rotmat @ gamma_neutral_gauge_sliced @ xnp.transpose(rotmat)
                )
        else:
            ind_mat = 2 * self.cfg.nvirtmodes_link * link_ind
            gamma_in_subst_layers = [
                rotmat @ gamma_neutral_gauge[dir] @ xnp.transpose(rotmat)
                for gamma_neutral_gauge in gamma_neutral_gauge_vec
            ]

        updates = [
            self.calculate_update_gamma_in(ind_mat, gamma_in_subst, gamma_in_sys)
            for gamma_in_subst, gamma_in_sys in zip(
                gamma_in_subst_layers, self.gamma_in_sys_vec
            )
        ]
        return self.update_lognorm_inc(ind_mat, updates, all_factors)

    def calculate_weight_attempt(
        self, link_ind: int, theta: xnp.array, all_factors=False
    ):
        """
        This method overwrites an abstract method in System2DBase. For now, we need it only for the D2n systems.

        Compute the weight of an update attempt in which the link index link_ind is substituted for theta
        The inclusion of all constant pre-factors can be switched on and off.

        Unlike the calculate_weight_attempt_non_singular method, this method checks whether the transition is singular
        (i.e., the update matrix is singular and therfore can't be inverted).
        If not, it calls the calculate_weight_attempt_non_singular method directly.
        Else, it updates the gauge in along a non singular path and then calls the
        calculate_weight_attempt_non_singular method.
        After the caclulation of the weight, it updates the system back to the original gauge field.

        Args:
            link_ind (int): Link index
            theta (xnp.array): New gauge field value
            all_factors (bool, optional): Include all constant factors. Defaults to False.

        Returns:
            float: Logarithm of the weight of the proposed configuration
        """
        # TODO: If we create a new state class object, avoiding singular transitions could be handled better
        # (by keeping the previous state and then not having to update the current system back to the original gauge field)
        current_theta = xnp.copy(self._gaugefieldvec[link_ind])
        singular = False
        color_to_check = None
        g_transition_1, g_transition_2 = (
            self.cfg.gaugemgr.transition_pair
        )  # The transition that connects the two unconnected subgtoups of elements that are connected by singular paths
        for (
            g_tuple
        ) in (
            self.cfg.gaugemgr.forbidden_transitions
        ):  # check if the update matrix is expected to be singular
            g1, g2 = g_tuple
            if (xnp.allclose(g1, current_theta) and xnp.allclose(g2, theta)) or (
                xnp.allclose(g1, theta) and xnp.allclose(g2, current_theta)
            ):
                singular = True
                break
        if singular:
            path = self.cfg.gaugemgr.get_nonsingular_path(
                current_theta, theta
            )  # get a non singular path between the two gauge values
            for g in path:
                self.update_gauge_ind(link_ind, g)
            current_g = xnp.copy(path[-1])
            if (
                xnp.allclose(current_g, g_transition_1)
                and xnp.allclose(theta, g_transition_2)
            ) or (
                xnp.allclose(current_g, g_transition_2)
                and xnp.allclose(theta, g_transition_1)
            ):

                color_to_check = 1
            weight = self.calculate_weight_attempt_non_singular(
                link_ind, theta, all_factors, color_to_check=color_to_check
            )  # calculate the weight of the last gauge value

            for g in path[-2::-1]:  # go back to the original gauge field
                self.update_gauge_ind(link_ind, g)
            self.update_gauge_ind(
                link_ind, current_theta
            )  # go back to the original gauge field

        else:  # the update matrix is not singular and we can update the gauge straightforwardly
            if (
                xnp.allclose(current_theta, g_transition_1)
                and xnp.allclose(theta, g_transition_2)
            ) or (
                xnp.allclose(current_theta, g_transition_2)
                and xnp.allclose(theta, g_transition_1)
            ):

                color_to_check = 1

            weight = self.calculate_weight_attempt_non_singular(
                link_ind, theta, all_factors, color_to_check=color_to_check
            )

        return weight

    # Gauging

    def generate_rotmat(self, group_element: xnp.ndarray, coord: tuple, dir: Direction):
        """Generate the matrix to rotate gamma_in_neutral according to a given gauge field value.

        The mode order is (as for gamma_in_neutral):
            1 copy:
                {l_1_1, l_2_1, r_1_1, r_2_1,l_1_2,l_2_2,r_1_2,r_2_2}
                or (for vertical links)
                {d_1_1, d_2_1, u_1_1, u_2_1,d_1_2,d_2_2,u_1_2,u_2_2},
            2 copies:
                {l1_1_1, l1_2_1, r1_1_1, r1_2_1,l2_1_1,l2_2_1,r2_1_1,r2_2_1,l1_1_2,l1_2_2,r1_1_2,r1_2_2,l2_1_2,l2_2_2,r2_1_2,r2_2_2}
                or (for vertical links)
                {d1_1_1, d1_2_1, u1_1_1, u1_2_1,d2_1_1,d2_2_1,u2_1_1,u2_2_1,d1_1_2,d1_2_2,u1_1_2,u1_2_2,d2_1_2,d2_2_2,u2_1_2,u2_2_2},

        The naming convention here is <mode letter><number of copy>_<majorana mode>_<color>.
        We order first by link and then by copy.

        For fermionic and pure gauge layers, the projectors don't mix copies to ensure the U(1) symmetry is obeyed.

        The sites are picked such that the left mode is right of the right modes,
        i.e. they are sitting on the same link.
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
        real_g_transpose = xnp.real(g_transpose)
        imag_g_transpose = xnp.imag(g_transpose)
        if xnp.sum(coord) % 2 == 0:  # gauging is different for different sublattices
            rot_right = xnp.block(  # Note that this gauging is true only for b modes and c virtual modes (in the conventions of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.110.054511).
                # TODO: Generalize this to fermionic layers as well.
                [
                    [real_g_transpose, imag_g_transpose],
                    [-imag_g_transpose, real_g_transpose],
                ],
            )  # This is the rot_right for the mode order of {r_1_1, r_1_2,r_2_1,r_2_2}
        else:
            rot_right = xnp.block(  # Note that this gauging is true only for b modes and c virtual modes (in the conventions of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.110.054511).
                # TODO: Generalizze this to fermionic layers as well.
                [
                    [real_g_transpose, -imag_g_transpose],
                    [imag_g_transpose, real_g_transpose],
                ],
            )

        # We have dim(representaion) left mode => 2*dim(representation) Majorana modes
        dim_rep = len(g)  # dimension of the representation
        rot_left = xnp.eye(2 * dim_rep)

        # The mode order is lr (horizontally) or du (vertically).
        # We rotate the different copies in the SAME way.
        dest = xscipy.linalg.block_diag(
            rot_left, rot_right
        )  # This is the rot for the mode order of {l_1_1, l_1_2,l_2_1,l_2_2, r_1_1, r_1_2,r_2_1,r_2_2}

        rotmat = xnp.kron(xnp.eye(self.cfg.ncopy), dest)

        wrong_order = (
            self.get_wrong_single_link_majorana_mode_order_by_copy_then_color()
        )
        correct_order_first_color_then_copy = self.get_single_link_majorana_mode_order()
        perm_mat = xnp.array(
            modearray.generate_permutation_matrix(
                wrong_order,
                correct_order_first_color_then_copy,
            )
        )  # Generate permutation matrix to change the modes's order to {l1_1_1, l1_2_1, r1_1_1, r1_2_1,l2_1_1,l2_2_1,r2_1_1,r2_2_1,l1_1_2, l1_2_2, r1_1_2, r1_2_2,l2_1_2,l2_2_2,r2_1_2,r2_2_2}.
        rotmat = xnp.transpose(perm_mat) @ rotmat @ perm_mat

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
        old_theta = xnp.copy(self._gaugefieldvec[link_ind])
        singular = False
        g_transition_1, g_transition_2 = (
            self.cfg.gaugemgr.transition_pair
        )  # The transition that connects the two unconnected subgtoups of elements that are connected by singular paths
        for (
            g_tuple
        ) in (
            self.cfg.gaugemgr.forbidden_transitions
        ):  # check if the update matrix is expected to be singular
            g1, g2 = g_tuple
            if (xnp.allclose(g1, old_theta) and xnp.allclose(g2, theta)) or (
                xnp.allclose(g1, theta) and xnp.allclose(g2, old_theta)
            ):
                singular = True
                break
        previous_g = xnp.copy(old_theta)
        if singular:  # if the update matrix is singular
            path = self.cfg.gaugemgr.get_nonsingular_path(
                old_theta, theta
            )  # get a non singular path between the two gauge values
            for g in path:
                color_to_update = None  # we update both colors
                if (
                    xnp.allclose(previous_g, g_transition_1)
                    and xnp.allclose(g, g_transition_2)
                ) or (
                    xnp.allclose(previous_g, g_transition_2)
                    and xnp.allclose(g, g_transition_1)
                ):  # in this case we update only the color m=1 (second color)
                    color_to_update = 1
                self.update_non_singular_gauge_ind(
                    link_ind, g, color_to_update=color_to_update
                )
                previous_g = xnp.copy(g)
        color_to_update = None  # we update both colors
        if (
            xnp.allclose(previous_g, g_transition_1)
            and xnp.allclose(theta, g_transition_2)
        ) or (
            xnp.allclose(previous_g, g_transition_2)
            and xnp.allclose(theta, g_transition_1)
        ):  # in this case we update only the color m=1 (second color)
            color_to_update = 1
        self.update_non_singular_gauge_ind(
            link_ind, theta, color_to_update=color_to_update
        )  # in case it was originally a singular, we update the gauge field to the final value. In the other case we can update the gauge straightforwardly

    # TODO: fix for JAX - DONE, except for stuff in utils
    def update_non_singular_gauge_ind(self, link_ind, theta, color_to_update=None):
        """Update method that is called upon changing a gauge field.
        This method is central to the algorithm since it changes the gauged projectors
        and updates all incremental trackers of determinants and inverses.
        The re-calculation of determinants and inverses for the norm would be prohibitively expensive.

        This method assumes that the two gauge values don't yield a singular update matrix.
        It is called by the update_gauge_ind method which takes care of not allowing singular updates.

        For updatting just one color we assume a specific ordering of the modes: (for example {copy=1_color=1,copy=2_color=1,copy=1_color=2,copy=2_color=2}).

        This method overwrites an abstract method in System2DBase.

        Args:
            link_ind (int): Link index to be updated
            theta (xnp.array): New gauge field value
            color_to_update (int, optional): Color to update. If None, both colors are updated. Defaults to None.
        """
        # Update the gaugefield
        if ggpeps.PREFERRED_BACKEND == "jax":
            self._gaugefieldvec = self._gaugefieldvec.at[link_ind].set(theta)
        else:
            self._gaugefieldvec[link_ind] = theta
        # There are two directions per vertex
        coord, dir = self.cfg.lattice.ind2coord_dir(link_ind)
        rotmat = self.generate_rotmat(theta, coord, dir)
        if color_to_update is None:  # if we update both colors.
            ind_mat = 2 * self.cfg.nvirtmodes_link * link_ind
        else:
            ind_mat = (
                2 * self.cfg.nvirtmodes_link * link_ind
                + 2 * color_to_update * self.cfg.nvirtmodes_link_per_color
            )
            rotmat = slice_matrix(  # In this case we slice rotmat to only contain the relevant color
                # We assume a specific ordering of the modes: (for example {copy=1_color=1,copy=2_color=1,copy=1_color=2,copy=2_color=2})
                rotmat,
                2 * self.cfg.nvirtmodes_link_per_color * color_to_update,
                2 * self.cfg.nvirtmodes_link_per_color * (color_to_update + 1),
                2 * self.cfg.nvirtmodes_link_per_color * color_to_update,
                2 * self.cfg.nvirtmodes_link_per_color * (color_to_update + 1),
            )

        update_vec = []
        for layer in range(self.cfg.nlayer):
            gamma_neutral_gauge = self.gamma_gauge_neutral_vec[layer][dir]
            if color_to_update is not None:
                gamma_neutral_gauge = slice_matrix(  # In this case we slice gamma_neutral_gauge to only contain the relevant color
                    # We assume a specific ordering of the modes: (for example {copy=1_color=1,copy=2_color=1,copy=1_color=2,copy=2_color=2})
                    xnp.copy(gamma_neutral_gauge),
                    2 * self.cfg.nvirtmodes_link_per_color * color_to_update,
                    2 * self.cfg.nvirtmodes_link_per_color * (color_to_update + 1),
                    2 * self.cfg.nvirtmodes_link_per_color * color_to_update,
                    2 * self.cfg.nvirtmodes_link_per_color * (color_to_update + 1),
                )
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
        mass_energy_op = xnp.zeros(self.cfg.nlayer)
        gradients = xnp.zeros(self.cfg.param_shape())
        return mass_energy_op, gradients

    def _compute_el_energy_op_vec(self, use_trans_inv: bool = True):
        """Computation of the electric energy.
        Since several operations needed for the computation of the gradient and the energy are similar,
        we can reuse many intermediate steps. These are saved at the end of the function.

        This method overwrites an abstract method in System2DBase.

        Args:
            use_trans_inv (bool, optional): Use the translationally invariant implementation. Defaults to True.

        Returns:
            list: list of electric energies for a single link
        """
        dest = xnp.zeros(self.cfg.nlayer)
        return xnp.asarray(dest)

    def _compute_el_grad_vec(self, use_trans_inv: bool = True):
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
            logger.error(
                "compute_el_energy: The non-translational invariant case is not implemented yet."
            )
            raise NotImplementedError(
                "The non-translational invariant case is not implemented yet."
            )

        gradients = xnp.zeros(self.cfg.param_shape())
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
        """Calculate the energy and energy gradient due to the interaction of the
        physical fermions with the gauge fields.

        Note: this function assumes that U = U^dagger, which is valid only for Z2.
        For other groups, the calculation will not be as simple.

        Returns:
            tuple: Tuple of (interaction energy for a single link, gradients)
        """

        int_energy_op = xnp.zeros(self.cfg.nlayer)
        gradients = xnp.zeros(self.cfg.param_shape())
        return int_energy_op, xnp.array(gradients)

    def _compute_chem_energy_op_vec_and_grad(self):
        """Calculate the chemical potential energy operator and its gradient."""
        chem_energy_op = xnp.zeros(self.cfg.nlayer)
        gradients = xnp.zeros(self.cfg.param_shape())
        return chem_energy_op, gradients

    def _meson_string_vec(self, path):
        """Compute a layer resolved meson string for the given path.
        This is \psi^dagger (start) * String * \psi(end) before particle-hole,
        and assumes that start and end are on the same sublattice.

        Args:
            path (list): List of tuples [(index,conj),....]. conj indicates whether the argument should be conjugated.

        Returns:
            array: meson_str_vec
        """
        meson_op_vec = xnp.zeros(self.cfg.nlayer)
        return xnp.array(meson_op_vec)

    def occupation(self, lay: int, site: int, after_ph: bool = False) -> float:
        """Compute the occupation number for the given layer and site.

        Args:
            lay (int): Layer index
            site (int): Site index
            after_ph (bool, optional): If True, compute the occupation number using the operators defined after the
                                       particle-hole transformation. Defaults to False.

        Returns:
            float: the occupation number for the given layer and site
        """

        return 0.0
