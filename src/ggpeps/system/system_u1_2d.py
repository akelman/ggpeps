import sys
import sympy
import logging

import numpy as np
from ggpeps import xnp as xnp

from scipy.linalg import block_diag
from pfapack import pfaffian as pf

import ggpeps
import ggpeps.lattice as lat
from ggpeps import utils, gauge
from ggpeps.lattice import Direction

from .system_base import (
    Config2DBase,
    System2DBase,
    calculate_lognorm,
    calculate_lognormvec_inc,
    extract_partial_covmats,
    calculate_lognorm_inc,
)

logger = logging.getLogger(ggpeps.LOGGER_NAME)

################### U1MultilayerSystem2D ###################


class U1System2DConfig(Config2DBase):
    _nparams = 3
    ncopy = 1
    nvirtmodes_link = 8
    nvirtmodes_link = 4
    nphysmodes_site = 1  # number of physical modes per site

    def __init__(
        self,
        lattice,
        g_el,
        g_mag,
        g_int,
        g_mass,
        g_chem,
        num_pg_layer=1,
        num_fermionic_layer=0,
        unitcell_size=1,
    ):
        # The parameters have the following order: [[t1,y1,z1],[t2,y2,z2],....]
        super().__init__(
            lattice,
            g_el,
            g_mag,
            g_int,
            g_mass,
            g_chem,
            num_pg_layer,
            num_fermionic_layer,
        )

        # Translation invariance
        if unitcell_size not in [1]:
            logger.error(
                "This ansatz only supports unitcell_size = 1 or 2. \
                This can be adapted by adding in a specification in the config to map sites to parameters."
            )
            raise ValueError("Invalid unitcell_size.")
        self.site_params_dict = {
            site: 0 for site in range(self.lattice.size)
        }  # map from site to index of independent parameters
        self.unitcell_size = 1
        self.gaugemgr: gauge.ZNGauge = gauge.ZNGauge(3)

    def make_pure_gauge(self):
        # The order of the parameters is [t,y,z]
        for lay in range(self.nlayer):
            for uc_ind in range(self.unitcell_size):
                self.paramvec[lay, uc_ind, 0] = 0

    def _create_symbolvec(self):
        t = sympy.Symbol("t", real=True)
        y = sympy.Symbol("y", real=True)
        z = sympy.Symbol("z", real=True)
        return [t, y, z]

    def compute_tmat_symb_single(self):
        [t, y, z] = self.symbolvec
        etap = sympy.exp(1.0j * sympy.pi / 4.0)
        zsqrt = z / sympy.sqrt(2)
        tmat_symb_single = sympy.Matrix(
            [
                [t, etap**2 * t, etap * t, etap**3 * t],
                [0, y, zsqrt, zsqrt],
                [-y, 0, -zsqrt, zsqrt],
                [-zsqrt, zsqrt, 0, y],
                [-zsqrt, -zsqrt, -y, 0],
            ]
        )
        return tmat_symb_single

    @property
    def tmat_symb(self):
        tmat_symb = sympy.zeros(9, 9)
        tmat_symb_single = self.compute_tmat_symb_single()
        tmat_symb[0:5, 5:] = tmat_symb_single
        tmat_symb[5:, 0:5] = -tmat_symb_single.T
        return tmat_symb

    def generate_gamma_gauge_neutral_dict(self):
        # Note: unlike in the Z2 case, here we can ignore the direction of the link
        dest = [0] * 2
        dest[Direction.X] = np.real(
            1.0j * np.kron(np.kron(utils.pauliy, utils.paulix), utils.paulix)
        )
        dest[Direction.Y] = np.real(
            1.0j * np.kron(np.kron(utils.pauliy, utils.paulix), utils.paulix)
        )
        return [dest] * self.nlayer


class U1System2D(System2DBase):
    """NOTE: The mode ordering of the T matrix in this class is different from all other classes in this repo.
    Order of the paramvec: [t,y,z]
    Mode order of T: {p,l,r,u,d}
    Mode Order of gamma_dirac:  {p, l+, l-, r+, r-, d+, d-, u+, u-, psi_dag, l+_dag, l-_dag, r+_dag, r-_dag, d+_dag, d-_dag, u+_dag, u-_dag}
    Mode Order of gamma_maj: {p_1,p_2,l+_1, l+_2, l-_1, l-_2, r+_1, r+_2, r-_1, r-_2, d+_1, d+_2, d-_1, d-_2, u+_1, u+_2, u-_1, u-_2}
    The subscript indices are Majorana mode indices here."""

    def __init__(self, cfg: U1System2DConfig):
        super().__init__(cfg)

        # Change the gaugemgr
        self.gaugemgr = gauge.ZNGauge(3)

        # Change the way the electric energy is calculated
        self.use_pfaffian = False

    def eval_tmat_symb_single(self, paramvec):
        tmat_eval = self.cfg.compute_tmat_symb_single().evalf(
            subs={self.symbolvec[i]: paramvec[i] for i in range(len(paramvec))}
        )
        return np.asarray(tmat_eval).astype(complex)

    def permutation_dirac(self):
        perm_single = np.zeros((9, 9))
        perm_single[0, 0] = 1
        perm_single[1, 1] = 1
        perm_single[2, 5] = 1
        perm_single[3, 6] = 1
        perm_single[4, 2] = 1
        perm_single[5, 4] = 1
        perm_single[6, 8] = 1
        perm_single[7, 7] = 1
        perm_single[8, 3] = 1
        # We have to permute the non-daggered and the daggered modes
        dest = np.kron(np.eye(2), perm_single)
        return dest

    @property
    def gamma_dirac_layervec_sitevec(self):
        """Return the vector of covariance matrices in dirac modes.

        Returns:
            [np.array]: Vector of covariance matrices in Dirac modes
        """
        if self._gamma_dirac_layervec_sitevec is None:

            perm = self.permutation_dirac()
            self._gamma_dirac_layervec_sitevec = []
            for lay in range(self.cfg.nlayer):
                gamma_dirac_lay = [
                    perm
                    @ xnp.array(utils.tmat_to_covariance_matrix(tmat))
                    @ np.transpose(perm)
                    for tmat in self.tmat_layervec_sitevec[lay]
                ]
                self._gamma_dirac_layervec_sitevec.append(gamma_dirac_lay)

            self._gamma_dirac_layervec_sitevec = xnp.array(
                self._gamma_dirac_layervec_sitevec
            )
        return self._gamma_dirac_layervec_sitevec

    def _expand_gamma_maj_to_system(self, covmats_layervec_sitevec):
        # To support non translationally-invariant systems, it would be necessary to use
        # covmats_layervec_sitevec to handle different values on different sites.
        # The U1 ansatz does not support this at the moment, so we just use the first site
        site = 0
        covmats = [
            covmats_layervec_sitevec[lay][site] for lay in range(self.cfg.nlayer)
        ]

        vec = []
        for covmat in covmats:
            permbuilder = lat.PermutationBuilderGMS2DU1(
                self.cfg.lattice, nmodes_per_link=2
            )
            mat_perm = permbuilder.perm()
            nsites = self.cfg.lattice.size
            id = np.eye(nsites)
            # Extract the parts of the covariance matrix
            # The 2 is the number of physical fermionic Majorana modes
            amat, bmat, dmat = extract_partial_covmats(covmat, 2)
            # Expand them
            amat_sys = np.kron(id, amat)
            bmat_sys = np.kron(id, bmat)
            dmat_sys = np.kron(id, dmat)
            # Reassemble them in the correct order
            mat_sys_unordered = np.block(
                [[amat_sys, bmat_sys], [-np.transpose(bmat_sys), dmat_sys]]
            )
            dest = (
                mat_perm @ mat_sys_unordered @ np.transpose(mat_perm)
            )  # Note that this uses a different permutation matrix convention than in Z2 case.
            vec.append(dest)
        return np.array(vec)

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

        For a 2x2 system, gamma_in has the order {l_1, r_0, l_0, r_1, l_3, r_2, l_2, r_3, d_2, u_0, d_0, u_2, d_3, u_1, d_1, d_3}.
        The modes are named as <mode letter>_<vertex site>. Each constitent in the list above labels two Majorana modes.
        """
        # TODO: Fix description

        size = self.cfg.lattice.size  # number of sites

        # Initialize gamma_in_sys for the full system
        # In the U1 parametrization, the direction of the link does not matter for the projector.
        # We just keep the same structure as in the Z2 parametrization for consistency
        id = np.eye(size)
        neutral_gauge_X = np.kron(
            id, self.gamma_gauge_neutral_vec[0][Direction.X]
        )  # just use the first gamma_gauge_neutral, since they're shared by all layers
        neutral_gauge_Y = np.kron(id, self.gamma_gauge_neutral_vec[0][Direction.Y])
        gamma_in_sys = block_diag(
            neutral_gauge_X, neutral_gauge_Y
        )  # for the 3D case, simply add in the Z covariance matrix as well

        diffvec = [mat_d_inv - gamma_in_sys for mat_d_inv in self.mat_d_inv_vec]
        wi_gamma_in_vec = [utils.WoodburyInverter(diff) for diff in diffvec]
        wi_gamma_out_vec = [
            utils.WoodburyInverter(mat_d - gamma_in_sys) for mat_d in self.mat_d_vec
        ]
        incdet_vec = [utils.IncLogAbsDeterminant(diff) for diff in diffvec]

        # Initialize the modified gamma_in_sys for the full system (and trackers)
        single_link_offset = 2 * self.cfg.nvirtmodes_link
        gamma_in_sys_mod = gamma_in_sys[single_link_offset:, single_link_offset:]
        diffvec_mod = [
            mat_d_inv - gamma_in_sys_mod for mat_d_inv in self.mat_d_mod_inv_vec
        ]
        wi_gamma_in_mod_vec = [utils.WoodburyInverter(diff) for diff in diffvec_mod]
        wi_gamma_out_mod_vec = [
            utils.WoodburyInverter(mat_d - gamma_in_sys_mod)
            for mat_d in self.mat_d_mod_vec
        ]
        incdet_mod_vec = [utils.IncLogAbsDeterminant(diff) for diff in diffvec_mod]

        # Though for this ansatz gamma_in_sys does not vary between layers, it is convenient to have gamma_in_sys_vec available as a vector with length = nlayers
        # for general methods in system base
        gamma_in_sys_vec = [gamma_in_sys] * self.cfg.nlayer

        return (
            gamma_in_sys_vec,
            (wi_gamma_in_vec, wi_gamma_out_vec, incdet_vec),
            (wi_gamma_in_mod_vec, wi_gamma_out_mod_vec, incdet_mod_vec),
        )

    ################## Local Gauge ######################

    def _generate_rotmat_half(self, theta):
        rot_right = np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )
        # We have only one left mode => 2 Majorana modes
        rot_left = np.eye(2)
        # The mode order is lr (horizontally) or du (vertically).
        dest = block_diag(rot_left, rot_right)
        return dest

    def generate_rotmat(self, theta, coord, dir):
        gauge_field = theta * pow(-1, np.sum(coord))
        rot_plus = self._generate_rotmat_half(gauge_field)
        rot_minus = self._generate_rotmat_half(-gauge_field)
        return block_diag(rot_plus, rot_minus)

    def update_gauge_ind(self, link_ind, theta):
        # Update the gaugefield
        if ggpeps.PREFERRED_BACKEND == "jax":
            self._gaugefieldvec = self._gaugefieldvec.at[link_ind].set(theta)
        else:
            self._gaugefieldvec[link_ind] = theta
        # There are two directions per vertex
        ind_mat = 2 * self.cfg.nvirtmodes_link * link_ind
        coord, dir = self.cfg.lattice.ind2coord_dir(link_ind)
        rotmat = self.generate_rotmat(theta, coord, dir)
        gamma_in_subst = (
            rotmat @ self.gamma_gauge_neutral_vec[0][dir] @ np.transpose(rotmat)
        )  # just use the first gamma_gauge_neutral, since they're shared by all layers
        update = self.calculate_update_gamma_in(ind_mat, gamma_in_subst)
        # Update the determinant
        mat_inv_vec = [wi_gamma_in.inv() for wi_gamma_in in self.wi_gamma_in_vec]
        detval_vec = [
            incdet.update_index(mat_inv, update, ind_mat, ind_mat)
            for mat_inv, incdet in zip(mat_inv_vec, self.incdet_vec)
        ]
        # Update the modified determinant
        offset = 2 * self.cfg.nvirtmodes_link
        if ind_mat - offset >= 0:
            for wi, incdet in zip(self.wi_gamma_in_mod_vec, self.incdet_mod_vec):
                mat_inv = wi.inv()
                incdet.update_index(mat_inv, update, ind_mat - offset, ind_mat - offset)
        # Update the weight
        self.weight = 0.5 * np.sum(detval_vec)
        # Update the matrix inversion
        [
            wi_gamma_in.update_index(update, ind_mat, ind_mat)
            for wi_gamma_in in self.wi_gamma_in_vec
        ]
        [
            wi_gamma_out.update_index(update, ind_mat, ind_mat)
            for wi_gamma_out in self.wi_gamma_out_vec
        ]

        if ind_mat - offset >= 0:
            # We do not update the matrix if the first link is updated (it is just not there)
            [
                wi_gamma_in_mod.update_index(update, ind_mat - offset, ind_mat - offset)
                for wi_gamma_in_mod in self.wi_gamma_in_mod_vec
            ]
            [
                wi_gamma_out_mod.update_index(
                    update, ind_mat - offset, ind_mat - offset
                )
                for wi_gamma_out_mod in self.wi_gamma_out_mod_vec
            ]
        # Substitute in the array
        self.gamma_in_sys[
            ind_mat : ind_mat + rotmat.shape[0], ind_mat : ind_mat + rotmat.shape[1]
        ] = gamma_in_subst
        # Invalidate gauge dependent quantities
        self.invalidate_gauge_update()

    ################## Observables ######################
    def _compute_mass_energy_op_vec_and_grad(self, use_trans_inv=True):
        raise NotImplementedError("The mass term has not yet been implemented for U1.")

        dest, dest_grad = 0, 0  # Needs to be calculated properly
        return dest, dest_grad

    def _compute_mag_energy_op(self, use_trans_inv=True):
        if use_trans_inv:
            # Evaluate one plaquette and multiply by number of plaquettes
            wilson_plaquette = self.cfg.lattice.generate_wilson_loop((0, 0), (1, 1))
            mag_energy_bare = np.real(self.compute_path(wilson_plaquette))
        else:
            # Evaluate every plaquette of the system
            logger.error("compute_mag_energy: not implemented yet")
            mag_energy_bare = None
        return mag_energy_bare

    def _compute_el_energy_op_and_grad_gaussian(self, use_trans_inv=True):
        if use_trans_inv:
            lognormvec_default_inc = self.calculate_lognormvec_inc(all_factors=True)
            # This is the usual norm without any modifications
            lognorm_default = np.sum(lognormvec_default_inc)
            # Number of fermions = # of sites
            # Since we have 1 copy, we get 2 virtual fermions per link, leading to 2 * 2 Majorana modes
            single_link_offset = 2 * self.cfg.nvirtmodes_link
            offset = 2 * self.cfg.lattice.size + single_link_offset
            # We have to cut one link from gamma_in_sys as well
            gamma_in_sys_mod = self.gamma_in_sys_mod
            nlinks = self.cfg.lattice.nlinks
            dest = []
            dest_grad = []

            for layerind in range(self.cfg.nlayer):
                layer_derivative = []
                # We shift the first virtual link (0,0,X) towards the physical modes to trace out everything else
                # The shifted matrices are extracted at the initalization
                # The offset is changed such that one virtual link is attributed to the physical part
                mat_a = self.mat_a_mod_vec[layerind]
                mat_b = self.mat_b_mod_vec[layerind]
                diff_d_gamma_inv = self.wi_gamma_out_mod_vec[layerind].inv()
                diff_d_inv_gamma_inv = self.wi_gamma_in_mod_vec[layerind].inv()

                ###################### Calculation of <P> ########################
                covmat_out = mat_a + mat_b @ self.wi_gamma_out_mod_vec[
                    layerind
                ].inv() @ np.transpose(mat_b)
                covmat_out_virt = covmat_out[-single_link_offset:, -single_link_offset:]
                # For the modified norm, we still have to take into account the other contributions from the unmodified parts
                norm_mod = calculate_lognorm_inc(
                    [self.incdet_mod_vec[layerind]],
                    [self.det_mat_d_mod_vec[layerind]],
                    gamma_in_sys_mod.shape[0],
                    all_factors=True,
                )
                # norm_mod = calculate_lognorm(gamma_in_sys_mod, [mat_d],
                # all_factors=True)
                norm_mod += np.sum(
                    utils.select_except(lognormvec_default_inc, layerind)
                )
                # The matrix elements yield only the real part of <P>
                # el_energy_layer = 0.25*( covmat_out_virt[0, 1] + covmat_out_virt[2, 3] + 1.j*covmat_out_virt[0,2] - 1.j*covmat_out_virt[0,3]) * np.exp(norm_mod - lognorm_default)
                el_energy_layer = (
                    0.25
                    * (covmat_out_virt[0, 1] + covmat_out_virt[2, 3])
                    * np.exp(norm_mod - lognorm_default)
                )
                dest.append(el_energy_layer)

                ###################### Calculation of the derivative ########################
                for symbol in self.symbolvec:
                    deriv_gamma_maj_sys = self.gamma_maj_sys_deriv_vec(symbol)[layerind]
                    d_mat_a, d_mat_b, d_mat_d = extract_partial_covmats(
                        deriv_gamma_maj_sys, offset
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
                    d_el_energy = (
                        0.25
                        * (d_covmat_out_virt[0, 1] + d_covmat_out_virt[2, 3])
                        * np.exp(norm_mod - lognorm_default)
                    )
                    # Summand with derivative of norms
                    trace_def = self.compute_grad_over_norm(symbol, layerind)
                    trace_mod = compute_grad_over_norm(
                        gamma_in_sys_mod,
                        diff_d_inv_gamma_inv,
                        d_mat_d,
                        self.mat_d_mod_inv_vec[layerind],
                    )
                    d_el_energy += dest[layerind] * (trace_mod - trace_def)
                    # Scale to system size
                    d_el_energy *= nlinks
                    layer_derivative.append(d_el_energy)
                dest_grad.append(layer_derivative)
            # We have to weight the different layers with the electric energy operator expectation of the other layers.
            # They act as a prefactor in the derivative
            dest = np.asarray(dest)
            dest_grad = np.asarray(dest_grad)
            if self.cfg.nlayer > 1:
                for i in range(self.cfg.nlayer):
                    prod_other_layers = utils.multiply_except(dest, i)
                    dest_grad[i] *= prod_other_layers
        else:
            # Evaluate every link of the system
            logger.error("compute_el_energy: not implemented yet")
            dest = np.asarray([None] * self.cfg.nlayer)
            dest_grad = np.asarray([[None] * len(self.symbolvec)] * self.cfg.nlayer)
        return dest, dest_grad

    def construct_gamma_in_sys_electric(self, coord, dir):
        link_ind = self.cfg.lattice.coord2ind_dir(coord, dir)
        current_phase = self.gaugefieldvec[link_ind]
        increment = -self.gaugemgr.get_increment()
        dest = self.gamma_in_sys.astype(complex).copy()
        adapted_no_gauge = self.generate_electric_full(increment)
        rotmat = self.generate_rotmat(current_phase, coord, dir)
        adapted = rotmat @ adapted_no_gauge @ rotmat.transpose()
        ind_mat = 2 * self.cfg.nvirtmodes_link * link_ind
        dest[
            ind_mat : ind_mat + adapted.shape[0], ind_mat : ind_mat + adapted.shape[1]
        ] = adapted
        return dest

    def generate_electric_single_mode(self, phi):
        # This function outputs a matrix in the order r(1),r(2),l(1),l(2)
        # The convention is that phi is taken as positive for the modes r+ and l-,
        # and negative for r- and l+
        t = np.tan(phi / 2.0)
        dest = np.array(
            [
                [0, -1.0j * t, -t, -1],
                [1.0j * t, 0, -1, t],
                [t, 1, 0, -1.0j * t],
                [1, -t, 1.0j * t, 0],
            ],
            dtype=complex,
        )
        return dest

    def generate_electric_full(self, phi):
        # Mode order of the matrix before reordering:r+,l-,r-,l+
        part = np.zeros((8, 8), dtype=complex)
        part[0:4, 0:4] = self.generate_electric_single_mode(phi)
        part[4:, 4:] = self.generate_electric_single_mode(-phi)
        # TODO: We can also insert the matrix elements already in the correct order
        # Mode order of the matrix after reordering:l+,l-,r+,r- (as needed for Gamma_in)
        perm_electric = np.zeros((8, 8))
        id = np.eye(2)
        perm_electric[0:2, 6:8] = id
        perm_electric[2:4, 2:4] = id
        perm_electric[4:6, 0:2] = id
        perm_electric[6:8, 4:6] = id
        return perm_electric @ part @ np.transpose(perm_electric)

    def _compute_el_energy_op_and_grad_pfaffian(self, use_trans_inv=True):
        # Store the current value of the overlap
        if use_trans_inv:
            dest = []
            increment = -self.gaugemgr.get_increment()
            prefactor = 0.5 * (1.0 + np.cos(increment))
            gamma_in_try = self.construct_gamma_in_sys_electric((0, 0), lat.Direction.X)
            # Build the new value for gamma_in
            # Since there can be singular matrices in the update, we can't use the
            # Determinant Lemma
            for i in range(self.cfg.ncopy):
                mat_d_inv = self.mat_d_inv_vec[i]
                # The 0.5 is the square root since incdet stores the log of the determinant
                overlap_same_gauge = np.exp(0.5 * self.incdet_vec[i].det())

                diff_try = gamma_in_try - mat_d_inv
                overlap_diff_gauge = pf.pfaffian(
                    np.array(diff_try)
                )  # When using jax, this line produces garbage unless diff_try is first cast to numpy, which causes a test to fail. TODO: investigate why
                dest.append(
                    prefactor * np.real(overlap_diff_gauge) / overlap_same_gauge
                )
            # TODO: Implement gradient
            dest_grad = np.asarray([[None] * len(self.symbolvec)] * self.cfg.nlayer)
        else:
            # Evaluate every link of the system
            logger.error("compute_el_energy: not implemented yet")
            dest = np.asarray([None] * self.cfg.nlayer)
            dest_grad = np.asarray([[None] * len(self.symbolvec)] * self.cfg.nlayer)
        return dest, dest_grad

    def _compute_el_energy_op_vec_and_grad(self):
        if self.use_pfaffian:
            return self._compute_el_energy_op_and_grad_pfaffian()
        else:
            return self._compute_el_energy_op_and_grad_gaussian()

    def _compute_int_energy_op_vec_and_grad(self):
        # This function is not implemented yet!
        raise NotImplementedError(
            "The interaction energy is not implemented yet for U(1)."
        )

    def _compute_chem_energy_op_vec_and_grad(self):
        """Calculate the chemical potential energy operator and its gradient."""
        raise NotImplementedError(
            "The chemical potential energy is not implemented yet for U(1)."
        )
