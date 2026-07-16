import unittest
from unittest import skip

import numpy as np
import sympy as sp
from scipy.linalg import block_diag

from ggpeps import gauge, lattice, utils
from ggpeps import system, exacteval
from ggpeps.modearray import generate_permutation_matrix
from ggpeps.system.config_base import generate_gauged_projector_terms

# ======================= D6 fermionic system (2 copies) =======================


class TestD2nSystem(unittest.TestCase):
    """Class for testing ansatz with matter, and 2 virtual copies per layer"""

    def setUp(self):

        lat = lattice.Lattice2D(2, 2)
        num_pg_layer = 2
        num_fermionic_layer = 0
        nlayer = num_pg_layer + num_fermionic_layer
        unitcell_size = 1
        paramvec = np.random.rand(nlayer, unitcell_size, 20)
        cfg = system.D6System2D_Config(
            lat,
            1,
            1,
            0,
            0,
            None,
            ncopy=2,
            num_pg_layer=num_pg_layer,
            num_fermionic_layer=num_fermionic_layer,
        )
        cfg.paramvec = paramvec
        self.system_D6 = system.D2nSystem2D(cfg)
        self.system_D6.cfg.enforce_parameter_conditions(self.system_D6.cfg.paramvec)

    def test_required_params_are_zero(self):
        """Ensure that the parameters that must vanish to guarantee ansatz symmetries do indeed vanish."""
        mat = self.system_D6.cfg.paramvec
        t_indices = [0, 3, 10, 13]  # index of t1r, t2r, t1i, t2i in symbolvec
        for layer_ind in range(self.system_D6.cfg.num_pg_layer):
            for uc_ind in range(self.system_D6.cfg.unitcell_size):
                for t_ind in t_indices:
                    with self.subTest(tind=t_ind, layerind=layer_ind):
                        coord = (layer_ind, uc_ind, t_ind)
                        self.assertAlmostEqual(mat[coord], 0)

        zero_for_fermionic_layer = [
            3,
            13,
            1,
            2,
            4,
            5,
            11,
            12,
            14,
            15,
        ]  # index of t2r, t2i, y1r, z1r, y2r, z2r, y1i, z1i, y2i, z2i in symbolvec
        for layer_ind in range(self.system_D6.cfg.num_pg_layer, self.system_D6.cfg.nlayer):
            for uc_ind in range(self.system_D6.cfg.unitcell_size):
                for ind in zero_for_fermionic_layer:
                    with self.subTest(ind=ind, layerind=layer_ind):
                        coord = (layer_ind, uc_ind, ind)
                        self.assertAlmostEqual(mat[coord], 0)

    def test_covmat_for_no_fermions(self):
        """Ensure the correct covariance matrix is generated when t = 0."""
        self.system_D6.cfg.make_pure_gauge()
        covmat_layer1 = self.system_D6.ferm_covmat_vec[0]
        covmat_layer2 = self.system_D6.ferm_covmat_vec[1]
        expected_covmat = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            ]
        )
        expected_covmat = block_diag(expected_covmat, expected_covmat)
        self.assertTrue(np.allclose(covmat_layer1, expected_covmat))
        self.assertTrue(np.allclose(covmat_layer2, expected_covmat))

    @skip("This test needs adjusments and expected to pass after implementing physical fermionic layers.")
    def test_covmat_with_fermions(self):
        """Ensure the covariance matrix is not the pure-gauge one when t != 0.
        This test must be done with a gauge configuration that includes some flux.
        Only the fermionic layer should have a covariance matrix different than the pure-gauge one.
        """  # TODO: Fix after implementing physical fermionic layers.
        identity = self.system_D6.cfg.gaugemgr.get_neutral_gauge_value()
        other_gauge = self.system_D6.cfg.gaugemgr.get_representaion(0, 1)
        config = np.array([identity] * 7 + [other_gauge] * 1)
        self.system_D6.update_gauge_full_system(config)

        covmat_layer1 = self.system_D6.ferm_covmat_vec[0]
        covmat_layer2 = self.system_D6.ferm_covmat_vec[1]
        expected_covmat = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            ]
        )
        expected_covmat = block_diag(expected_covmat, expected_covmat)
        self.assertTrue(np.allclose(covmat_layer1, expected_covmat))
        self.assertFalse(np.allclose(covmat_layer2, expected_covmat))

    def test_valid_covmat(self):
        """Ensure the covariance matrix satisfies the conditions to be a covariance matrix.
        This test is done with a gauge configuration that includes some flux.
        """
        identity = self.system_D6.cfg.gaugemgr.get_neutral_gauge_value()
        other_gauge = self.system_D6.cfg.gaugemgr.get_representation(0, 1)
        config = np.array([identity] * 7 + [other_gauge] * 1)
        self.system_D6.update_gauge_full_system(config)

        covmat_layer1 = self.system_D6.ferm_covmat_vec[0]
        # covmat_layer2 = self.system_D6.ferm_covmat_vec[1]
        self.assertTrue(utils.is_covmat(covmat_layer1))
        # self.assertTrue(utils.is_covmat(covmat_layer2))

    def test_valid_gamma_in_sys(self):
        """Ensure the gamma_sys matrix satisfies the conditions to be a covariance matrix."""
        for lay in range(self.system_D6.cfg.nlayer):
            gamma_in_sys = self.system_D6.gamma_in_sys_vec[lay]
            self.assertTrue(utils.is_covmat(gamma_in_sys))

    @skip("This test needs adjusments and expected to pass after implementing physical fermionic layers.")
    def test_t_zero(self):
        """Ensure mass and interaction energy are zero when t = 0"""
        # TODO: Fix after implementing physical fermionic layers.
        ec_config = exacteval.ExactEvaluatorConfig()
        ec_config.gauge_fixing = False
        self.system_D6.cfg.make_pure_gauge()  # sets t params to zero
        ex_eval = exacteval.ExactEvaluator(ec_config, self.system_D6)
        dest_dict = ex_eval.evaluate()
        self.assertTrue(np.allclose(0, dest_dict["mass_energy"]))
        self.assertTrue(np.allclose(0, dest_dict["int_energy"]))

    @skip("This test needs adjusments and expected to pass after implementing physical fermionic layers.")
    def test_t_nonzero(self):
        """Ensure mass and interaction energy are zero when t != 0.
        This checks for random params, which we assume do not give t = 0"""
        # TODO: Fix after implementing physical fermionic layers.
        ec_config = exacteval.ExactEvaluatorConfig()
        ec_config.gauge_fixing = False
        ex_eval = exacteval.ExactEvaluator(ec_config, self.system_z2)
        dest_dict = ex_eval.evaluate()
        self.assertFalse(np.allclose(0, dest_dict["mass_energy"]))
        self.assertFalse(np.allclose(0, dest_dict["int_energy"]))

    def test_Tmat_symmetries_analytic(self):
        """This only tests rotation invariance and the antisymmetry properties."""

        # rotation invariance
        # mode order: lrdu
        eta = sp.exp(1j * sp.pi / 4)
        R = eta * sp.Matrix(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )
        R = block_diag(R, R)  # block diagonal of 2 colors
        permutation_mat = np.array(
            # Reorder R from the BlockDiag(color1, color2) source order into the
            # tmat mode order (color-outer):
            #   Psi_1,Psi_2, l1_1,r1_1,d1_1,u1_1, l2_1,r2_1,d2_1,u2_1,
            #                l1_2,r1_2,d1_2,u1_2, l2_2,r2_2,d2_2,u2_2
            # (modes labelled direction{copy}_{color}, same as config_D6_2d correct_order)
            generate_permutation_matrix(
                list(range(1, 19)),
                [1, 10, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18],
            )
        )
        R = np.transpose(permutation_mat) @ R @ permutation_mat
        tmat = self.system_D6.cfg.tmat_symb
        res_rot = R.T @ tmat @ R - tmat
        res = sp.simplify(sp.simplify(res_rot))  # for some reason, two passes are needed
        self.assertFalse(any(res))

        res = sp.simplify(tmat + tmat.T)
        self.assertFalse(any(res))

    def test_Tmat_symmetries_numeric(self):
        """This only tests rotation invariance and the antisymmetry properties."""

        # rotation invariance
        # mode order: lrdu
        eta = np.exp(1j * np.pi / 4)
        R = eta * np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )
        R = block_diag(R, R)  # block diagonal of 2 colors
        permutation_mat = np.array(
            # Reorder R from the BlockDiag(color1, color2) source order into the
            # tmat mode order (color-outer):
            #   Psi_1,Psi_2, l1_1,r1_1,d1_1,u1_1, l2_1,r2_1,d2_1,u2_1,
            #                l1_2,r1_2,d1_2,u1_2, l2_2,r2_2,d2_2,u2_2
            # (modes labelled direction{copy}_{color}, same as config_D6_2d correct_order)
            generate_permutation_matrix(
                list(range(1, 19)),
                [1, 10, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18],
            )
        )
        R = np.transpose(permutation_mat) @ R @ permutation_mat

        tmats = self.system_D6.tmat_layervec_sitevec
        for lay in range(self.system_D6.cfg.nlayer):
            for site in range(self.system_D6.cfg.lattice.size):
                tmat = tmats[lay][site]
                res_rot = R.T @ tmat @ R - tmat
                self.assertTrue(np.allclose(res_rot, 0))

                res = tmat + tmat.T
                self.assertTrue(np.allclose(res, 0))

    def test_free_fermions_gs_energy(self):
        """Ensure gs energy for free fermion case matches the analytic result"""
        pass

    # TODO: TESTS TO ADD
    # get the correct gamma_in_sys for all layers
    # when interaction is off, ground state is: no fermions, pure-gauge ground state

    # test site-specific mass values in limits where this is known

    # T-mat has required structure
    # test different number of pg layers

    # random parameters comparison with ED

    ###### Test Energy Gradients ######
    def test_grad_el_energy_2C(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # for the 2 copy fermionic ansatz
        eps = 1e-5
        paramvec = np.random.rand(2, 20)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.D6System2D_Config(
            lat_2x2,
            1.0,
            0.0,
            0.0,
            0.0,
            None,
            ncopy=2,
        )
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.D2nSystem2D(system_cfg)
        deriv_ana = system_z2_2_2.el_energy_op_grad_vec
        symbolvec = system_z2_2_2.symbolvec

        uc_ind = 0
        idxs = np.random.randint(0, len(symbolvec), size=2)

        for layerind in range(2):
            for ind in idxs:
                with self.subTest(symbol=symbolvec[ind], layerind=layerind):
                    paramvec_left = np.copy(paramvec)
                    paramvec_right = np.copy(paramvec)
                    paramvec_left[layerind, ind] -= eps
                    paramvec_right[layerind, ind] += eps
                    system_cfg_left = system.D6System2D_Config(
                        lat_2x2,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        None,
                        ncopy=2,
                    )
                    system_cfg_right = system.D6System2D_Config(
                        lat_2x2,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        None,
                        ncopy=2,
                    )

                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    system_z2_2_2_left = system.D2nSystem2D(system_cfg_left)
                    system_z2_2_2_right = system.D2nSystem2D(system_cfg_right)

                    val_left = system_z2_2_2_left.el_energy_op
                    val_right = system_z2_2_2_right.el_energy_op
                    deriv_num = (val_right - val_left) / (2 * eps)

                    # print(f"left: {val_left}, right: {val_right}")
                    # print(f"symbol: {symbolvec[ind]}, analytic: {deriv_ana[layerind,ind]}, numerical: {deriv_num}")
                    self.assertAlmostEqual(deriv_ana[layerind, uc_ind, ind], deriv_num, places=5)

    @skip("This test needs adjusments and expected to pass after implementing mass energy.")
    def test_grad_mass_energy_2C(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # for the 2 copy fermionic ansatz
        eps = 1e-5
        paramvec = np.random.rand(2, 20)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_Config(lat_2x2, 0.0, 0.0, 0.0, 1.0, None, ncopy=2)
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(system_cfg)

        config = np.array([0] * 7 + [np.pi] * 1)
        system_z2_2_2.update_gauge_full_system(config)

        deriv_ana = system_z2_2_2.mass_energy_op_grad_vec
        symbolvec = system_z2_2_2.symbolvec

        uc_ind = 0

        for layerind in range(2):
            # we could skip the first layer, since the first layer does not contribute to the
            # mass energy
            for ind in range(len(symbolvec)):
                with self.subTest(symbol=symbolvec[ind], layerind=layerind):
                    paramvec_left = np.copy(paramvec)
                    paramvec_right = np.copy(paramvec)
                    paramvec_left[layerind, ind] -= eps
                    paramvec_right[layerind, ind] += eps
                    system_cfg_left = system.Z2System2D_Config(lat_2x2, 0.0, 0.0, 1.0, 1.0, None, ncopy=2)
                    system_cfg_right = system.Z2System2D_Config(lat_2x2, 0.0, 0.0, 1.0, 1.0, None, ncopy=2)

                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    system_z2_2_2_left = system.Z2System2D(system_cfg_left)
                    system_z2_2_2_right = system.Z2System2D(system_cfg_right)
                    system_z2_2_2_left.update_gauge_full_system(config)
                    system_z2_2_2_right.update_gauge_full_system(config)

                    val_left = system_z2_2_2_left.mass_energy_op
                    val_right = system_z2_2_2_right.mass_energy_op
                    deriv_num = (val_right - val_left) / (2 * eps)

                    # print(f"left: {val_left}, right: {val_right}")
                    # print(f"symbol: {symbolvec[ind]}, analytic: {deriv_ana[layerind,ind]}, numerical: {deriv_num}")
                    self.assertAlmostEqual(deriv_ana[layerind, uc_ind, ind], deriv_num, places=5)

    @skip("This test needs adjusments and expected to pass after implementing mass energy.")
    def test_grad_mass_energy_2flavor(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # for the 2 copy fermionic ansatz with 2 physical flavors
        eps = 1e-5
        paramvec = np.random.rand(3, 20)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_Config(
            lat_2x2, 0.0, 0.0, 0.0, 1.0, None, num_pg_layer=1, num_fermionic_layer=2,
            ncopy=2,
        )
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(system_cfg)

        config = np.array([0] * 7 + [np.pi] * 1)
        system_z2_2_2.update_gauge_full_system(config)

        deriv_ana = system_z2_2_2.mass_energy_op_grad_vec
        symbolvec = system_z2_2_2.symbolvec

        uc_ind = 0

        for layerind in range(3):
            # we could skip the first layer, since the first layer does not contribute to the
            # mass energy
            for ind in range(len(symbolvec)):
                with self.subTest(symbol=symbolvec[ind], layerind=layerind):
                    paramvec_left = np.copy(paramvec)
                    paramvec_right = np.copy(paramvec)
                    paramvec_left[layerind, ind] -= eps
                    paramvec_right[layerind, ind] += eps
                    system_cfg_left = system.Z2System2D_Config(
                        lat_2x2,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        None,
                        num_pg_layer=1,
                        num_fermionic_layer=2,
                        ncopy=2,
                    )
                    system_cfg_right = system.Z2System2D_Config(
                        lat_2x2,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        None,
                        num_pg_layer=1,
                        num_fermionic_layer=2,
                        ncopy=2,
                    )

                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    system_z2_2_2_left = system.Z2System2D(system_cfg_left)
                    system_z2_2_2_right = system.Z2System2D(system_cfg_right)
                    system_z2_2_2_left.update_gauge_full_system(config)
                    system_z2_2_2_right.update_gauge_full_system(config)

                    val_left = system_z2_2_2_left.mass_energy_op
                    val_right = system_z2_2_2_right.mass_energy_op
                    deriv_num = (val_right - val_left) / (2 * eps)

                    # print(f"left: {val_left}, right: {val_right}")
                    # print(f"symbol: {symbolvec[ind]}, analytic: {deriv_ana[layerind,ind]}, numerical: {deriv_num}")
                    self.assertAlmostEqual(deriv_ana[layerind, uc_ind, ind], deriv_num, places=5)

    @skip("This test needs adjusments and expected to pass after implementing intercation energy.")
    def test_grad_int_energy_2C(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # for the 2 copy fermionic ansatz
        eps = 1e-5
        paramvec = np.random.rand(2, 20)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_Config(
            lat_2x2, 0.0, 0.0, 1.0, 0.0, None, num_pg_layer=1, num_fermionic_layer=1,
            ncopy=2,
        )
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(system_cfg)

        # the interaction energy vanishes for the default configuration (no flux on any link)
        # so we choose a configuration where we know the interaction energy is not negligible
        config = np.array([0] * 7 + [np.pi] * 1)
        system_z2_2_2.update_gauge_full_system(config)

        deriv_ana = system_z2_2_2.int_energy_op_grad_vec
        symbolvec = system_z2_2_2.symbolvec

        uc_ind = 0

        for layerind in range(2):
            # we could skip the first layer, since the first layer does not contribute to the
            # interaction energy
            for ind in range(len(symbolvec)):
                with self.subTest(symbol=symbolvec[ind], layerind=layerind):
                    paramvec_left = np.copy(paramvec)
                    paramvec_right = np.copy(paramvec)
                    paramvec_left[layerind, ind] -= eps
                    paramvec_right[layerind, ind] += eps
                    system_cfg_left = system.Z2System2D_Config(
                        lat_2x2,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        None,
                        num_pg_layer=1,
                        num_fermionic_layer=1,
                        ncopy=2,
                    )
                    system_cfg_right = system.Z2System2D_Config(
                        lat_2x2,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        None,
                        num_pg_layer=1,
                        num_fermionic_layer=1,
                        ncopy=2,
                    )

                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    system_z2_2_2_left = system.Z2System2D(system_cfg_left)
                    system_z2_2_2_right = system.Z2System2D(system_cfg_right)
                    system_z2_2_2_left.update_gauge_full_system(config)
                    system_z2_2_2_right.update_gauge_full_system(config)

                    val_left = system_z2_2_2_left.int_energy_op
                    val_right = system_z2_2_2_right.int_energy_op
                    deriv_num = (val_right - val_left) / (2 * eps)

                    # print(f"left: {val_left}, right: {val_right}")
                    # print(f"symbol: {symbolvec[ind]}, analytic: {deriv_ana[layerind,ind]}, numerical: {deriv_num}")
                    self.assertAlmostEqual(deriv_ana[layerind, uc_ind, ind], deriv_num, places=5)

    @skip("This test needs adjusments and expected to pass after implementing chmical energy.")
    def test_grad_chem_energy_2flavor(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # for the 2 copy fermionic ansatz with 2 physical flavors
        eps = 1e-5
        g_chem = [0, -0.4, 2]
        paramvec = np.random.rand(3, 20)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_Config(
            lat_2x2,
            0.0,
            0.0,
            0.0,
            0.0,
            g_chem,
            num_pg_layer=1,
            num_fermionic_layer=2,
            ncopy=2,
        )
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(system_cfg)

        config = np.array([0] * 7 + [np.pi] * 1)
        system_z2_2_2.update_gauge_full_system(config)

        uc_ind = 0

        deriv_ana = system_z2_2_2.chem_energy_op_grad_vec
        # Scale the gradients by the appropriate chemical potential
        for lay in range(3):
            deriv_ana[lay, :, :] *= g_chem[lay]
        symbolvec = system_z2_2_2.symbolvec
        for layerind in range(3):
            # we could skip the pure gauge layers, since they do not contribute
            for ind in range(len(symbolvec)):
                with self.subTest(symbol=symbolvec[ind], layerind=layerind):
                    paramvec_left = np.copy(paramvec)
                    paramvec_right = np.copy(paramvec)
                    paramvec_left[layerind, ind] -= eps
                    paramvec_right[layerind, ind] += eps
                    system_cfg_left = system.Z2System2D_Config(
                        lat_2x2,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        g_chem,
                        num_pg_layer=1,
                        num_fermionic_layer=2,
                        ncopy=2,
                    )
                    system_cfg_right = system.Z2System2D_Config(
                        lat_2x2,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        g_chem,
                        num_pg_layer=1,
                        num_fermionic_layer=2,
                        ncopy=2,
                    )

                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    system_z2_2_2_left = system.Z2System2D(system_cfg_left)
                    system_z2_2_2_right = system.Z2System2D(system_cfg_right)
                    system_z2_2_2_left.update_gauge_full_system(config)
                    system_z2_2_2_right.update_gauge_full_system(config)

                    val_left = system_z2_2_2_left.chem_energy
                    val_right = system_z2_2_2_right.chem_energy
                    deriv_num = (val_right - val_left) / (2 * eps)

                    self.assertAlmostEqual(deriv_ana[layerind, uc_ind, ind], deriv_num, places=5)

    @skip("This test needs adjusments and mot expected to pass for now.")
    def test_FM(self):

        system_type = system.Z2System2D
        lat = lattice.Lattice2D(2, 2)

        # Compare with ED for g_int = 1, g = 1.1 m = 0
        g_int0 = 1
        g0 = 1.1
        FM_ed = 0.056378472489371945
        param = np.array(
            [
                [
                    0.0,
                    -2.40404381,
                    0.06235486,
                    0.0,
                    -0.03930116,
                    0.02570194,
                    -0.86789621,
                    1.63420534,
                    -0.48050886,
                    0.32365748,
                    0.0,
                    -0.57834489,
                    0.67735453,
                    0.0,
                    -0.00508246,
                    0.06794535,
                    0.93614442,
                    0.62925059,
                    0.19743901,
                    -0.80041006,
                ],
                [
                    3.0479293,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.72591674,
                    0.6554002,
                    0.64491155,
                    0.65708742,
                    1.35546787,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.23331132,
                    1.59756399,
                    1.42258382,
                    1.59441826,
                ],
            ]
        )
        system_cfg = system.Z2System2D_Config(
            lat,
            g0 / 2,
            1 / (2 * g0),
            g_int0,
            0,
            [0, 0],
            num_pg_layer=1,
            num_fermionic_layer=1,
            ncopy=2,
        )
        system_cfg.paramvec = param
        sys = system_type(system_cfg)
        eval_config = exacteval.ExactEvaluatorConfig()
        ex_eval = exacteval.ExactEvaluator(eval_config, sys)
        res = ex_eval.evaluate()
        FM = res["FM_1x1"]
        self.assertAlmostEqual(FM, FM_ed, places=2)


class TestD6TmatColorStructure(unittest.TestCase):
    """Regression tests pinning the *color* structure of the D6 T-matrix.

     Antisymmetry and rotation invariance are already covered by test_Tmat_symmetries_*,
    so they are intentionally not repeated here.

    tmat mode order (config_D6_2d.py, color-outer): indices
        0      = Psi1 (color-1 physical),   1      = Psi2 (color-2 physical)
        2..9   = color-1 virtuals,          10..17 = color-2 virtuals
    """

    COLOR1_IDX = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    COLOR2_IDX = [1, 10, 11, 12, 13, 14, 15, 16, 17]

    def setUp(self):
        lat = lattice.Lattice2D(2, 2)
        num_pg_layer = 1
        num_fermionic_layer = 1  # a fermionic layer keeps t1 (phys-virt coupling) nonzero
        paramvec = np.random.rand(num_pg_layer + num_fermionic_layer, 1, 20)
        cfg = system.D6System2D_Config(
            lat,
            1,
            1,
            0,
            0,
            None,
            ncopy=2,
            num_pg_layer=num_pg_layer,
            num_fermionic_layer=num_fermionic_layer,
        )
        cfg.paramvec = paramvec
        self.system_D6 = system.D2nSystem2D(cfg)
        self.system_D6.cfg.enforce_parameter_conditions(self.system_D6.cfg.paramvec)

    def test_tmat_no_cross_color_coupling(self):
        """T has zero coupling between color-1 and color-2 modes (block-diagonal in color).
        Includes the physical modes: Psi1 must not couple to color-2 virtuals or to Psi2.
        """
        for lay in range(self.system_D6.cfg.nlayer):
            tmat = np.array(self.system_D6.tmat_layervec_sitevec[lay][0])
            cross_block = tmat[np.ix_(self.COLOR1_IDX, self.COLOR2_IDX)]
            with self.subTest(layer=lay):
                self.assertTrue(np.allclose(cross_block, 0))

    def test_tmat_color_blocks_identical(self):
        """The two color blocks (each = its physical mode + its 8 virtual modes) are equal:
        the two colors share the same variational parameters.
        """
        for lay in range(self.system_D6.cfg.nlayer):
            tmat = np.array(self.system_D6.tmat_layervec_sitevec[lay][0])
            block1 = tmat[np.ix_(self.COLOR1_IDX, self.COLOR1_IDX)]
            block2 = tmat[np.ix_(self.COLOR2_IDX, self.COLOR2_IDX)]
            with self.subTest(layer=lay):
                self.assertTrue(np.allclose(block1, block2))

    def test_tmat_physical_virtual_coupling_is_color_local(self):
        """Anti-vacuity guard for the tests above: a fermionic layer must actually couple
        each physical mode to its own color's virtuals (t1 != 0), while a pure-gauge layer
        must not (t1 == 0). Combined with test_tmat_no_cross_color_coupling this confirms
        each physical mode couples to its own color only.
        """
        cfg = self.system_D6.cfg
        for lay in range(cfg.nlayer):
            tmat = np.array(self.system_D6.tmat_layervec_sitevec[lay][0])
            psi1_to_color1 = tmat[0, 2:10]  # Psi1 -> color-1 virtuals
            with self.subTest(layer=lay):
                if lay < cfg.num_pg_layer:
                    self.assertTrue(np.allclose(psi1_to_color1, 0))
                else:
                    self.assertFalse(np.allclose(psi1_to_color1, 0))


# class TestTransVariance(unittest.TestCase):
#     """Test the ansatz when it is not translationally invariant.
#     This class only tests the case when even/odd sublattices have different parameters,
#     but parameters are the same within each sublattice.
#     Many of the tests could be adapted to the more general case."""

#     def setUp(self):

#         lat = lattice.Lattice2D(2, 2)
#         num_pg_layer = 1
#         num_fermionic_layer = 2
#         nlayer = num_pg_layer + num_fermionic_layer
#         unitcell_size = 2
#         paramvec = np.random.rand(nlayer, unitcell_size, 20)
#         cfg = system.Z2System2D_Config(
#             lat,
#             1,
#             1,
#             1,
#             1,
#             None,
#             num_pg_layer=num_pg_layer,
#             num_fermionic_layer=num_fermionic_layer,
#             unitcell_size=unitcell_size,
#             ncopy=2,
#         )
#         cfg.paramvec = paramvec
#         self.system_z2 = system.Z2System2D(cfg)
#         self.system_z2.cfg.enforce_parameter_conditions(self.system_z2.cfg.paramvec)

#     def test_tmat_layervec_sitevec(self):
#         for lay in range(self.system_z2.cfg.nlayer):
#             tmats = self.system_z2.tmat_layervec_sitevec[lay]
#             for site in range(self.system_z2.cfg.lattice.size):
#                 x, y = self.system_z2.cfg.lattice.ind2coord(site)
#                 tm = tmats[site]
#                 if (x + y) % 2:
#                     # odd sublattice - all odd sites should have the same tmat
#                     self.assertTrue(np.allclose(tm, tmats[1]))
#                 else:
#                     # even sublattice - all even sites should have the same tmat
#                     self.assertTrue(np.allclose(tm, tmats[0]))

#             # The tmat for even and odd sites should be different (with high probability for random parameters)
#             self.assertFalse(np.allclose(tmats[0], tmats[1]))

#     def test_gamma_maj_layervec_sitevec(self):
#         for lay in range(self.system_z2.cfg.nlayer):
#             gammas = self.system_z2.gamma_maj_layervec_sitevec[lay]
#             for site in range(self.system_z2.cfg.lattice.size):
#                 x, y = self.system_z2.cfg.lattice.ind2coord(site)
#                 gamma = gammas[site]
#                 if (x + y) % 2:
#                     # odd sublattice - all odd sites should have the same gamma_maj
#                     self.assertTrue(np.allclose(gamma, gammas[1]))
#                 else:
#                     # even sublattice - all even sites should have the same gamma_maj
#                     self.assertTrue(np.allclose(gamma, gammas[0]))

#             # The gamma_maj for even and odd sites should be different (with high probability for random parameters)
#             self.assertFalse(np.allclose(gammas[0], gammas[1]))

#     def test_mat_a_even(self):
#         """If t=0 on a given site, then mat_a should be [[0, 1], [-1, 0]] on that site."""
#         # Set t = 0 on even sites
#         t_inds = [0, 3, 10, 13]  # index of t1r, t2r, t1i, t2i in symbolvec
#         paramvec = self.system_z2.cfg.paramvec
#         for lay in range(self.system_z2.cfg.nlayer):
#             uc_ind = 0  # index for even sites
#             for t_ind in t_inds:
#                 paramvec[lay, uc_ind, t_ind] = 0.0
#         self.system_z2.cfg.paramvec = paramvec

#         target_even = np.array([[0, 1], [-1, 0]])
#         lay = self.system_z2.cfg.num_pg_layer  # the index of the first fermionic layer
#         mat_a = self.system_z2.mat_a_vec[lay]
#         for site in range(self.system_z2.cfg.lattice.size):
#             site_ind = 2 * site
#             mat = mat_a[site_ind : site_ind + 2, site_ind : site_ind + 2]

#             x, y = self.system_z2.cfg.lattice.ind2coord(site)
#             if (x + y) % 2 == 0:
#                 # on even sites (where t was set to zero), mat_a should be target
#                 self.assertTrue(np.allclose(mat, target_even))
#             else:
#                 # on odd sites, mat_a should not be target
#                 self.assertFalse(np.allclose(mat, target_even))

#                 # all the odd sites should still be the same as each other
#                 mat_site_1 = mat_a[2:4, 2:4]
#                 self.assertTrue(np.allclose(mat, mat_site_1))

#         # Check that the off-diagonal blocks are zero
#         for site1 in range(self.system_z2.cfg.lattice.size):
#             ind1 = 2 * site1
#             for site2 in range(self.system_z2.cfg.lattice.size):
#                 ind2 = 2 * site2
#                 if site1 != site2:
#                     block = mat_a[ind1 : ind1 + 2, ind2 : ind2 + 2]
#                     self.assertTrue(np.allclose(block, 0))

#     def test_mat_a_odd(self):
#         """If t=0 on a given site, then mat_a should be [[0, 1], [-1, 0]] on that site.
#         Same as previous test, but for odd sites."""
#         # Set t = 0 on odd sites
#         t_inds = [0, 3, 10, 13]  # index of t1r, t2r, t1i, t2i in symbolvec
#         paramvec = self.system_z2.cfg.paramvec
#         for lay in range(self.system_z2.cfg.nlayer):
#             uc_ind = 1  # index for odd sites
#             for t_ind in t_inds:
#                 paramvec[lay, uc_ind, t_ind] = 0.0
#         self.system_z2.cfg.paramvec = paramvec

#         target_odd = np.array([[0, 1], [-1, 0]])
#         lay = self.system_z2.cfg.num_pg_layer  # the index of the first fermionic layer
#         mat_a = self.system_z2.mat_a_vec[lay]
#         for site in range(self.system_z2.cfg.lattice.size):
#             site_ind = 2 * site
#             mat = mat_a[site_ind : site_ind + 2, site_ind : site_ind + 2]

#             x, y = self.system_z2.cfg.lattice.ind2coord(site)
#             if (x + y) % 2 == 0:
#                 # on even sites (where t was set to zero), mat_a should be target
#                 self.assertFalse(np.allclose(mat, target_odd))

#                 # all the even sites should still be the same as each other
#                 mat_site_0 = mat_a[0:2, 0:2]
#                 self.assertTrue(np.allclose(mat, mat_site_0))
#             else:
#                 # on odd sites, mat_a should not be target
#                 self.assertTrue(np.allclose(mat, target_odd))

#         # Check that the off-diagonal blocks are zero
#         for site1 in range(self.system_z2.cfg.lattice.size):
#             ind1 = 2 * site1
#             for site2 in range(self.system_z2.cfg.lattice.size):
#                 ind2 = 2 * site2
#                 if site1 != site2:
#                     block = mat_a[ind1 : ind1 + 2, ind2 : ind2 + 2]
#                     self.assertTrue(np.allclose(block, 0))

#     def test_mat_b_even(self):
#         """If t=0 on a given site, then mat_b should be all zeros on that site."""
#         # Set t = 0 on even sites
#         t_inds = [0, 3, 10, 13]  # index of t1r, t2r, t1i, t2i in symbolvec
#         paramvec = self.system_z2.cfg.paramvec
#         for lay in range(self.system_z2.cfg.nlayer):
#             uc_ind = 0  # index for even sites
#             for t_ind in t_inds:
#                 paramvec[lay, uc_ind, t_ind] = 0.0
#         self.system_z2.cfg.paramvec = paramvec

#         shape = (2, 2 * self.system_z2.cfg.ncopy * 4)
#         target_even = np.zeros(shape)

#         lay = self.system_z2.cfg.num_pg_layer  # the index of the first fermionic layer
#         mat_b = self.system_z2.mat_b_vec[lay]

#         # Change mat_b to site-based mode order
#         modes_link_order = self.system_z2.get_link_based_mode_order()
#         modes_site_order = self.system_z2.get_site_based_mode_order()
#         mat_perm = generate_permutation_matrix(modes_link_order, modes_site_order)
#         mat_b = mat_b @ mat_perm

#         for site in range(self.system_z2.cfg.lattice.size):
#             site_ind = 2 * site
#             mat = mat_b[site_ind : site_ind + 2, 8 * site_ind : 8 * (site_ind + 2)]

#             x, y = self.system_z2.cfg.lattice.ind2coord(site)
#             if (x + y) % 2 == 0:
#                 # on even sites (where t was set to zero), mat_b should be target
#                 self.assertTrue(np.allclose(mat, target_even))
#             else:
#                 # on odd sites, mat_b should not be target
#                 self.assertFalse(np.allclose(mat, target_even))

#                 # all the odd sites should still be the same as each other
#                 mat_site_1 = mat_b[2:4, 16:32]
#                 self.assertTrue(np.allclose(mat, mat_site_1))

#         # Check that the off-diagonal blocks are zero
#         for site1 in range(self.system_z2.cfg.lattice.size):
#             ind1 = 2 * site1
#             for site2 in range(self.system_z2.cfg.lattice.size):
#                 ind2 = 2 * site2
#                 if site1 != site2:
#                     block = mat_b[ind1 : ind1 + 2, 8 * ind2 : 8 * (ind2 + 2)]
#                     self.assertTrue(np.allclose(block, 0))

#     def test_mat_d(self):
#         """Test that the D matrix is the same on all even sites, and the same on all odd sites,
#         and zero where sites are mixed."""

#         mat_d = self.system_z2.mat_d_vec[0]  # we only have one layer in this test

#         # Change mat_d to site-based mode order
#         modes_link_order = self.system_z2.get_link_based_mode_order()
#         modes_site_order = self.system_z2.get_site_based_mode_order()
#         mat_perm = generate_permutation_matrix(modes_link_order, modes_site_order)
#         mat_perm = np.array(
#             mat_perm
#         )  # multiplication of ModeArray with np.ndarray is not working properly
#         mat_d = np.transpose(mat_perm) @ mat_d @ mat_perm

#         mat_site_0 = mat_d[0:16, 0:16]
#         mat_site_1 = mat_d[16:32, 16:32]
#         self.assertFalse(np.allclose(mat_site_0, mat_site_1))
#         for site in range(self.system_z2.cfg.lattice.size):
#             site_ind = 2 * site
#             mat = mat_d[
#                 8 * site_ind : 8 * (site_ind + 2), 8 * site_ind : 8 * (site_ind + 2)
#             ]

#             x, y = self.system_z2.cfg.lattice.ind2coord(site)
#             if (x + y) % 2 == 0:
#                 # on even sites mat_d should match site 0
#                 self.assertTrue(np.allclose(mat, mat_site_0))
#             else:
#                 # on odd sites mat_d should match site 1
#                 self.assertTrue(np.allclose(mat, mat_site_1))

#         # Check that the off-diagonal blocks are zero
#         for site1 in range(self.system_z2.cfg.lattice.size):
#             ind1 = 2 * site1
#             for site2 in range(self.system_z2.cfg.lattice.size):
#                 ind2 = 2 * site2
#                 if site1 != site2:
#                     block = mat_d[8 * ind1 : 8 * (ind1 + 2), 8 * ind2 : 8 * (ind2 + 2)]
#                     self.assertTrue(np.allclose(block, 0))

#     def test_covmat(self):

#         # Check the covmat for all fermionic layers
#         for lay in range(self.system_z2.cfg.num_pg_layer, self.system_z2.cfg.nlayer):
#             covmats = []
#             for site in range(self.system_z2.cfg.lattice.size):

#                 # Set the gauge configuration -
#                 #   there must be some flux, since otherwise the mass will be zero,
#                 #   so we choose to set the link to the right of the site under consideration to pi
#                 x, y = self.system_z2.cfg.lattice.ind2coord(site)
#                 config = np.array([0] * 7 + [0] * 1)
#                 ind = self.system_z2.cfg.lattice.coord2ind_dir(
#                     (x, y), lattice.Direction.X
#                 )
#                 config[ind] = np.pi
#                 self.system_z2.update_gauge_full_system(config)
#                 covmat = self.system_z2.ferm_covmat_vec[lay]

#                 site_ind = 2 * site
#                 mat = covmat[site_ind : site_ind + 2, site_ind : site_ind + 2]
#                 covmats.append(mat)

#             # Check the covmats
#             covmat_even = covmats[0]
#             covmat_odd = covmats[1]
#             self.assertFalse(
#                 np.allclose(covmat_even, covmat_odd)
#             )  # with high probability for random parameters
#             for site in range(self.system_z2.cfg.lattice.size):
#                 x, y = self.system_z2.cfg.lattice.ind2coord(site)
#                 mat = covmats[site]
#                 x, y = self.system_z2.cfg.lattice.ind2coord(site)
#                 if (x + y) % 2 == 0:
#                     # on even sites covmats should be the same
#                     self.assertTrue(np.allclose(mat, covmat_even))
#                 else:
#                     # on odd sites covmat should be the same
#                     self.assertTrue(np.allclose(mat, covmat_odd))

#     def test_mass(self):
#         """Ensure mass is the same on all even sites, the same on all odd sites, and different between them."""

#         # Check the mass
#         mass_even = 0
#         for lay in range(self.system_z2.cfg.num_pg_layer, self.system_z2.cfg.nlayer):

#             # Calculate the mass for each site
#             masses = []
#             for site in range(self.system_z2.cfg.lattice.size):

#                 # Set the gauge configuration -
#                 #   there must be some flux, since otherwise the mass will be zero,
#                 #   so we choose to set the link to the right of the site under consideration to pi
#                 x, y = self.system_z2.cfg.lattice.ind2coord(site)
#                 config = np.array([0] * 7 + [0] * 1)
#                 ind = self.system_z2.cfg.lattice.coord2ind_dir(
#                     (x, y), lattice.Direction.X
#                 )
#                 config[ind] = np.pi
#                 self.system_z2.update_gauge_full_system(config)
#                 covmat = self.system_z2.ferm_covmat_vec[lay]

#                 site_ind = 2 * site  # index into covariance matrix
#                 mass_site = 0.5 * (1 + covmat[site_ind + 1, site_ind])
#                 masses.append(mass_site)

#             # Check the masses
#             mass_even = masses[0]
#             mass_odd = masses[1]
#             self.assertFalse(
#                 np.allclose(mass_even, mass_odd)
#             )  # with high probability for random parameters
#             for site in range(self.system_z2.cfg.lattice.size):
#                 x, y = self.system_z2.cfg.lattice.ind2coord(site)
#                 mass_site = masses[site]
#                 if (x + y) % 2 == 0:
#                     # on even sites (where t was set to zero), mass should be zero
#                     self.assertTrue(np.allclose(mass_site, mass_even))
#                 else:
#                     # on odd sites, mass should not be zero
#                     self.assertTrue(np.allclose(mass_site, mass_odd))

#     def test_swap_even_odd(self):
#         """Swapping the parameters on the even and odd sites should:
#         - not change the mass or interaction energy,
#         - multiply the chem energy by minus 1,
#         - swap the blocks in gamma_maj,
#         """

#         # Set the gauge configuration
#         config = np.zeros(8)
#         config[0] = np.pi
#         self.system_z2.update_gauge_full_system(config)

#         # Use the paramvec from setUp(), and extract various values for comparison
#         gamma_maj_even = self.system_z2.gamma_maj_layervec_sitevec[0][0]
#         mass_op = self.system_z2.mass_energy_op
#         int_op = self.system_z2.int_energy_op
#         chem_op = np.sum(self.system_z2.chem_energy_op_vec)

#         # Swap the parameters for the even and odd sites, build a new system
#         new_paramvec = np.copy(self.system_z2.cfg.paramvec)
#         new_paramvec[:, [0, 1], :] = new_paramvec[:, [1, 0], :]
#         cfg = self.system_z2.cfg
#         cfg.paramvec = new_paramvec
#         system_z2 = system.Z2System2D(cfg)
#         system_z2.cfg.enforce_parameter_conditions(self.system_z2.cfg.paramvec)

#         # Set the config for the new system - it must be shifted to account for the
#         # swapping of the even/odd sublattices
#         config = np.zeros(8)
#         config[1] = np.pi
#         system_z2.update_gauge_full_system(config)

#         # Extract the values from the new system for comparison
#         new_gamma_maj_odd = system_z2.gamma_maj_layervec_sitevec[0][1]
#         new_mass_op = system_z2.mass_energy_op
#         new_int_op = system_z2.int_energy_op
#         new_chem_op = np.sum(system_z2.chem_energy_op_vec)

#         # Correct for chem offsets
#         chem_offset = (
#             0.5
#             * self.system_z2.cfg.lattice.size
#             * self.system_z2.cfg.num_fermionic_layer
#         )
#         chem_val = chem_op - chem_offset
#         new_chem_val = new_chem_op - chem_offset

#         # Compare
#         self.assertTrue(np.allclose(gamma_maj_even, new_gamma_maj_odd))
#         self.assertAlmostEqual(mass_op, new_mass_op)
#         self.assertAlmostEqual(int_op, new_int_op)
#         self.assertAlmostEqual(chem_val, -new_chem_val)

#     def test_grad_mass_energy(self):
#         # This is comparison of the analytic derivative against the numeric derivative
#         # for the 2 copy fermionic ansatz with 2 physical flavors
#         eps = 1e-5
#         system_z2 = self.system_z2
#         lat_2x2 = system_z2.cfg.lattice
#         paramvec = self.system_z2.cfg.paramvec
#         unitcell_size = self.system_z2.cfg.unitcell_size

#         config = np.array([0] * 7 + [np.pi] * 1)
#         system_z2.update_gauge_full_system(config)

#         deriv_ana = system_z2.mass_energy_op_grad_vec
#         symbolvec = system_z2.symbolvec

#         for layerind in range(self.system_z2.cfg.nlayer):
#             # we could skip the pure gauge layers, since they do not contribute
#             for uc_ind in range(unitcell_size):
#                 for ind in range(len(symbolvec)):
#                     with self.subTest(
#                         symbol=symbolvec[ind], layerind=layerind, uc_ind=uc_ind
#                     ):
#                         paramvec_left = np.copy(paramvec)
#                         paramvec_right = np.copy(paramvec)
#                         paramvec_left[layerind, uc_ind, ind] -= eps
#                         paramvec_right[layerind, uc_ind, ind] += eps
#                         system_cfg_left = system.Z2System2D_Config(
#                             lat_2x2,
#                             0.0,
#                             0.0,
#                             1.0,
#                             1.0,
#                             None,
#                             num_pg_layer=self.system_z2.cfg.num_pg_layer,
#                             num_fermionic_layer=self.system_z2.cfg.num_fermionic_layer,
#                             unitcell_size=unitcell_size,
#                             ncopy=2,
#                         )
#                         system_cfg_right = system.Z2System2D_Config(
#                             lat_2x2,
#                             0.0,
#                             0.0,
#                             1.0,
#                             1.0,
#                             None,
#                             num_pg_layer=self.system_z2.cfg.num_pg_layer,
#                             num_fermionic_layer=self.system_z2.cfg.num_fermionic_layer,
#                             unitcell_size=unitcell_size,
#                             ncopy=2,
#                         )

#                         system_cfg_left.paramvec = paramvec_left
#                         system_cfg_right.paramvec = paramvec_right

#                         system_z2_2_2_left = system.Z2System2D(system_cfg_left)
#                         system_z2_2_2_right = system.Z2System2D(system_cfg_right)
#                         system_z2_2_2_left.update_gauge_full_system(config)
#                         system_z2_2_2_right.update_gauge_full_system(config)

#                         val_left = system_z2_2_2_left.mass_energy_op
#                         val_right = system_z2_2_2_right.mass_energy_op
#                         deriv_num = (val_right - val_left) / (2 * eps)

#                         self.assertAlmostEqual(
#                             deriv_ana[layerind, uc_ind, ind], deriv_num, places=3
#                         )

#     def test_grad_chem_energy(self):
#         # This is comparison of the analytic derivative against the numeric derivative
#         # for the 2 copy fermionic ansatz with 2 physical flavors
#         eps = 1e-5
#         system_z2 = self.system_z2
#         lat_2x2 = system_z2.cfg.lattice
#         paramvec = self.system_z2.cfg.paramvec
#         unitcell_size = self.system_z2.cfg.unitcell_size

#         config = np.array([0] * 7 + [np.pi] * 1)
#         system_z2.update_gauge_full_system(config)

#         deriv_ana = system_z2.chem_energy_op_grad_vec
#         symbolvec = system_z2.symbolvec

#         for layerind in range(self.system_z2.cfg.nlayer):
#             # we could skip the pure gauge layers, since they do not contribute
#             for uc_ind in range(unitcell_size):
#                 for ind in range(len(symbolvec)):
#                     with self.subTest(
#                         symbol=symbolvec[ind], layerind=layerind, uc_ind=uc_ind
#                     ):
#                         paramvec_left = np.copy(paramvec)
#                         paramvec_right = np.copy(paramvec)
#                         paramvec_left[layerind, uc_ind, ind] -= eps
#                         paramvec_right[layerind, uc_ind, ind] += eps
#                         system_cfg_left = system.Z2System2D_Config(
#                             lat_2x2,
#                             0.0,
#                             0.0,
#                             1.0,
#                             1.0,
#                             None,
#                             num_pg_layer=self.system_z2.cfg.num_pg_layer,
#                             num_fermionic_layer=self.system_z2.cfg.num_fermionic_layer,
#                             unitcell_size=unitcell_size,
#                             ncopy=2,
#                         )
#                         system_cfg_right = system.Z2System2D_Config(
#                             lat_2x2,
#                             0.0,
#                             0.0,
#                             1.0,
#                             1.0,
#                             None,
#                             num_pg_layer=self.system_z2.cfg.num_pg_layer,
#                             num_fermionic_layer=self.system_z2.cfg.num_fermionic_layer,
#                             unitcell_size=unitcell_size,
#                             ncopy=2,
#                         )

#                         system_cfg_left.paramvec = paramvec_left
#                         system_cfg_right.paramvec = paramvec_right

#                         system_z2_2_2_left = system.Z2System2D(system_cfg_left)
#                         system_z2_2_2_right = system.Z2System2D(system_cfg_right)
#                         system_z2_2_2_left.update_gauge_full_system(config)
#                         system_z2_2_2_right.update_gauge_full_system(config)

#                         val_left = system_z2_2_2_left.chem_energy_op_vec[layerind]
#                         val_right = system_z2_2_2_right.chem_energy_op_vec[layerind]
#                         deriv_num = (val_right - val_left) / (2 * eps)

#                         self.assertAlmostEqual(
#                             deriv_ana[layerind, uc_ind, ind], deriv_num, places=3
#                         )

#     def test_grad_norm(self):
#         # This is comparison of the analytic derivative against the numeric derivative
#         # for the 2 copy fermionic ansatz with 2 physical flavors
#         eps = 1e-5
#         system_z2 = self.system_z2
#         lat_2x2 = system_z2.cfg.lattice
#         paramvec = self.system_z2.cfg.paramvec
#         unitcell_size = self.system_z2.cfg.unitcell_size

#         config = np.array([0] * 7 + [np.pi] * 1)
#         system_z2.update_gauge_full_system(config)

#         deriv_ana = system_z2.compute_grad_norm_vec()
#         symbolvec = system_z2.symbolvec

#         for layerind in range(self.system_z2.cfg.nlayer):
#             # we could skip the pure gauge layers, since they do not contribute
#             for uc_ind in range(unitcell_size):
#                 for ind in range(len(symbolvec)):
#                     with self.subTest(
#                         symbol=symbolvec[ind], layerind=layerind, uc_ind=uc_ind
#                     ):
#                         paramvec_left = np.copy(paramvec)
#                         paramvec_right = np.copy(paramvec)
#                         paramvec_left[layerind, uc_ind, ind] -= eps
#                         paramvec_right[layerind, uc_ind, ind] += eps
#                         system_cfg_left = system.Z2System2D_Config(
#                             lat_2x2,
#                             0.0,
#                             0.0,
#                             1.0,
#                             1.0,
#                             None,
#                             num_pg_layer=self.system_z2.cfg.num_pg_layer,
#                             num_fermionic_layer=self.system_z2.cfg.num_fermionic_layer,
#                             unitcell_size=unitcell_size,
#                             ncopy=2,
#                         )
#                         system_cfg_right = system.Z2System2D_Config(
#                             lat_2x2,
#                             0.0,
#                             0.0,
#                             1.0,
#                             1.0,
#                             None,
#                             num_pg_layer=self.system_z2.cfg.num_pg_layer,
#                             num_fermionic_layer=self.system_z2.cfg.num_fermionic_layer,
#                             unitcell_size=unitcell_size,
#                             ncopy=2,
#                         )

#                         system_cfg_left.paramvec = paramvec_left
#                         system_cfg_right.paramvec = paramvec_right

#                         system_z2_2_2_left = system.Z2System2D(system_cfg_left)
#                         system_z2_2_2_right = system.Z2System2D(system_cfg_right)
#                         system_z2_2_2_left.update_gauge_full_system(config)
#                         system_z2_2_2_right.update_gauge_full_system(config)

#                         val_left = system_z2_2_2_left.calculate_lognorm(
#                             all_factors=True
#                         )
#                         val_right = system_z2_2_2_right.calculate_lognorm(
#                             all_factors=True
#                         )
#                         deriv_num = (val_right - val_left) / (2 * eps)


#                         self.assertAlmostEqual(
#                             deriv_ana[layerind, uc_ind, ind], deriv_num, places=3
#                         )
class TestElectricEnergyUniformityRandomk(unittest.TestCase):
    """
    Pick k random (unique) subsets of links with varying sizes and verify that
    the per-link electric energy (mean over the selected links) is identical
    across all of them for a translationally invariant, neutral-gauge setup.
    """

    def setUp(self):
        # Fixed lattice: 2x2 with PBC -> nlinks = 8
        self.lat = lattice.Lattice2D(2, 2)
        self.num_pg_layer = 2
        self.num_fermionic_layer = 0
        self.nlayer = self.num_pg_layer + self.num_fermionic_layer
        self.unitcell_size = 1  # translation invariance

        # Reproducible random params shared by all systems in this test
        rng = np.random.RandomState(1234)

        self.paramvec = rng.rand(self.nlayer, self.unitcell_size, 20)

        # Build a neutral gauge configuration once
        Dn = gauge.D2nGauge(3)
        neutral = Dn.get_neutral_gauge_value()
        self.neutral_config = np.array([neutral] * self.lat.nlinks)

    def _energy_for_subset(self, link_inds: tuple[int, ...]) -> float:
        """Return the mean per selected link (sys.el_energy_op) for the given subset."""
        cfg = system.D6System2D_Config(
            self.lat,
            g_el=1.0,
            g_mag=0.0,
            g_int=0.0,
            g_mass=0.0,
            g_chem=None,
            num_pg_layer=self.num_pg_layer,
            num_fermionic_layer=self.num_fermionic_layer,
            unitcell_size=self.unitcell_size,
            mod_link_inds=tuple(sorted(link_inds)),
        )
        cfg.paramvec = np.copy(self.paramvec)
        sys = system.D2nSystem2D(cfg)
        sys.update_gauge_full_system(self.neutral_config)
        return float(sys.el_energy_op)  # mean over the selected links

    def _random_unique_subsets(self, nlinks: int, k: int, seed: int = 2025):
        """
        Generate k unique random non-empty subsets with varying sizes in [1..nlinks].
        Ensures at least two different sizes appear (if nlinks >= 2).
        """
        rng = np.random.RandomState(seed)
        subsets = set()
        # Try a reasonable number of attempts to avoid rare infinite loops
        attempts, max_attempts = 0, 10000

        while len(subsets) < k and attempts < max_attempts:
            s = rng.randint(1, nlinks + 1)  # size in [1..nlinks]
            choice = tuple(sorted(rng.choice(nlinks, size=s, replace=False)))
            subsets.add(choice)
            attempts += 1

        # Ensure size diversity when possible (nlinks >= 2)
        if nlinks >= 2:
            sizes_present = {len(t) for t in subsets}
            if len(sizes_present) == 1:
                # Force-inject another size
                if 1 not in sizes_present:
                    subsets.add((0,))  # singleton
                if nlinks not in sizes_present:
                    subsets.add(tuple(range(nlinks)))  # full set
                # Trim back to k if we overshot
                subsets = set(list(subsets)[:k])

        # If we somehow fell short, pad deterministically
        while len(subsets) < k:
            # cycle sizes 1..nlinks deterministically
            s = (len(subsets) % nlinks) + 1
            # take first s indices
            subsets.add(tuple(range(s)))

        return list(subsets)[:k]

    def test_el_energy_uniform_over_k_random_subsets(self):
        nlinks = self.lat.nlinks
        k = 10

        subsets = self._random_unique_subsets(nlinks, k, seed=2025)
        # Compute per-link energies for each subset and compare
        energies = [self._energy_for_subset(sub) for sub in subsets]
        base = energies[0]
        for e in energies[1:]:
            self.assertAlmostEqual(e, base, places=8)


class TestElectricEnergyDropRealZero(unittest.TestCase):
    """Regression test for `drop_real_zero` pruning in `generate_gauged_projector_terms`.

    Background:
    - Electric energy is evaluated from Pfaffians of submatrices of a Majorana covariance matrix.
    - The physical observable is of the form sum_j f_j |jmn><jmn| and must be real.
    - Terms that contribute only a purely imaginary piece cancel in <sum_j f_j |jmn><jmn|>.

    The feature `drop_real_zero=True` prunes such terms to reduce the number of Pfaffians.

    Contract:
    - For the same parameters and the same gauge configuration,
      el_energy_op (and el_energy_op_vec) must be invariant under toggling drop_real_zero.
    """

    def setUp(self) -> None:
        """Set up a small deterministic D6 2D system and a nontrivial gauge configuration.

        This initializes shared test fixtures:
        - a 2x2 lattice (PBC),
        - a reproducible random parameter tensor (paramvec),
        - and a gauge configuration containing one flux link.

        Args:
            None

        Returns:
            None
        """
        # Keep the test small and deterministic.
        self.lat = lattice.Lattice2D(2, 2)
        self.num_pg_layer = 2
        self.num_fermionic_layer = 0
        self.nlayer = self.num_pg_layer + self.num_fermionic_layer
        self.unitcell_size = 1

        rng = np.random.RandomState(123)  # deterministic
        self.paramvec = rng.rand(self.nlayer, self.unitcell_size, 20)

        # Use a non-trivial gauge configuration (include one flux) so the test is not vacuous.
        Dn = gauge.D2nGauge(3)
        neutral = Dn.get_neutral_gauge_value()
        flux = Dn.get_representation(1, 1)  # nontrivial flux value
        # 2x2 with PBC -> nlinks = 8
        self.gauge_config = np.array([neutral] * 7 + [flux] * 1)

    @staticmethod
    def _rebuild_idxarr_vec(cfg, *, drop_real_zero: bool) -> None:
        """Rebuild `cfg.idxarr_vec` with an explicit `drop_real_zero` setting.

        This is a test-only reimplementation of the production logic in
        `D2nSystem2D.init_el_energy_terms()` (see
        `ggpeps/system/config_D2n_2d.py`), with the *only* intentional difference
        being that we pass `drop_real_zero` through to `generate_gauged_projector_terms()`.

        The produced structure matches the production layout:
        for each group element used in the electric energy, and for each layer, we build a
        tuple of "quads" (H0, H1, V0, V1), where each entry is an (prefactor, indices) term.

        Args:
            cfg: Config instance to mutate. On return, `cfg.idxarr_vec` is overwritten.
            drop_real_zero: If True, drop terms whose contribution is guaranteed to cancel in
                <P + P_dagger> (i.e., terms that would contribute only purely imaginary pieces).
                If False, keep those terms as well.

        Returns:
            None. This function mutates `cfg.idxarr_vec` in-place.

        NOTE:
            - Future-proofing requirement: this helper should remain a *verbatim code duplicate*
              of `D2nSystem2D.init_el_energy_terms()`, except for the explicit
              `drop_real_zero` plumbing. If the production construction of `idxarr_vec` changes
              (ordering, layering, loops, etc.), update this helper accordingly to avoid false
              positives/negatives in the test.
        """
        result = []
        for group_element in cfg.gaugemgr.group_elements_for_el_energy:
            # --- Pure gauge (mix_copies=True) ---
            idxarr_lay_pg_h_0, _ = generate_gauged_projector_terms(
                cfg.ncopy,
                cfg.ncolors,
                False,
                lattice.Direction.X,
                group_element,
                site=0,
                drop_imag=drop_real_zero,
            )
            idxarr_lay_pg_h_1, _ = generate_gauged_projector_terms(
                cfg.ncopy,
                cfg.ncolors,
                False,
                lattice.Direction.X,
                group_element,
                site=1,
                drop_imag=drop_real_zero,
            )
            idxarr_lay_pg_v_0, _ = generate_gauged_projector_terms(
                cfg.ncopy,
                cfg.ncolors,
                False,
                lattice.Direction.Y,
                group_element,
                site=0,
                drop_imag=drop_real_zero,
            )
            idxarr_lay_pg_v_1, _ = generate_gauged_projector_terms(
                cfg.ncopy,
                cfg.ncolors,
                False,
                lattice.Direction.Y,
                group_element,
                site=1,
                drop_imag=drop_real_zero,
            )

            # --- Fermionic (mix_copies=False) ---
            idxarr_lay_pf_h_0, _ = generate_gauged_projector_terms(
                cfg.ncopy,
                cfg.ncolors,
                False,
                lattice.Direction.X,
                group_element,
                site=0,
                drop_imag=drop_real_zero,
            )
            idxarr_lay_pf_h_1, _ = generate_gauged_projector_terms(
                cfg.ncopy,
                cfg.ncolors,
                False,
                lattice.Direction.X,
                group_element,
                site=1,
                drop_imag=drop_real_zero,
            )
            idxarr_lay_pf_v_0, _ = generate_gauged_projector_terms(
                cfg.ncopy,
                cfg.ncolors,
                False,
                lattice.Direction.Y,
                group_element,
                site=0,
                drop_imag=drop_real_zero,
            )
            idxarr_lay_pf_v_1, _ = generate_gauged_projector_terms(
                cfg.ncopy,
                cfg.ncolors,
                False,
                lattice.Direction.Y,
                group_element,
                site=1,
                drop_imag=drop_real_zero,
            )

            zipped_pg = tuple(zip(idxarr_lay_pg_h_0, idxarr_lay_pg_h_1, idxarr_lay_pg_v_0, idxarr_lay_pg_v_1))
            zipped_pf = tuple(zip(idxarr_lay_pf_h_0, idxarr_lay_pf_h_1, idxarr_lay_pf_v_0, idxarr_lay_pf_v_1))

            # First pure-gauge layers, then fermionic layers.
            result.append(tuple([zipped_pg] * cfg.num_pg_layer + [zipped_pf] * cfg.num_fermionic_layer))

        cfg.idxarr_vec = tuple(result)

    def _build_system(self) -> system.D2nSystem2D:
        """Construct a D2nSystem2D with the test's lattice and deterministic parameters.

        The returned system:
        - uses the same ansatz/layer counts as the test fixtures,
        - receives a copy of `self.paramvec`,
        - and has parameter constraints enforced.

        Args:
            None

        Returns:
            A fully constructed `system.Z2System2D` instance (gauge not yet applied).
        """
        cfg = system.D6System2D_Config(
            self.lat,
            g_el=1.0,
            g_mag=0.0,
            g_int=0.0,
            g_mass=0.0,
            g_chem=None,
            num_pg_layer=self.num_pg_layer,
            num_fermionic_layer=self.num_fermionic_layer,
            unitcell_size=self.unitcell_size,
            mod_link_inds=(0,),
        )
        cfg.paramvec = np.copy(self.paramvec)
        sys_obj = system.D2nSystem2D(cfg)
        sys_obj.cfg.enforce_parameter_conditions(sys_obj.cfg.paramvec)
        return sys_obj

    def test_el_energy_invariant_under_drop_real_zero(self) -> None:
        """Verify invariance of electric energy under toggling `drop_real_zero`.

        We build two identical systems and apply the same gauge configuration.
        The only difference is that for one system we rebuild `cfg.idxarr_vec` using
        `drop_real_zero=False`, so it includes also terms that would contribute only
        purely imaginary pieces that should cancel in <sum_j f_j |jmn><jmn|>.

        Args:
            None

        Returns:
            None
        """
        # Build two identical systems.
        sys_drop = self._build_system()  # default drop_real_zero=True inside config
        sys_keep = self._build_system()

        # Override only idxarr_vec for sys_keep to represent drop_real_zero=False.
        self._rebuild_idxarr_vec(sys_keep.cfg, drop_real_zero=False)

        # Apply the same gauge configuration to both.
        sys_drop.update_gauge_full_system(self.gauge_config)
        sys_keep.update_gauge_full_system(self.gauge_config)

        # Compare both scalar and vector forms.
        self.assertTrue(
            np.allclose(sys_drop.el_energy_op, sys_keep.el_energy_op, atol=1e-12, rtol=1e-12),
            msg="el_energy_op changed when toggling drop_real_zero (should be invariant).",
        )
        self.assertTrue(
            np.allclose(sys_drop.el_energy_op_vec, sys_keep.el_energy_op_vec, atol=1e-12, rtol=1e-12),
            msg="el_energy_op_vec changed when toggling drop_real_zero (should be invariant).",
        )


class TestMajoranaModeOrdering(unittest.TestCase):
    """Tests for the generation of Majorana mode orderings on a single link."""

    def test_wrong_single_link_majorana_mode_order_by_copy_then_color(self):
        """Test the 'wrong' ordering function used for generation of rotmat.
        This function orders by copy first, then color, and has a specific interleaved
        mode sequence (l1, then l2, then r1, then r2).
        Note: num_colors is hardcoded to 2 in this function.
        """
        # Test for 1 copy (num_copies=1)
        expected_1_copy = [
            "l1_1_1",
            "l1_1_2",  # copy 1, colors 1 & 2 for l1
            "l2_1_1",
            "l2_1_2",  # copy 1, colors 1 & 2 for l2
            "r1_1_1",
            "r1_1_2",  # copy 1, colors 1 & 2 for r1
            "r2_1_1",
            "r2_1_2",  # copy 1, colors 1 & 2 for r2
        ]
        result_1_copy = system.D2nSystem2D.get_wrong_single_link_majorana_mode_order_by_copy_then_color(num_copies=1)
        self.assertEqual(result_1_copy, expected_1_copy)

        # Test for 2 copies (num_copies=2)
        expected_2_copies = [
            # Copy 1
            "l1_1_1",
            "l1_1_2",
            "l2_1_1",
            "l2_1_2",
            "r1_1_1",
            "r1_1_2",
            "r2_1_1",
            "r2_1_2",
            # Copy 2
            "l1_2_1",
            "l1_2_2",
            "l2_2_1",
            "l2_2_2",
            "r1_2_1",
            "r1_2_2",
            "r2_2_1",
            "r2_2_2",
        ]
        result_2_copies = system.D2nSystem2D.get_wrong_single_link_majorana_mode_order_by_copy_then_color(num_copies=2)
        self.assertEqual(result_2_copies, expected_2_copies)

    def test_get_single_link_majorana_mode_order(self):
        """Test the 'correct' actual implementation ordering function.
        This function orders by color first, then copy, and appends the
        modes sequentially (l1, l2, r1, r2) within the inner loop.
        """
        # Test for 1 copy, 2 colors
        expected_1_copy_2_color = [
            # Color 1, Copy 1
            "l1_1_1",
            "l2_1_1",
            "r1_1_1",
            "r2_1_1",
            # Color 2, Copy 1
            "l1_1_2",
            "l2_1_2",
            "r1_1_2",
            "r2_1_2",
        ]
        # Changed from system.System2DBase to system.D2nSystem2D
        result_1_copy_2_color = system.D2nSystem2D.get_single_link_majorana_mode_order(num_copies=1, num_colors=2)
        self.assertEqual(result_1_copy_2_color, expected_1_copy_2_color)

        # Test for 2 copies, 2 colors
        expected_2_copy_2_color = [
            # Color 1, Copy 1
            "l1_1_1",
            "l2_1_1",
            "r1_1_1",
            "r2_1_1",
            # Color 1, Copy 2
            "l1_2_1",
            "l2_2_1",
            "r1_2_1",
            "r2_2_1",
            # Color 2, Copy 1
            "l1_1_2",
            "l2_1_2",
            "r1_1_2",
            "r2_1_2",
            # Color 2, Copy 2
            "l1_2_2",
            "l2_2_2",
            "r1_2_2",
            "r2_2_2",
        ]
        # Changed from system.System2DBase to system.D2nSystem2D
        result_2_copy_2_color = system.D2nSystem2D.get_single_link_majorana_mode_order(num_copies=2, num_colors=2)
        self.assertEqual(result_2_copy_2_color, expected_2_copy_2_color)
