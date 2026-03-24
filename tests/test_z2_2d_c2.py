import unittest

import numpy as np

from ggpeps import utils
from ggpeps import system
from ggpeps import lattice

# ======================= Z2 fermionic system (2 copies) =======================


class TestZ2C2SystemMethods(unittest.TestCase):
    def setUp(self):

        lat = lattice.Lattice2D(2, 2)
        paramvec_real = np.random.rand(1, 10)
        paramvec_real = np.concatenate([paramvec_real, np.zeros((1, 10))], axis=1)
        cfg = system.Z2System2D2CConfig(lat, 0, 0, 0, 0, None)
        cfg.paramvec = paramvec_real
        self.system_z2_2_2_real = system.Z2System2D(cfg)

        lat = lattice.Lattice2D(2, 2)
        paramvec = np.random.rand(1, 20)
        cfg = system.Z2System2D2CConfig(lat, 0, 0, 0, 0, None)
        cfg.paramvec = paramvec
        self.system_z2_2_2 = system.Z2System2D(cfg)

    def test_tmat_deriv_symb(self):
        eps = 1e-5
        lat = lattice.Lattice2D(2, 2)
        paramvec = np.random.rand(1, 20)
        paramvec_left = np.copy(paramvec)
        paramvec_left[0, 1] -= eps
        paramvec_right = np.copy(paramvec)
        paramvec_right[0, 1] += eps

        cfg = system.Z2System2D2CConfig(lat, 0, 0, 0, 0, None)
        cfg.paramvec = paramvec
        cfg_left = system.Z2System2D2CConfig(lat, 0, 0, 0, 0, None)
        cfg_left.paramvec = paramvec_left
        cfg_right = system.Z2System2D2CConfig(lat, 0, 0, 0, 0, None)
        cfg_right.paramvec = paramvec_right

        system_z2_2_2 = system.Z2System2D(cfg)
        system_z2_2_2_left = system.Z2System2D(cfg_left)
        system_z2_2_2_right = system.Z2System2D(cfg_right)

        symbvec = system_z2_2_2.symbolvec
        # Derivative wrt yr
        deriv_ana = system_z2_2_2.compute_tmat_deriv(symbvec[1])
        lay = 0
        site = 0
        tmat_left = system_z2_2_2_left.tmat_layervec_sitevec[lay][site]
        tmat_right = system_z2_2_2_right.tmat_layervec_sitevec[lay][site]
        deriv_num = (tmat_right - tmat_left) / (2 * eps)
        self.assertTrue(np.allclose(deriv_ana, deriv_num))

    def test_tmat_antisymmetric_real(self):
        lay = 0
        site = 0
        tmat = self.system_z2_2_2_real.tmat_layervec_sitevec[lay][site]
        self.assertTrue(utils.is_antisymmetric(tmat))

    def test_tmat_antisymmetric(self):
        lay = 0
        site = 0
        tmat = self.system_z2_2_2.tmat_layervec_sitevec[lay][site]
        self.assertTrue(utils.is_antisymmetric(tmat))

    def test_gamma_dirac_covariance_real(self):
        lay = 0
        site = 0
        gamma_dirac = self.system_z2_2_2_real.gamma_dirac_layervec_sitevec[lay][site]
        m, n = gamma_dirac.shape
        res = gamma_dirac @ np.transpose(np.conjugate(gamma_dirac))
        ref = 0.25 * np.eye(gamma_dirac.shape[0])
        self.assertTrue(np.allclose(ref, res))
        self.assertEqual(m, n)
        self.assertTrue(utils.is_antisymmetric(gamma_dirac))

    def test_gamma_dirac_covariance(self):
        lay = 0
        site = 0
        gamma_dirac = self.system_z2_2_2.gamma_dirac_layervec_sitevec[lay][site]
        m, n = gamma_dirac.shape
        res = gamma_dirac @ np.transpose(np.conjugate(gamma_dirac))
        ref = 0.25 * np.eye(gamma_dirac.shape[0])
        self.assertTrue(np.allclose(ref, res))
        self.assertEqual(m, n)
        self.assertTrue(utils.is_antisymmetric(gamma_dirac))

    def test_gamma_maj_covariance_real(self):
        lay = 0
        site = 0
        gamma_maj = self.system_z2_2_2_real.gamma_maj_layervec_sitevec[lay][site]
        m, n = gamma_maj.shape
        self.assertEqual(m, n)
        self.assertEqual(m, 18)
        self.assertTrue(np.allclose(np.imag(gamma_maj), 0))
        self.assertTrue(utils.is_antisymmetric(gamma_maj))
        self.assertTrue(np.allclose(gamma_maj @ gamma_maj, -np.eye(m)))
        self.assertTrue(np.allclose(gamma_maj @ np.transpose(gamma_maj), np.eye(m)))

    def test_gamma_maj_covariance(self):
        lay = 0
        site = 0
        gamma_maj = self.system_z2_2_2.gamma_maj_layervec_sitevec[lay][site]
        m, n = gamma_maj.shape
        self.assertEqual(m, n)
        self.assertEqual(m, 18)
        self.assertTrue(np.allclose(np.imag(gamma_maj), 0))
        self.assertTrue(utils.is_antisymmetric(gamma_maj))
        self.assertTrue(np.allclose(gamma_maj @ gamma_maj, -np.eye(m)))
        self.assertTrue(np.allclose(gamma_maj @ np.transpose(gamma_maj), np.eye(m)))

    def test_gamma_maj_sys_covariance_real(self):
        gamma_maj = self.system_z2_2_2_real.gamma_maj_sys_vec[0]
        m, n = gamma_maj.shape
        self.assertEqual(m, n)
        self.assertTrue(np.allclose(np.imag(gamma_maj), 0))
        self.assertTrue(utils.is_antisymmetric(gamma_maj))
        self.assertTrue(np.allclose(gamma_maj @ gamma_maj, -np.eye(m)))
        self.assertTrue(np.allclose(gamma_maj @ np.transpose(gamma_maj), np.eye(m)))

    def test_gamma_maj_sys_covariance(self):
        gamma_maj = self.system_z2_2_2.gamma_maj_sys_vec[0]
        m, n = gamma_maj.shape
        self.assertEqual(m, n)
        self.assertTrue(np.allclose(np.imag(gamma_maj), 0))
        self.assertTrue(utils.is_antisymmetric(gamma_maj))
        self.assertTrue(np.allclose(gamma_maj @ gamma_maj, -np.eye(m)))
        self.assertTrue(np.allclose(gamma_maj @ np.transpose(gamma_maj), np.eye(m)))

    def test_gamma_in_sys_covariance(self):
        gamma_in = self.system_z2_2_2.gamma_in_sys_vec[0]
        m, n = gamma_in.shape
        self.assertEqual(m, n)
        self.assertTrue(utils.is_antisymmetric(gamma_in))
        self.assertTrue(np.allclose(gamma_in @ gamma_in, -np.eye(m)))
        self.assertTrue(np.allclose(gamma_in @ np.transpose(gamma_in), np.eye(m)))

    def test_derivative_gamma_sys_finite_diff(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # There is no sampling involved here. The gauge field is 0.
        eps = 1e-5
        paramvec = np.random.rand(1, 20)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D2CConfig(lat_2x2, 1.0, None, None, None, None)
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(system_cfg)
        symbolvec = self.system_z2_2_2_real.symbolvec
        for ind in range(len(symbolvec)):
            with self.subTest(symbol=symbolvec[ind]):
                paramvec_left = np.copy(paramvec)
                paramvec_right = np.copy(paramvec)
                # We are only modifying the first layer (there is only one)
                paramvec_left[0, ind] -= eps
                paramvec_right[0, ind] += eps
                system_cfg_left = system.Z2System2D2CConfig(lat_2x2, 1.0, None, None, None, None)
                system_cfg_right = system.Z2System2D2CConfig(lat_2x2, 1.0, None, None, None, None)
                system_cfg_left.paramvec = paramvec_left
                system_cfg_right.paramvec = paramvec_right

                system_z2_2_2_left = system.Z2System2D(system_cfg_left)
                system_z2_2_2_right = system.Z2System2D(system_cfg_right)

                deriv_maj_sys = system_z2_2_2.gamma_maj_sys_deriv_vec(symbolvec[ind])[0]
                deriv_maj_sys_left = system_z2_2_2_left.gamma_maj_sys_vec[0]
                deriv_maj_sys_right = system_z2_2_2_right.gamma_maj_sys_vec[0]

                deriv_maj_sys_num = (deriv_maj_sys_right - deriv_maj_sys_left) / (2 * eps)

                self.assertTrue(np.allclose(deriv_maj_sys_num, deriv_maj_sys))

    def test_norm_minimal(self):
        # This update is a nullop since we initialize the gauge-field with 0
        zeroarr = (np.zeros((1, 1)),)
        # The factor of 2 compensates for the
        logdet_inc = 2 * self.system_z2_2_2_real.update_lognorm_inc(0, np.asarray(zeroarr), all_factors=False)
        # This is equivalent to
        # logdet_inc = self.system_z2_2_2.incdet.det()
        diff = self.system_z2_2_2_real.mat_d_inv_vec[0] - self.system_z2_2_2_real.gamma_in_sys_vec[0]
        sign, logdet = np.linalg.slogdet(diff)
        self.assertGreater(sign, 0)
        self.assertAlmostEqual(logdet_inc, logdet)

    def test_norm_incremental(self):
        # Test that the incremental update is equivalent to re-calculation of the norm
        # This update is a nullop since we initialize the gauge-field with 0
        zeroarr = (np.zeros((1, 1)),)
        weight_inc = self.system_z2_2_2_real.update_lognorm_inc(0, np.asarray(zeroarr), all_factors=True)
        weight_recalc = self.system_z2_2_2_real.calculate_lognorm(all_factors=True)
        self.assertAlmostEqual(weight_inc, weight_recalc)

    def test_norm_incremental_update(self):
        # Test that the incremental update is equivalent to re-calculation of the norm
        ind = 0
        theta = np.array([[-1.0]])
        weight_inc = self.system_z2_2_2_real.calculate_weight_attempt(ind, theta, all_factors=True)
        self.system_z2_2_2_real.update_gauge_ind(ind, theta)
        weight_recalc = self.system_z2_2_2_real.calculate_lognorm(all_factors=True)
        self.assertAlmostEqual(weight_inc, weight_recalc)

    def test_grad_over_norm(self):
        # This is comparison of the analytic derivative against the numeric derivative
        eps = 1e-5
        lay = 0
        uc_ind = 0
        paramvec = np.random.rand(1, 20)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D2CConfig(
            lat_2x2, 1.0, None, None, None, None, num_pg_layer=1, num_fermionic_layer=0
        )
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(system_cfg)
        symbolvec = system_z2_2_2.symbolvec
        for ind in range(len(symbolvec)):
            with self.subTest(symbol=symbolvec[ind]):
                paramvec_left = np.copy(paramvec)
                paramvec_right = np.copy(paramvec)
                paramvec_left[0, ind] -= eps
                paramvec_right[0, ind] += eps
                system_cfg_left = system.Z2System2D2CConfig(
                    lat_2x2,
                    1.0,
                    None,
                    None,
                    None,
                    None,
                    num_pg_layer=1,
                    num_fermionic_layer=0,
                )
                system_cfg_right = system.Z2System2D2CConfig(
                    lat_2x2,
                    1.0,
                    None,
                    None,
                    None,
                    None,
                    num_pg_layer=1,
                    num_fermionic_layer=0,
                )
                system_cfg_left.paramvec = paramvec_left
                system_cfg_right.paramvec = paramvec_right

                system_z2_2_2_left = system.Z2System2D(system_cfg_left)
                system_z2_2_2_right = system.Z2System2D(system_cfg_right)

                # This is a single layer construction, we always use layer 0 to test.
                deriv_ana = system_z2_2_2.grad_over_norm_vec[lay, uc_ind, ind]
                norm_left = system_z2_2_2_left.calculate_lognorm(all_factors=True)
                norm_right = system_z2_2_2_right.calculate_lognorm(all_factors=True)
                deriv_num = (norm_right - norm_left) / (2 * eps)

                self.assertAlmostEqual(deriv_ana, deriv_num, places=5)

    def test_grad_el_energy_1_layer(self):
        # This is comparison of the analytic derivative against the numeric derivative
        eps = 1e-5
        paramvec = np.random.rand(1, 20)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D2CConfig(lat_2x2, 1.0, None, None, None, None)
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(system_cfg)
        symbolvec = system_z2_2_2.symbolvec
        deriv_ana = system_z2_2_2.el_energy_op_grad_vec

        uc_ind = 0

        for ind in range(len(symbolvec)):
            with self.subTest(symbol=symbolvec[ind]):
                paramvec_left = np.copy(paramvec)
                paramvec_right = np.copy(paramvec)
                paramvec_left[0, ind] -= eps
                paramvec_right[0, ind] += eps
                system_cfg_left = system.Z2System2D2CConfig(
                    lat_2x2,
                    1.0,
                    None,
                    None,
                    None,
                    None,
                    num_pg_layer=1,
                    num_fermionic_layer=0,
                )
                system_cfg_right = system.Z2System2D2CConfig(
                    lat_2x2,
                    1.0,
                    None,
                    None,
                    None,
                    None,
                    num_pg_layer=1,
                    num_fermionic_layer=0,
                )

                system_cfg_left.paramvec = paramvec_left
                system_cfg_right.paramvec = paramvec_right

                system_z2_2_2_left = system.Z2System2D(system_cfg_left)
                system_z2_2_2_right = system.Z2System2D(system_cfg_right)

                val_left = system_z2_2_2_left.el_energy_op
                val_right = system_z2_2_2_right.el_energy_op
                deriv_num = (val_right - val_left) / (2 * eps)

                self.assertAlmostEqual(deriv_ana[0, uc_ind, ind], deriv_num, places=5)

    def test_grad_el_energy_2_layer(self):
        # This is comparison of the analytic derivative against the numeric derivative
        eps = 1e-5
        nlayer = 2
        paramvec = np.random.rand(2, 20)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D2CConfig(
            lat_2x2, 1.0, None, None, None, None, num_pg_layer=2, num_fermionic_layer=0
        )
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(system_cfg)
        deriv_ana = system_z2_2_2.el_energy_op_grad_vec
        symbolvec = system_z2_2_2.symbolvec

        uc_ind = 0

        for layerind in range(nlayer):
            for ind in range(len(symbolvec)):
                with self.subTest(symbol=symbolvec[ind], layerind=layerind):
                    paramvec_left = np.copy(paramvec)
                    paramvec_right = np.copy(paramvec)
                    paramvec_left[layerind, ind] -= eps
                    paramvec_right[layerind, ind] += eps
                    system_cfg_left = system.Z2System2D2CConfig(
                        lat_2x2,
                        1.0,
                        None,
                        None,
                        None,
                        None,
                        num_pg_layer=2,
                        num_fermionic_layer=0,
                    )
                    system_cfg_right = system.Z2System2D2CConfig(
                        lat_2x2,
                        1.0,
                        None,
                        None,
                        None,
                        None,
                        num_pg_layer=2,
                        num_fermionic_layer=0,
                    )

                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    system_z2_2_2_left = system.Z2System2D(system_cfg_left)
                    system_z2_2_2_right = system.Z2System2D(system_cfg_right)

                    val_left = system_z2_2_2_left.el_energy_op
                    val_right = system_z2_2_2_right.el_energy_op
                    deriv_num = (val_right - val_left) / (2 * eps)

                    self.assertAlmostEqual(deriv_ana[layerind, uc_ind, ind], deriv_num, places=3)

    def test_el_energy_1_layer_single_eval(self):
        # Calculate the electric energy of an empty system.
        paramvec = np.zeros((1, 20))
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D2CConfig(
            lat_2x2, 1.0, None, None, None, None, num_pg_layer=1, num_fermionic_layer=0
        )
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(system_cfg)
        el_energy = system_z2_2_2.el_energy
        self.assertAlmostEqual(el_energy, 0.0)
