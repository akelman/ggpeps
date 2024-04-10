import unittest 
from unittest import skip

import numpy as np

from ggpeps import utils
from ggpeps import lattice
from ggpeps import system, exacteval

from ggpeps.lattice import Direction
from ggpeps.mc import MonteCarloEstimatorConfig, MonteCarloEstimator, MonteCarloManager
from ggpeps.system import Z2System2D_G2C_F2C_Config, Z2System2D_G2C_F2C
from ggpeps.minimizer import Minimizer, MinimizerConfig
from ggpeps.utils import compare_array_elementwise

# ======================= Z2 fermionic system (4 copies) =========================================

class TestZ2C4System(unittest.TestCase):
    def setUp(self):

        lat = lattice.Lattice2D(2,2)
        paramvec = np.random.rand(2, 20)
        cfg = system.Z2System2D_G2C_F2C_Config(lat, 1,1,1,1,None)
        cfg.paramvec = paramvec
        self.system_z2 = system.Z2System2D_G2C_F2C(cfg) 
        self.system_z2.cfg.enforce_parameter_conditions(self.system_z2.cfg.paramvec)   
    
    def test_required_params_are_zero(self):
        """Ensure that the parameters that must vanish to guarantee ansatz symmetries do indeed vanish.
        """
        mat = self.system_z2.cfg.paramvec
        t_indices = [0,3,10,13] # index of t1r, t2r, t1i, t2i in symbolvec
        for layer_ind in range(self.system_z2.cfg.num_pg_layer):
            for t_ind in t_indices:
                with self.subTest(tind=t_ind, layerind=layer_ind):
                    coord = (layer_ind, t_ind)
                    self.assertAlmostEqual(mat[coord], 0)
        
        zero_for_fermionic_layer = [3,13,1,2,4,5,11,12,14,15] # index of t2r, t2i, y1r, z1r, y2r, z2r, y1i, z1i, y2i, z2i in symbolvec
        for layer_ind in range(self.system_z2.cfg.num_pg_layer, self.system_z2.cfg.nlayer):
            for ind in zero_for_fermionic_layer:
                with self.subTest(ind=ind, layerind=layer_ind):
                    coord = (layer_ind, ind)
                    self.assertAlmostEqual(mat[coord], 0)
    
    def test_covmat_for_no_fermions(self):
        """Ensure the correct covariance matrix is generated when t = 0.
        """
        self.system_z2.cfg.make_pure_gauge()
        covmat_layer1 = self.system_z2.compute_ferm_cov(layer = 0)
        covmat_layer2 = self.system_z2.compute_ferm_cov(layer = 1)
        expected_covmat = np.array([[ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                                    [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                                    [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.]])
        self.assertTrue(np.allclose(covmat_layer1, expected_covmat))
        self.assertTrue(np.allclose(covmat_layer2, expected_covmat))
    
    def test_covmat_with_fermions(self):
        """Ensure the covariance matrix is not the pure-gauge one when t != 0.
        This test must be done with a gauge configuration that includes some flux.
        Only the fermionic layer should have a covariance matrix different than the pure-gauge one.
        """
        config = np.array([0]*7 + [np.pi]*1)
        self.system_z2.update_gauge_full_system(config)

        covmat_layer1 = self.system_z2.compute_ferm_cov(layer = 0)
        covmat_layer2 = self.system_z2.compute_ferm_cov(layer = 1)
        expected_covmat = np.array([[ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                                    [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
                                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                                    [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.]])
        self.assertTrue(np.allclose(covmat_layer1, expected_covmat))
        self.assertFalse(np.allclose(covmat_layer2, expected_covmat))

    def test_t_zero(self):
        """Ensure mass and interaction energy are zero when t = 0"""

        self.system_z2.cfg.make_pure_gauge() # sets t params to zero
        ex_eval = exacteval.ExactEvaluator(self.system_z2)
        dest_dict = ex_eval.evaluate()
        self.assertTrue(np.allclose(0, dest_dict['mass_energy']))
        self.assertTrue(np.allclose(0, dest_dict['int_energy']))

    def test_t_nonzero(self):
        """Ensure mass and interaction energy are zero when t != 0. 
        This checks for random params, which we assume do not give t = 0"""
        
        ex_eval = exacteval.ExactEvaluator(self.system_z2)
        dest_dict = ex_eval.evaluate()
        self.assertFalse(np.allclose(0, dest_dict['mass_energy']))
        self.assertFalse(np.allclose(0, dest_dict['int_energy']))

  
    def test_free_fermions_gs_energy(self):
        """Ensure gs energy for free fermion case matches the analytic result"""
        pass

    # TESTS TO ADD
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
        paramvec = np.random.rand(2,20)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_G2C_F2C_Config(lat_2x2, 1.0, 0.0, 0.0, 0.0, None)
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D_G2C_F2C(system_cfg)
        deriv_ana = system_z2_2_2.el_energy_op_grad_vec
        symbolvec = system_z2_2_2.symbolvec
        for layerind in range(2):
            for ind in range(len(symbolvec)):
                with self.subTest(symbol=symbolvec[ind], layerind=layerind):
                    paramvec_left = np.copy(paramvec)
                    paramvec_right = np.copy(paramvec)
                    paramvec_left[layerind, ind] -= eps
                    paramvec_right[layerind, ind] += eps
                    system_cfg_left = system.Z2System2D_G2C_F2C_Config(lat_2x2, 1.0, 0.0, 0.0, 0.0, None)
                    system_cfg_right = system.Z2System2D_G2C_F2C_Config(lat_2x2, 1.0, 0.0, 0.0, 0.0, None)

                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    system_z2_2_2_left = system.Z2System2D_G2C_F2C(system_cfg_left)
                    system_z2_2_2_right = system.Z2System2D_G2C_F2C(system_cfg_right)

                    val_left = system_z2_2_2_left.el_energy_op
                    val_right = system_z2_2_2_right.el_energy_op
                    deriv_num = (val_right - val_left) / (2 * eps)

                    print(f"left: {val_left}, right: {val_right}")
                    print(f"symbol: {symbolvec[ind]}, analytic: {deriv_ana[layerind,ind]}, numerical: {deriv_num}")
                    self.assertAlmostEqual(deriv_ana[layerind,ind], deriv_num, places=5)

    @skip("This gradient tests with the 4 copy ansatz take to long")
    def test_grad_el_energy_4C(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # for the 4 copy fermionic ansatz
        eps = 1e-5
        paramvec = np.random.rand(2,52)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_G2C_F4C_Config(lat_2x2, 1.0, 0.0, 0.0, 0.0, None)
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D_G2C_F4C(system_cfg)
        deriv_ana = system_z2_2_2.el_energy_op_grad_vec
        symbolvec = system_z2_2_2.symbolvec
        for layerind in range(2):
            for ind in range(len(symbolvec)):
                with self.subTest(symbol=symbolvec[ind], layerind=layerind):
                    paramvec_left = np.copy(paramvec)
                    paramvec_right = np.copy(paramvec)
                    paramvec_left[layerind, ind] -= eps
                    paramvec_right[layerind, ind] += eps
                    system_cfg_left = system.Z2System2D_G2C_F4C_Config(lat_2x2, 1.0, 0.0, 0.0, 0.0, None)
                    system_cfg_right = system.Z2System2D_G2C_F4C_Config(lat_2x2, 1.0, 0.0, 0.0, 0.0, None)

                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    system_z2_2_2_left = system.Z2System2D_G2C_F4C(system_cfg_left)
                    system_z2_2_2_right = system.Z2System2D_G2C_F4C(system_cfg_right)

                    val_left = system_z2_2_2_left.el_energy_op
                    val_right = system_z2_2_2_right.el_energy_op
                    deriv_num = (val_right - val_left) / (2 * eps)

                    self.assertAlmostEqual(deriv_ana[layerind,ind], deriv_num, places=5)
    

    def test_grad_mass_energy_2C(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # for the 2 copy fermionic ansatz
        eps = 1e-5
        paramvec = np.random.rand(2,20)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_G2C_F2C_Config(lat_2x2, 0.0, 0.0, 0.0, 1.0, None)
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D_G2C_F2C(system_cfg)
 
        config = np.array([0]*7 + [np.pi]*1)
        system_z2_2_2.update_gauge_full_system(config)

        deriv_ana = system_z2_2_2.mass_energy_op_grad_vec
        symbolvec = system_z2_2_2.symbolvec
        for layerind in range(2):
            # we could skip the first layer, since the first layer does not contribute to the
            # mass energy
            for ind in range(len(symbolvec)):
                with self.subTest(symbol=symbolvec[ind], layerind=layerind):
                    paramvec_left = np.copy(paramvec)
                    paramvec_right = np.copy(paramvec)
                    paramvec_left[layerind, ind] -= eps
                    paramvec_right[layerind, ind] += eps
                    system_cfg_left = system.Z2System2D_G2C_F2C_Config(lat_2x2, 0.0, 0.0, 1.0, 1.0, None)
                    system_cfg_right = system.Z2System2D_G2C_F2C_Config(lat_2x2, 0.0, 0.0, 1.0, 1.0, None)

                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    system_z2_2_2_left = system.Z2System2D_G2C_F2C(system_cfg_left)
                    system_z2_2_2_right = system.Z2System2D_G2C_F2C(system_cfg_right)
                    system_z2_2_2_left.update_gauge_full_system(config)
                    system_z2_2_2_right.update_gauge_full_system(config)

                    val_left = system_z2_2_2_left.mass_energy_op
                    val_right = system_z2_2_2_right.mass_energy_op
                    deriv_num = (val_right - val_left) / (2 * eps)

                    print(f"left: {val_left}, right: {val_right}")
                    print(f"symbol: {symbolvec[ind]}, analytic: {deriv_ana[layerind,ind]}, numerical: {deriv_num}")
                    self.assertAlmostEqual(deriv_ana[layerind,ind], deriv_num, places=5)

    @skip("This gradient tests with the 4 copy ansatz take to long")
    def test_grad_mass_energy_4C(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # for the 4 copy fermionic ansatz
        eps = 1e-5
        paramvec = np.random.rand(2,52)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_G2C_F4C_Config(lat_2x2, 0.0, 0.0, 0.0, 1.0, None)
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D_G2C_F4C(system_cfg)

        config = np.array([0]*7 + [np.pi]*1)
        system_z2_2_2.update_gauge_full_system(config)

        deriv_ana = system_z2_2_2.mass_energy_op_grad_vec
        symbolvec = system_z2_2_2.symbolvec
        for layerind in range(1,2):
            # we skip the first layer, since the first layer does not contribute to the
            # mass energy, and is less important to test
            for ind in range(len(symbolvec)):
                with self.subTest(symbol=symbolvec[ind], layerind=layerind):
                    paramvec_left = np.copy(paramvec)
                    paramvec_right = np.copy(paramvec)
                    paramvec_left[layerind, ind] -= eps
                    paramvec_right[layerind, ind] += eps
                    system_cfg_left = system.Z2System2D_G2C_F4C_Config(lat_2x2, 0.0, 0.0, 0.0, 1.0, None)
                    system_cfg_right = system.Z2System2D_G2C_F4C_Config(lat_2x2, 0.0, 0.0, 0.0, 1.0, None)

                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    system_z2_2_2_left = system.Z2System2D_G2C_F4C(system_cfg_left)
                    system_z2_2_2_right = system.Z2System2D_G2C_F4C(system_cfg_right)
                    system_z2_2_2_left.update_gauge_full_system(config)
                    system_z2_2_2_right.update_gauge_full_system(config)

                    val_left = system_z2_2_2_left.mass_energy_op
                    val_right = system_z2_2_2_right.mass_energy_op
                    deriv_num = (val_right - val_left) / (2 * eps)

                    print(f"left: {val_left}, right: {val_right}")
                    print(f"symbol: {symbolvec[ind]}, analytic: {deriv_ana[layerind,ind]}, numerical: {deriv_num}")
                    self.assertAlmostEqual(deriv_ana[layerind,ind], deriv_num, places=5)
    

    def test_grad_int_energy_2C(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # for the 2 copy fermionic ansatz
        eps = 1e-5
        paramvec = np.random.rand(2,20)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_G2C_F2C_Config(lat_2x2, 0.0, 0.0, 1.0, 0.0, None)
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D_G2C_F2C(system_cfg)

        # the interaction energy vanishes for the default configuration (no flux on any link)
        # so we choose a configuration where we know the interaction energy is not negligible 
        config = np.array([0]*7 + [np.pi]*1)
        system_z2_2_2.update_gauge_full_system(config)

        deriv_ana = system_z2_2_2.int_energy_op_grad_vec
        symbolvec = system_z2_2_2.symbolvec
        for layerind in range(2):
            # we could skip the first layer, since the first layer does not contribute to the
            # interaction energy
            for ind in range(len(symbolvec)):
                with self.subTest(symbol=symbolvec[ind], layerind=layerind):
                    paramvec_left = np.copy(paramvec)
                    paramvec_right = np.copy(paramvec)
                    paramvec_left[layerind, ind] -= eps
                    paramvec_right[layerind, ind] += eps
                    system_cfg_left = system.Z2System2D_G2C_F2C_Config(lat_2x2, 0.0, 0.0, 1.0, 0.0, None)
                    system_cfg_right = system.Z2System2D_G2C_F2C_Config(lat_2x2, 0.0, 0.0, 1.0, 0.0, None)

                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    system_z2_2_2_left = system.Z2System2D_G2C_F2C(system_cfg_left)
                    system_z2_2_2_right = system.Z2System2D_G2C_F2C(system_cfg_right)
                    system_z2_2_2_left.update_gauge_full_system(config)
                    system_z2_2_2_right.update_gauge_full_system(config)

                    val_left = system_z2_2_2_left.int_energy_op
                    val_right = system_z2_2_2_right.int_energy_op
                    deriv_num = (val_right - val_left) / (2 * eps)

                    print(f"left: {val_left}, right: {val_right}")
                    print(f"symbol: {symbolvec[ind]}, analytic: {deriv_ana[layerind,ind]}, numerical: {deriv_num}")
                    self.assertAlmostEqual(deriv_ana[layerind,ind], deriv_num, places=5)
    
    @skip("This gradient tests with the 4 copy ansatz take to long")
    def test_grad_int_energy_4C(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # for the 4 copy fermionic ansatz
        eps = 1e-5
        paramvec = np.random.rand(2,52)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_G2C_F4C_Config(lat_2x2, 0.0, 0.0, 1.0, 0.0, None)
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D_G2C_F4C(system_cfg)

        # the interaction energy vanishes for the default configuration (no flux on any link)
        # so we choose a configuration where we know the interaction energy is not negligible 
        config = np.array([0]*7 + [np.pi]*1)
        system_z2_2_2.update_gauge_full_system(config)

        deriv_ana = system_z2_2_2.int_energy_op_grad_vec
        symbolvec = system_z2_2_2.symbolvec
        for layerind in range(1,2):
            # we skip the first layer, since the first layer does not contribute to the
            # interaction energy, and is less important to test
            for ind in range(len(symbolvec)):
                with self.subTest(symbol=symbolvec[ind], layerind=layerind):
                    paramvec_left = np.copy(paramvec)
                    paramvec_right = np.copy(paramvec)
                    paramvec_left[layerind, ind] -= eps
                    paramvec_right[layerind, ind] += eps
                    system_cfg_left = system.Z2System2D_G2C_F4C_Config(lat_2x2, 0.0, 0.0, 1.0, 0.0, None)
                    system_cfg_right = system.Z2System2D_G2C_F4C_Config(lat_2x2, 0.0, 0.0, 1.0, 0.0, None)

                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    system_z2_2_2_left = system.Z2System2D_G2C_F4C(system_cfg_left)
                    system_z2_2_2_right = system.Z2System2D_G2C_F4C(system_cfg_right)
                    system_z2_2_2_left.update_gauge_full_system(config)
                    system_z2_2_2_right.update_gauge_full_system(config)

                    val_left = system_z2_2_2_left.int_energy_op
                    val_right = system_z2_2_2_right.int_energy_op
                    deriv_num = (val_right - val_left) / (2 * eps)

                    print(f"left: {val_left}, right: {val_right}")
                    print(f"symbol: {symbolvec[ind]}, analytic: {deriv_ana[layerind,ind]}, numerical: {deriv_num}")
                    self.assertAlmostEqual(deriv_ana[layerind,ind], deriv_num, places=5)