import unittest 
from unittest import skip
import numpy as np

from ggpeps import lattice
from ggpeps.lattice import Direction
from ggpeps import system, exacteval
from ggpeps import utils
from ggpeps.mc import MonteCarloEstimatorConfig, MonteCarloEstimator, MonteCarloManager
from ggpeps.system import Z2System2D_G2C_F2C_Config, Z2System2D_G2C_F2C
from ggpeps.minimizer import Minimizer, MinimizerConfig
from ggpeps.utils import compare_array_elementwise

# ======================= Z2 fermionic system (4 copies) =========================================

class TestZ2C4System(unittest.TestCase):
    def setUp(self):

        lat = lattice.Lattice2D(2,2)
        paramvec = np.random.rand(2, 20)
        cfg = system.Z2System2D_G2C_F2C_Config(lat, 1,1,1,1)
        cfg.paramvec = paramvec
        self.system_z2 = system.Z2System2D_G2C_F2C(cfg) 
        self.system_z2.cfg.enforce_parameter_conditions(self.system_z2.cfg.paramvec)   
        
    
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
    # ensure covmat is not the no-fermions one in cases where t != 0
    # required parameters are zero (i.e. the ones that are zero by def of the ansatz) - before starting, and remain that way through minimization

    # test site-specific mass values in limits where this is known



    ##

    def test_grad_int_energy_2C(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # for the 2 copy fermionic ansatz
        eps = 1e-5
        paramvec = np.random.rand(2,20)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_G2C_F2C_Config(lat_2x2, 0.0, 0.0, 1.0, 0.0)
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D_G2C_F2C(system_cfg)
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
                    system_cfg_left = system.Z2System2D_G2C_F2C_Config(lat_2x2, 0.0, 0.0, 1.0, 0.0)
                    system_cfg_right = system.Z2System2D_G2C_F2C_Config(lat_2x2, 0.0, 0.0, 1.0, 0.0)

                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    system_z2_2_2_left = system.Z2System2D_G2C_F2C(system_cfg_left)
                    system_z2_2_2_right = system.Z2System2D_G2C_F2C(system_cfg_right)

                    val_left = system_z2_2_2_left.int_energy_op
                    val_right = system_z2_2_2_right.int_energy_op
                    deriv_num = (val_right - val_left) / (2 * eps)

                    self.assertAlmostEqual(deriv_ana[layerind,ind], deriv_num, places=5)

    def test_grad_int_energy_4C(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # for the 4 copy fermionic ansatz
        eps = 1e-5
        paramvec = np.random.rand(2,52)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_G2C_F4C_Config(lat_2x2, 0.0, 0.0, 1.0, 0.0)
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D_G2C_F4C(system_cfg)
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
                    system_cfg_left = system.Z2System2D_G2C_F4C_Config(lat_2x2, 0.0, 0.0, 1.0, 0.0)
                    system_cfg_right = system.Z2System2D_G2C_F4C_Config(lat_2x2, 0.0, 0.0, 1.0, 0.0)

                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    system_z2_2_2_left = system.Z2System2D_G2C_F4C(system_cfg_left)
                    system_z2_2_2_right = system.Z2System2D_G2C_F4C(system_cfg_right)

                    val_left = system_z2_2_2_left.int_energy_op
                    val_right = system_z2_2_2_right.int_energy_op
                    deriv_num = (val_right - val_left) / (2 * eps)

                    self.assertAlmostEqual(deriv_ana[layerind,ind], deriv_num, places=5)