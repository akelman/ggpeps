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
