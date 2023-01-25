import unittest 
from unittest import skip
import numpy as np

from ggpeps import lattice
from ggpeps.lattice import Direction
from ggpeps import system
from ggpeps import utils
from ggpeps.mc import MonteCarloEstimatorConfig, MonteCarloEstimator, MonteCarloManager
from ggpeps.utils import compare_array_elementwise

# ======================= Z2 fermionic system (4 copies) =========================================

class TestZ2C4System(unittest.TestCase):
    def setUp(self):

        lat = lattice.Lattice2D(2,2)
        paramvec = np.random.rand(2, 20)
        cfg = system.Z2System2D4C_Config(lat, 0, 0, 0, 0)
        cfg.paramvec = paramvec
        self.system_z2 = system.Z2System2D4C(cfg) 
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

    # TESTS TO ADD
    # get the correct gamma_in_sys for all layers
    # mass energy is zero when t = 0
    # int energy is zero when t = 0
    # when interaction is off, ground state is: no fermions, pure-gauge ground state
    # ensure covmat is not the no-fermions one in cases where t != 0
    # required parameters are zero (i.e. the ones that are zero by def of the ansatz) - before starting, and remain that way through minimization
