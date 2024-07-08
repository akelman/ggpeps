import unittest 
from unittest import skip

import numpy as np

from ggpeps import utils
from ggpeps import lattice
from ggpeps import system, exacteval

from ggpeps.lattice import Direction
from ggpeps.mc import MonteCarloEvaluatorConfig, MonteCarloEvaluator
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
    
    def test_maximal_tree_generation(self):
        pass #TODO

    def test_complementary_maximal_tree_generation(self):
        pass #TODO

    def test_configvec(self):
        pass #TODO

    


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
