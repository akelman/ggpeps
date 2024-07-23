import unittest 
from unittest import skip

import numpy as np

from ggpeps import lattice
from ggpeps import system, exacteval

from ggpeps.mc import MonteCarloEvaluatorConfig, MonteCarloEvaluator


# ======================= Z2 fermionic system (4 copies) =========================================

class Testgaugefixing(unittest.TestCase):

    def setUp(self):
        self.lat2 = lattice.Lattice2D(2,2)
        self.lat4 = lattice.Lattice2D(4,4)
        
        eval_cfg = None
        paramvec = np.random.rand(2, 20)
        
        # Build 2x2 system and evaluator
        cfg2 = system.Z2System2D_G2C_F2C_Config(self.lat2, 1,1,1,1)
        cfg2.paramvec = paramvec
        self.system_z2_2 = system.Z2System2D_G2C_F2C(cfg2) 
        self.system_z2_2.cfg.enforce_parameter_conditions(self.system_z2_2.cfg.paramvec)
        self.evaluator2 = exacteval.ExactEvaluator(eval_cfg, self.system_z2_2, False)
        self.configvec2 = self.evaluator2.generate_config_vec()
        self.neutral_gauge2 = self.system_z2_2.gaugemgr.get_neutral_gauge_value()        

        # Build 4x4 system and evaluator
        cfg4 = system.Z2System2D_G2C_F2C_Config(self.lat4, 1,1,1,1)
        self.system_z2_4 = system.Z2System2D_G2C_F2C(cfg4) 
        cfg4.paramvec = paramvec
        self.system_z2_4.cfg.enforce_parameter_conditions(self.system_z2_4.cfg.paramvec)
        self.evaluator4 = exacteval.ExactEvaluator(eval_cfg, self.system_z2_4, False)
        self.configvec4 = self.evaluator4.generate_config_vec()
        self.netural_gauge4 = self.system_z2_4.gaugemgr.get_neutral_gauge_value() 

    def test_configvec_2x2(self):
        """Ensure that the configvec for gauge fixing is generated correctly. 
        Ensure that the links in the tree are set to the unity in all configurations 
        and that all configurations are unique.
        """

        self.assertEqual(len(self.configvec2), 2**self.lat2.ncomptreelinks) 

        tuple_configvec2 = [] # converting each configuration in configvec to a tuple - because it's hashable
        for config in self.configvec2: # 2x2 lattice
            tuple_configvec2.append(tuple(config)) 
            for link in self.lat2.maximal_tree:
                self.assertEqual(config[link], self.neutral_gauge2)

        unique_configvec2 = set(tuple_configvec2) # configvec with unique combinations only
        self.assertEqual(len(tuple_configvec2), len(unique_configvec2)) # assert that there are no repeated configurations
    
    def test_configvec_4x4(self):
        """Test configvec for 4x4 lattice"""

        self.assertEqual(len(self.configvec4), 2**self.lat4.ncomptreelinks) 

        tuple_configvec4 = [] 
        for config in self.configvec4: #4x4 lattice
            tuple_configvec4.append(tuple(config))
            for link in self.lat4.maximal_tree:
                self.assertEqual(config[link], self.netural_gauge4)
        
        unique_configvec4 = set(tuple_configvec4) # configvec with unique combinations only
        self.assertEqual(len(tuple_configvec4), len(unique_configvec4))
    
    def test_exacteval(self):
        """Ensure that exact evaluation gives the same results with and without gauge fixing"""

        self.evaluator2.gauge_fixing = False
        no_gauge_fixing_eval = self.evaluator2.evaluate()

        self.evaluator2.obsdict = None
        self.evaluator2.gauge_fixing = True
        gauge_fixing_eval = self.evaluator2.evaluate()

        for key, val in no_gauge_fixing_eval.items():
            self.assertTrue(np.allclose(val, gauge_fixing_eval[key]))

    # @skip("Too long")
    def test_mceval(self):

        # MC config
        mc_config = MonteCarloEvaluatorConfig()
        mc_config.warmup_steps = 20000 #20000
        mc_config.meas_steps = 20000 #40000
        mc_config.binsize = 1
        mc_config.update_size_per_step = 2
        
        # MC evaluators - with and without gauge fixing
        cfg = system.Z2System2D_G2C_F2C_Config(self.lat2, 1,1,1,1)
        cfg.paramvec = np.random.rand(2, 20)
        system_z2_2_A = system.Z2System2D_G2C_F2C(cfg) 
        system_z2_2_B = system.Z2System2D_G2C_F2C(cfg) 

        mc_evaluator_no_gf = MonteCarloEvaluator(mc_config, system_z2_2_A, gauge_fixing=False)
        mc_evaluator_gf = MonteCarloEvaluator(mc_config, system_z2_2_B, gauge_fixing=True)

        # Evaluations
        res_no_gf = mc_evaluator_no_gf.evaluate()
        no_gauge_fixing_energy = mc_evaluator_no_gf.get_obs_mean("energy")

        res_gf = mc_evaluator_gf.evaluate()
        gauge_fixing_energy = mc_evaluator_gf.get_obs_mean("energy")

        self.assertAlmostEqual(gauge_fixing_energy,no_gauge_fixing_energy,places=0)

        #for key, val in res_no_gf.items():
        #    self.assertTrue(np.allclose(val, res_gf[key]))