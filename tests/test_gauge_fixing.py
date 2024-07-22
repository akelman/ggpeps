import unittest 
from unittest import skip
import time


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

class Testgaugefixing(unittest.TestCase):
    def setUp(self):
        self.lat2 = lattice.Lattice2D(2,2)
        self.lat4 = lattice.Lattice2D(4,4)
        self.tree2 = self.lat2.generate_maximal_tree()
        self.tree4 = self.lat4.generate_maximal_tree()
        
        paramvec = np.random.rand(2, 20)
        
        cfg2 = system.Z2System2D_G2C_F2C_Config(self.lat2, 1,1,1,1)
        cfg2.paramvec = paramvec
        self.system_z2_2 = system.Z2System2D_G2C_F2C(cfg2) 
        self.system_z2_2.cfg.enforce_parameter_conditions(self.system_z2_2.cfg.paramvec)
        eval_cfg = None
        self.evaluator2 = exacteval.ExactEvaluator(eval_cfg, self.system_z2_2)
        self.configvec2 = self.evaluator2.generate_config_vec()
        self.netural_gauge2 = self.system_z2_2.gaugemgr.get_neutral_gauge_value()        

        cfg4 = system.Z2System2D_G2C_F2C_Config(self.lat4, 1,1,1,1)
        self.system_z2_4 = system.Z2System2D_G2C_F2C(cfg4) 
        cfg4.paramvec = paramvec
        self.system_z2_4.cfg.enforce_parameter_conditions(self.system_z2_4.cfg.paramvec)
        self.evaluator4 = exacteval.ExactEvaluator(eval_cfg, self.system_z2_4)
        self.configvec4 = self.evaluator4.generate_config_vec()
        self.netural_gauge4 = self.system_z2_4.gaugemgr.get_neutral_gauge_value() 

        warmup_steps = 2000
        meas_steps = 1000
        binsize = 1
        update_size_per_step = 2

        #define MC evaluator without gauge fixing
        mc_config = MonteCarloEvaluatorConfig()
        mc_config.warmup_steps = warmup_steps
        mc_config.meas_steps = meas_steps
        mc_config.binsize = binsize
        mc_config.update_size_per_step = update_size_per_step
        self.mc_evaluator = MonteCarloEvaluator(mc_config, self.system_z2_2)
        self.mc_evaluator.gauge_fixing = False

        #define MC evaluator without gauge fixing
        mc_config_gf = MonteCarloEvaluatorConfig()
        mc_config_gf.warmup_steps = warmup_steps
        mc_config_gf.meas_steps = meas_steps
        mc_config_gf.binsize = binsize
        mc_config_gf.update_size_per_step = update_size_per_step
        self.mc_evaluator_gf = MonteCarloEvaluator(mc_config_gf, self.system_z2_2)
        self.mc_evaluator_gf.gauge_fixing = True



    def test_configvec(self):
        """Ensure that the configvec for gauge fixing is generated correctly. Ensure that the links in the tree are set to the unity in all configurations 
        and that all configurations are unique."""
        self.assertEqual(len(self.configvec4),2**self.lat4.ncomptreelinks) 
        tuple_configvec2 = [] # converting each configuration in configvec to a tuple - because it's hashable
        for config in self.configvec2: #2x2 lattice
            tuple_configvec2.append(tuple(config)) 
            for link in self.tree2:
                self.assertEqual(config[link],self.netural_gauge2)

        unique_configvec2 = set(tuple_configvec2) # configvec with unique combinations only
        self.assertEqual(len(tuple_configvec2),len(unique_configvec2)) # assert that there are no repeated configurations
        
        # now for 4x4 lattice
        tuple_configvec4 = [] 
        for config in self.configvec4: #4x4 lattice
            tuple_configvec4.append(tuple(config))
            for link in self.tree4:
                self.assertEqual(config[link],self.netural_gauge4)
        unique_configvec4 = set(tuple_configvec4) # configvec with unique combinations only
        self.assertEqual(len(tuple_configvec4),len(unique_configvec4))
    
    def test_exacteval(self):
        """Ensure that exact evaluation gives the same results with and without gauge fixing"""
        self.evaluator2.gauge_fixing = False
        start_time = time.time()
        no_gauge_fixing_eval = self.evaluator2.evaluate()
        end_time = time.time()
        print("no gauge_fixing",end_time-start_time)

        self.evaluator2.obsdict = None
        self.evaluator2.gauge_fixing = True
        start_time = time.time()
        gauge_fixing_eval = self.evaluator2.evaluate()
        end_time = time.time()
        print("gauge_fixing",end_time-start_time)
        for k,val in no_gauge_fixing_eval.items():
            self.assertTrue(np.allclose(val, gauge_fixing_eval[k]))
    
    # def test_exacteval4(self):
    # """Ensure that exact evaluation gives the same results with and without gauge fixing, for 2x2. Running time too lomg for personal computer. """ 
    #     self.evaluator4.gauge_fixing = True
    #     start_time = time.time()
    #     no_gauge_fixing_eval = self.evaluator4.evaluate()
    #     end_time = time.time()
    #     print("gauge_fixing",end_time-start_time)

    #     self.evaluator4.obsdict = None
    #     self.evaluator4.gauge_fixing = False
    #     start_time = time.time()
    #     gauge_fixing_eval = self.evaluator4.evaluate()
    #     end_time = time.time()
    #     print("no gauge_fixing",end_time-start_time)
    #     for k,val in no_gauge_fixing_eval.items():
    #         self.assertTrue(np.allclose(val, gauge_fixing_eval[k]))

#    @skip("Too long")
    def test_mceval(self):
        start = time.time()
        self.mc_evaluator.evaluate()
        no_gauge_fixing_energy = self.mc_evaluator.get_obs_mean("energy")
        end= time.time()
        print(no_gauge_fixing_energy,"no gf",end-start)
        start = time.time()
        self.mc_evaluator_gf.evaluate()
        gauge_fixing_energy = self.mc_evaluator_gf.get_obs_mean("energy")
        end = time.time()
        print(gauge_fixing_energy,"gf",end-start)
        self.assertAlmostEqual(gauge_fixing_energy,no_gauge_fixing_energy,places=0)

        # for k,val in no_gauge_fixing_eval.items():
        #     self.assertTrue(np.allclose(val, gauge_fixing_eval[k]))