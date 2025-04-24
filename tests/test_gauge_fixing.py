import unittest
from unittest import skip

import numpy as np

from ggpeps import lattice
from ggpeps import system, exacteval

from ggpeps.exacteval import ExactEvaluatorConfig
from ggpeps.mc import MonteCarloEvaluatorConfig, MonteCarloEvaluator


# ======================= Z2 fermionic system (4 copies) =========================================


class Testgaugefixing(unittest.TestCase):

    def setUp(self):
        pass

    def test_configvec_2x2(self):
        """Ensure that the configvec for gauge fixing is generated correctly.
        Ensure that the links in the tree are set to the unity in all configurations
        and that all configurations are unique.
        """
        lat2 = lattice.Lattice2D(2, 2, -1)  # With gauge fixing
        paramvec = np.random.rand(2, 20)
        cfg2 = system.Z2System2D_G2C_F2C_Config(lat2, 1, 1, 1, 1, [0, 0])
        cfg2.paramvec = paramvec
        system_z2_2 = system.Z2System2D(cfg2)
        system_z2_2.cfg.enforce_parameter_conditions(system_z2_2.cfg.paramvec)

        evaluator2 = exacteval.ExactEvaluator(ExactEvaluatorConfig(), system_z2_2)
        configvec2 = [config for config in evaluator2.generate_config_vec()]
        neutral_gauge2 = system_z2_2.cfg.gaugemgr.get_neutral_gauge_value()

        self.assertEqual(len(configvec2), 2 ** (len(lat2.comp_tree)))

        tuple_configvec2 = []
        for config in configvec2:
            # Convert each numpy array in the config to a tuple
            tuple_config = tuple(tuple(arr.flatten()) for arr in config)
            tuple_configvec2.append(tuple_config)
            for link in lat2.fixed_tree:
                self.assertEqual(config[link], neutral_gauge2)
        unique_configvec2 = set(tuple_configvec2)
        self.assertEqual(len(tuple_configvec2), len(unique_configvec2))

    def test_configvec_4x4(self):
        """Test configvec for 4x4 lattice"""

        lat4 = lattice.Lattice2D(4, 4, -1)  # Without gauge fixing
        paramvec = np.random.rand(2, 20)
        cfg4 = system.Z2System2D_G2C_F2C_Config(lat4, 1, 1, 1, 1, [0, 0])
        cfg4.paramvec = paramvec
        system_z2_4 = system.Z2System2D(cfg4)
        system_z2_4.cfg.enforce_parameter_conditions(system_z2_4.cfg.paramvec)

        evaluator4 = exacteval.ExactEvaluator(ExactEvaluatorConfig(), system_z2_4)
        configvec4 = [config for config in evaluator4.generate_config_vec()]
        neutral_gauge4 = system_z2_4.cfg.gaugemgr.get_neutral_gauge_value()

        self.assertEqual(len(configvec4), 2 ** (len(lat4.comp_tree)))

        tuple_configvec4 = []
        for config in configvec4:
            tuple_config = tuple(tuple(arr.flatten()) for arr in config)
            tuple_configvec4.append(tuple_config)
            for link in lat4.fixed_tree:
                self.assertEqual(config[link], neutral_gauge4)

        # Ensure all configurations are unique
        unique_configvec4 = set(tuple_configvec4)
        self.assertEqual(len(tuple_configvec4), len(unique_configvec4))

    def test_exacteval(self):
        """Ensure that exact evaluation gives the same results with and without gauge fixing"""

        lat2_with_gf = lattice.Lattice2D(2, 2, -1)  # With gauge fixing
        lat2_without_gf = lattice.Lattice2D(2, 2)  # Without gauge fixing
        paramvec = np.random.rand(2, 20)

        # System with gauge fixing
        cfg_with_gf = system.Z2System2D_G2C_F2C_Config(lat2_with_gf, 1, 1, 1, 1, [0, 0])
        cfg_with_gf.paramvec = paramvec
        system_with_gf = system.Z2System2D(cfg_with_gf)
        system_with_gf.cfg.enforce_parameter_conditions(cfg_with_gf.paramvec)

        # System without gauge fixing
        cfg_without_gf = system.Z2System2D_G2C_F2C_Config(
            lat2_without_gf, 1, 1, 1, 1, [0, 0]
        )
        cfg_without_gf.paramvec = paramvec
        system_without_gf = system.Z2System2D(cfg_without_gf)
        system_without_gf.cfg.enforce_parameter_conditions(cfg_without_gf.paramvec)

        # Evaluation
        evaluator_with_gf = exacteval.ExactEvaluator(
            ExactEvaluatorConfig(), system_with_gf
        )
        evaluator_without_gf = exacteval.ExactEvaluator(
            ExactEvaluatorConfig(), system_without_gf
        )

        eval_with_gf = evaluator_with_gf.evaluate()
        eval_without_gf = evaluator_without_gf.evaluate()

        for key, val in eval_with_gf.items():
            self.assertTrue(np.allclose(val, eval_without_gf[key]))

    def test_gf_some_rows_exacteval(self):
        """Test exact evaluation when fixing only 1 row."""
        lat2_with_gf = lattice.Lattice2D(2, 2, 1)  # With gauge fixing
        lat2_without_gf = lattice.Lattice2D(2, 2)  # Without gauge fixing
        paramvec = np.random.rand(2, 20)

        # System with gauge fixing
        cfg_with_gf = system.Z2System2D_G2C_F2C_Config(lat2_with_gf, 1, 1, 1, 1, [0, 0])
        cfg_with_gf.paramvec = paramvec
        system_with_gf = system.Z2System2D(cfg_with_gf)
        system_with_gf.cfg.enforce_parameter_conditions(cfg_with_gf.paramvec)

        # System without gauge fixing
        cfg_without_gf = system.Z2System2D_G2C_F2C_Config(
            lat2_without_gf, 1, 1, 1, 1, [0, 0]
        )
        cfg_without_gf.paramvec = paramvec
        system_without_gf = system.Z2System2D(cfg_without_gf)
        system_without_gf.cfg.enforce_parameter_conditions(cfg_without_gf.paramvec)

        # Evaluation
        evaluator_with_gf = exacteval.ExactEvaluator(
            ExactEvaluatorConfig(), system_with_gf
        )
        evaluator_without_gf = exacteval.ExactEvaluator(
            ExactEvaluatorConfig(), system_without_gf
        )

        eval_with_gf = evaluator_with_gf.evaluate()
        eval_without_gf = evaluator_without_gf.evaluate()

        for key, val in eval_with_gf.items():
            self.assertTrue(np.allclose(val, eval_without_gf[key]))

    @skip("Too long")
    def test_mceval(self):
        """Ensure that MC evaluation gives the same results with and without gauge fixing"""
        # MC config
        lat2_with_gf = lattice.Lattice2D(2, 2, -1)  # With gauge fixing
        lat2_without_gf = lattice.Lattice2D(2, 2)  # Without gauge fixing
        paramvec = np.random.rand(2, 20)

        # Configuration
        cfg_with_gf = system.Z2System2D_G2C_F2C_Config(lat2_with_gf, 1, 1, 1, 1)
        cfg_with_gf.paramvec = paramvec
        cfg_without_gf = system.Z2System2D_G2C_F2C_Config(lat2_without_gf, 1, 1, 1, 1)
        cfg_without_gf.paramvec = paramvec

        system_with_gf = system.Z2System2D(cfg_with_gf)
        system_without_gf = system.Z2System2D(cfg_without_gf)

        # MC evaluation
        mc_config = MonteCarloEvaluatorConfig()
        mc_config.warmup_steps = 20000
        mc_config.meas_steps = 20000
        mc_config.binsize = 1
        mc_config.update_size_per_step = 2

        mc_evaluator_with_gf = MonteCarloEvaluator(mc_config, system_with_gf)
        mc_config.gauge_fixing = True
        mc_evaluator_without_gf = MonteCarloEvaluator(mc_config, system_without_gf)

        no_gauge_fixing_energy = mc_evaluator_with_gf.get_obs_mean("energy")
        gauge_fixing_energy = mc_evaluator_without_gf.get_obs_mean("energy")

        self.assertAlmostEqual(no_gauge_fixing_energy, gauge_fixing_energy, places=0)
