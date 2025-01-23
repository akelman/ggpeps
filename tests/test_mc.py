import unittest
from unittest import skip

import numpy as np

from ggpeps import system, lattice
from ggpeps.minimizer import Minimizer, MinimizerConfig
from ggpeps.evaluator_manager import EvaluatorManager
from ggpeps.exacteval import ExactEvaluator, ExactEvaluatorConfig
from ggpeps.mc import MonteCarloEvaluator, MonteCarloEvaluatorConfig


class TestMC(unittest.TestCase):

    def setUp(self):
        paramvec = [[0.0, 0.5, 0.5, 0.0, 0.0, 0.0]]
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2DConfig(
            lat_2x2, 1.0, 0.0, 0.0, 0.0, [0], num_pg_layer=1, num_fermionic_layer=0
        )
        system_cfg.paramvec = paramvec
        self.sys = system.Z2System2D(system_cfg)

    def test_mc_observable_length_binsize1(self):
        """Ensure that the length of the observables is correct"""

        mc_cfg = MonteCarloEvaluatorConfig()
        mc_cfg.warmup_steps = 7
        mc_cfg.meas_steps = 4
        mc_cfg.binsize = 1
        mc_cfg.compute_grads = True
        mc_evaluator = MonteCarloEvaluator(mc_cfg, self.sys)
        mc_evaluator.evaluate()

        for obs in mc_evaluator.obsdict:

            datavec = mc_evaluator.obsdict[obs].datavec
            if obs == "acceptance_prob":
                self.assertEqual(len(datavec), mc_cfg.warmup_steps + mc_cfg.meas_steps)
            else:
                self.assertEqual(len(datavec), mc_cfg.meas_steps)

    def test_mc_observable_length_binsize5(self):
        """Ensure that the length of the observables is correct with binning during measurement"""

        binsize = 5
        mc_cfg = MonteCarloEvaluatorConfig()
        mc_cfg.warmup_steps = 70
        mc_cfg.meas_steps = 40
        mc_cfg.binsize = binsize
        mc_cfg.compute_grads = True
        mc_evaluator = MonteCarloEvaluator(mc_cfg, self.sys)
        mc_evaluator.evaluate()

        for obs in mc_evaluator.obsdict:

            datavec = mc_evaluator.obsdict[obs].datavec
            if obs == "acceptance_prob":
                self.assertEqual(
                    len(datavec) * binsize, mc_cfg.warmup_steps + mc_cfg.meas_steps
                )
            else:
                self.assertEqual(len(datavec) * binsize, mc_cfg.meas_steps)
