import unittest
from unittest import skip

import numpy as np

from ggpeps import system, lattice
from ggpeps.minimizer import Minimizer, MinimizerConfig
from ggpeps.evaluator_manager import EvaluatorManager
from ggpeps.exacteval import ExactEvaluator, ExactEvaluatorConfig
from ggpeps.mc import MonteCarloEvaluator, MonteCarloEvaluatorConfig


# @skip("This class of tests takes too long")
class TestMinimizerZ2(unittest.TestCase):

    def test_derivative_mag_energy_exact_1_layer(self):
        eps = 1e-5
        paramvec = [[0.0, 0.5, 0.5, 0.0, 0.0, 0.0]]
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2DConfig(
            lat_2x2, 1.0, 0.0, 0.0, 0.0, [0], num_pg_layer=1, num_fermionic_layer=0
        )
        system_cfg.paramvec = paramvec
        sys = system.Z2System2D(system_cfg)
        exact_cfg = ExactEvaluatorConfig()
        exact_ev = ExactEvaluator(exact_cfg, sys)
        res = exact_ev.evaluate()

        for ind in range(3):
            with self.subTest(ind=ind):
                paramvec = system_cfg.paramvec
                paramvec_left = np.copy(paramvec)
                paramvec_right = np.copy(paramvec)
                paramvec_left[0, ind] -= eps
                paramvec_right[0, ind] += eps
                system_cfg_left = system.Z2System2DConfig(
                    lat_2x2,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    [0],
                    num_pg_layer=1,
                    num_fermionic_layer=0,
                )
                system_cfg_right = system.Z2System2DConfig(
                    lat_2x2,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    [0],
                    num_pg_layer=1,
                    num_fermionic_layer=0,
                )

                system_cfg_left.paramvec = paramvec_left
                system_cfg_right.paramvec = paramvec_right

                sys_left = system.Z2System2D(system_cfg_left)
                sys_right = system.Z2System2D(system_cfg_right)

                exact_ev_left = ExactEvaluator(exact_cfg, sys_left)
                exact_ev_right = ExactEvaluator(exact_cfg, sys_right)

                res_left = exact_ev_left.evaluate()
                res_right = exact_ev_right.evaluate()

                mag_energy_deriv_num = (
                    res_right["mag_energy"] - res_left["mag_energy"]
                ) / (2 * eps)
                mag_energy_deriv_ana = res["mag_energy_grad"][0, ind]
                self.assertAlmostEqual(mag_energy_deriv_num, mag_energy_deriv_ana)

    def test_derivative_mag_energy_exact_2_layer(self):
        eps = 1e-5
        nlayer = 2
        paramvec = [[0.0, 0.5, 0.5, 0.0, 0.0, 0.0], [0.0, 0.3, 0.8, 0.0, 0.0, 0.0]]
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2DConfig(
            lat_2x2, 1.0, 0.0, 0.0, 0.0, [0, 0], num_pg_layer=2, num_fermionic_layer=0
        )
        system_cfg.paramvec = paramvec
        sys = system.Z2System2D(system_cfg)
        exact_cfg = ExactEvaluatorConfig()
        exact_ev = ExactEvaluator(exact_cfg, sys)
        res = exact_ev.evaluate()

        for layer in range(nlayer):
            for ind in range(3):
                with self.subTest(ind=ind, layer=layer):
                    paramvec = system_cfg.paramvec
                    paramvec_left = np.copy(paramvec)
                    paramvec_right = np.copy(paramvec)
                    paramvec_left[layer, ind] -= eps
                    paramvec_right[layer, ind] += eps
                    system_cfg_left = system.Z2System2DConfig(
                        lat_2x2,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        [0, 0],
                        num_pg_layer=2,
                        num_fermionic_layer=0,
                    )
                    system_cfg_right = system.Z2System2DConfig(
                        lat_2x2,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        [0, 0],
                        num_pg_layer=2,
                        num_fermionic_layer=0,
                    )

                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    sys_left = system.Z2System2D(system_cfg_left)
                    sys_right = system.Z2System2D(system_cfg_right)

                    exact_ev_left = ExactEvaluator(exact_cfg, sys_left)
                    exact_ev_right = ExactEvaluator(exact_cfg, sys_right)

                    res_left = exact_ev_left.evaluate()
                    res_right = exact_ev_right.evaluate()

                    mag_energy_deriv_num = (
                        res_right["mag_energy"] - res_left["mag_energy"]
                    ) / (2 * eps)
                    mag_energy_deriv_ana = res["mag_energy_grad"][layer, ind]

                    self.assertAlmostEqual(mag_energy_deriv_num, mag_energy_deriv_ana)

    def test_derivative_el_energy_exact_1_layer(self):
        eps = 1e-5
        paramvec = [[0.2, 0.5, 0.5, 0.0, 0.0, 0.0]]
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2DConfig(lat_2x2, 1.0, 0.0, 0.0, 0.0, [0])
        system_cfg.paramvec = paramvec
        sys = system.Z2System2D(system_cfg)
        exact_cfg = ExactEvaluatorConfig()
        exact_ev = ExactEvaluator(exact_cfg, sys)
        res = exact_ev.evaluate()

        for ind in range(3):
            with self.subTest(ind=ind):
                paramvec_left = np.copy(paramvec)
                paramvec_right = np.copy(paramvec)
                paramvec_left[0, ind] -= eps
                paramvec_right[0, ind] += eps
                system_cfg_left = system.Z2System2DConfig(
                    lat_2x2, 1.0, 0.0, 0.0, 0.0, [0]
                )
                system_cfg_right = system.Z2System2DConfig(
                    lat_2x2, 1.0, 0.0, 0.0, 0.0, [0]
                )

                system_cfg_left.paramvec = paramvec_left
                system_cfg_right.paramvec = paramvec_right

                sys_left = system.Z2System2D(system_cfg_left)
                sys_right = system.Z2System2D(system_cfg_right)

                exact_ev_left = ExactEvaluator(exact_cfg, sys_left)
                exact_ev_right = ExactEvaluator(exact_cfg, sys_right)

                res_left = exact_ev_left.evaluate()
                res_right = exact_ev_right.evaluate()

                el_energy_deriv_num = (
                    res_right["el_energy"] - res_left["el_energy"]
                ) / (2 * eps)
                el_energy_deriv_ana = res["el_energy_grad"][0, ind]

                self.assertAlmostEqual(
                    el_energy_deriv_num, el_energy_deriv_ana, places=5
                )

    def test_derivative_el_energy_exact_2_layer(self):
        eps = 1e-5
        nlayer = 2
        paramvec = [[0.2, 0.5, 0.5, 0.0, 0.0, 0.0], [0.1, 0.4, 0.2, 0.0, 0.0, 0.0]]
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2DConfig(
            lat_2x2, 1.0, 0.0, 0.0, 0.0, [0, 0], num_pg_layer=2, num_fermionic_layer=0
        )
        system_cfg.paramvec = paramvec
        sys = system.Z2System2D(system_cfg)
        exact_cfg = ExactEvaluatorConfig()
        exact_ev = ExactEvaluator(exact_cfg, sys)
        res = exact_ev.evaluate()

        for layerind in range(nlayer):
            for ind in range(3):
                with self.subTest(ind=ind, layerind=layerind):
                    paramvec_left = np.copy(paramvec)
                    paramvec_right = np.copy(paramvec)
                    paramvec_left[layerind, ind] -= eps
                    paramvec_right[layerind, ind] += eps
                    system_cfg_left = system.Z2System2DConfig(
                        lat_2x2,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        [0, 0],
                        num_pg_layer=2,
                        num_fermionic_layer=0,
                    )
                    system_cfg_right = system.Z2System2DConfig(
                        lat_2x2,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        [0, 0],
                        num_pg_layer=2,
                        num_fermionic_layer=0,
                    )

                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    sys_left = system.Z2System2D(system_cfg_left)
                    sys_right = system.Z2System2D(system_cfg_right)

                    exact_ev_left = ExactEvaluator(exact_cfg, sys_left)
                    exact_ev_right = ExactEvaluator(exact_cfg, sys_right)

                    res_left = exact_ev_left.evaluate()
                    res_right = exact_ev_right.evaluate()

                    el_energy_deriv_num = (
                        res_right["el_energy"] - res_left["el_energy"]
                    ) / (2 * eps)
                    el_energy_deriv_ana = res["el_energy_grad"][layerind, ind]

                    self.assertAlmostEqual(
                        el_energy_deriv_num, el_energy_deriv_ana, places=6
                    )

    @skip("Too long")
    def test_derivative_mag_energy_y(self):
        eps = 1e-4
        paramvec = [[0.1, 0.3, 1.4, 0.0, 0.0, 0.0]]
        paramvec_left = [[0.1, 0.3 - eps, 1.4, 0.0, 0.0, 0.0]]
        paramvec_right = [[0.1, 0.3 + eps, 1.4, 0.0, 0.0, 0.0]]
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2DConfig(lat_2x2, 1.0, 0.0, 0.0, 0.0)
        system_cfg_left = system.Z2System2DConfig(lat_2x2, 1.0, 0.0, 0.0, 0.0)
        system_cfg_right = system.Z2System2DConfig(lat_2x2, 1.0, 0.0, 0.0, 0.0)
        system_cfg.paramvec = paramvec
        system_cfg_left.paramvec = paramvec_left
        system_cfg_right.paramvec = paramvec_right
        sys_left = system.Z2System2D(system_cfg_left)
        sys_right = system.Z2System2D(system_cfg_right)

        mc_config = MonteCarloEvaluatorConfig()
        mc_config.warmup_steps = 1000
        mc_config.meas_steps = 10000
        mc_config.binsize = 1
        mc_config.compute_grads = True
        mc_config.gauge_fixing = False

        min_config = MinimizerConfig()

        mc_mgr = EvaluatorManager(system.Z2System2D, system_cfg, mc_config, 0)
        minimizer = Minimizer(min_config, mc_mgr)
        mc_left = MonteCarloEvaluator(mc_config, sys_left)
        mc_right = MonteCarloEvaluator(mc_config, sys_right)

        minimizer.last_result = minimizer.evaluator_manager.simulate()
        mc_left.evaluate()
        mc_right.evaluate()

        # mag_energy_deriv = minimizer.energy_gradient(minimizer.last_result)
        mag_energy_deriv = (
            minimizer.last_result.energy_gradient_mc()
        )  # get_obs_mean('energy_grad')
        mag_energy_left = mc_left.get_obs_mean("mag_energy")
        mag_energy_right = mc_right.get_obs_mean("mag_energy")

        mag_energy_deriv_num = (mag_energy_right - mag_energy_left) / (2 * eps)

        self.assertAlmostEqual(mag_energy_deriv[1], mag_energy_deriv_num)
