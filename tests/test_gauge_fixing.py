import unittest
from unittest import skip

import numpy as np

from ggpeps import lattice
from ggpeps import system, exacteval, utils

from ggpeps.exacteval import ExactEvaluatorConfig
from ggpeps.mc import MonteCarloEvaluatorConfig, MonteCarloEvaluator
from ggpeps.evaluator_manager import EvaluatorManager


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
        cfg2 = system.Z2System2D_Config(lat2, 1, 1, 1, 1, None, ncopy=2)
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
        cfg4 = system.Z2System2D_Config(lat4, 1, 1, 1, 1, None, ncopy=2)
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
        """Ensure that exact evaluation gives the same results with and without
        gauge fixing"""

        lat2_with_gf = lattice.Lattice2D(2, 2, -1)  # With gauge fixing
        lat2_without_gf = lattice.Lattice2D(2, 2)  # Without gauge fixing
        paramvec = np.random.rand(2, 20)

        # System with gauge fixing
        cfg_with_gf = system.Z2System2D_Config(lat2_with_gf, 1, 1, 1, 1, None, ncopy=2)
        cfg_with_gf.paramvec = paramvec
        system_with_gf = system.Z2System2D(cfg_with_gf)
        system_with_gf.cfg.enforce_parameter_conditions(cfg_with_gf.paramvec)

        # System without gauge fixing
        cfg_without_gf = system.Z2System2D_Config(lat2_without_gf, 1, 1, 1, 1, None, ncopy=2)
        cfg_without_gf.paramvec = paramvec
        system_without_gf = system.Z2System2D(cfg_without_gf)
        system_without_gf.cfg.enforce_parameter_conditions(cfg_without_gf.paramvec)

        # Evaluation
        evaluator_with_gf = exacteval.ExactEvaluator(ExactEvaluatorConfig(), system_with_gf)
        evaluator_without_gf = exacteval.ExactEvaluator(ExactEvaluatorConfig(), system_without_gf)

        evaluator_with_gf.evaluate()
        evaluator_without_gf.evaluate()
        eval_with_gf = evaluator_with_gf.obsdict
        eval_without_gf = evaluator_without_gf.obsdict

        for key, val in eval_with_gf.items():
            self.assertTrue(np.allclose(val, eval_without_gf[key], equal_nan=True))

    def test_gf_some_rows_exacteval(self):
        """Test exact evaluation when fixing only 1 row."""
        lat2_with_gf = lattice.Lattice2D(2, 2, 1)  # With gauge fixing
        lat2_without_gf = lattice.Lattice2D(2, 2)  # Without gauge fixing
        paramvec = np.random.rand(2, 20)

        # System with gauge fixing
        cfg_with_gf = system.Z2System2D_Config(lat2_with_gf, 1, 1, 1, 1, None, ncopy=2)
        cfg_with_gf.paramvec = paramvec
        system_with_gf = system.Z2System2D(cfg_with_gf)
        system_with_gf.cfg.enforce_parameter_conditions(cfg_with_gf.paramvec)

        # System without gauge fixing
        cfg_without_gf = system.Z2System2D_Config(lat2_without_gf, 1, 1, 1, 1, None, ncopy=2)
        cfg_without_gf.paramvec = paramvec
        system_without_gf = system.Z2System2D(cfg_without_gf)
        system_without_gf.cfg.enforce_parameter_conditions(cfg_without_gf.paramvec)

        # Evaluation
        evaluator_with_gf = exacteval.ExactEvaluator(ExactEvaluatorConfig(), system_with_gf)
        evaluator_without_gf = exacteval.ExactEvaluator(ExactEvaluatorConfig(), system_without_gf)

        evaluator_with_gf.evaluate()
        evaluator_without_gf.evaluate()
        eval_with_gf = evaluator_with_gf.obsdict
        eval_without_gf = evaluator_without_gf.obsdict

        for key, val in eval_with_gf.items():
            self.assertTrue(np.allclose(val, eval_without_gf[key], equal_nan=True))

    def test_exacteval_chess(self):
        """Ensure that exact evaluation gives the same results with gauge fixing like a chess board and without"""

        lat2_with_gf = lattice.Lattice2D(2, 2, -2)  # With gauge fixing
        lat2_without_gf = lattice.Lattice2D(2, 2)  # Without gauge fixing
        paramvec = np.random.rand(2, 20)

        # System with gauge fixing
        cfg_with_gf = system.Z2System2D_Config(lat2_with_gf, 1, 1, 1, 1, [0], ncopy=2)
        cfg_with_gf.paramvec = paramvec
        system_with_gf = system.Z2System2D(cfg_with_gf)
        system_with_gf.cfg.enforce_parameter_conditions(cfg_with_gf.paramvec)

        # System without gauge fixing
        cfg_without_gf = system.Z2System2D_Config(lat2_without_gf, 1, 1, 1, 1, [0], ncopy=2)
        cfg_without_gf.paramvec = paramvec
        system_without_gf = system.Z2System2D(cfg_without_gf)
        system_without_gf.cfg.enforce_parameter_conditions(cfg_without_gf.paramvec)

        # Evaluation
        evaluator_with_gf = exacteval.ExactEvaluator(ExactEvaluatorConfig(), system_with_gf)
        evaluator_without_gf = exacteval.ExactEvaluator(ExactEvaluatorConfig(), system_without_gf)

        evaluator_with_gf.evaluate()
        evaluator_without_gf.evaluate()
        eval_with_gf = evaluator_with_gf.obsdict
        eval_without_gf = evaluator_without_gf.obsdict

        for key, val in eval_with_gf.items():
            self.assertTrue(np.allclose(val, eval_without_gf[key]))

    def test_exacteval_explosive_params_pinned_link(self):
        """Gauge fixing must reproduce the no-gauge-fixing electric energy even when the
        measured link is a pinned tree link and the parameters drive the modnorm/norm ratio
        huge.

        Regression for the incremental modified-norm/Woodbury tracker drift: these specific
        1-copy parameters make exp(norm_mod - lognorm_default) ~ 1e14 for some configs. With
        the measured link (0) pinned by gauge fixing, the incremental modified trackers used
        to drift catastrophically and give a large-negative (unphysical) electric energy
        (~ -483) while no gauge fixing gives ~ +16. The physical electric energy is bounded
        below by 0.
        """
        # Parameters (1 copy, 1 PG layer) known to expose the bug; the measured link is link 0,
        # which is on the maximal gauge-fixing tree (pinned to identity).
        explosive_params = np.array([0.0, 2.79433214e-04, -7.06773559e-01, 0.0, 9.99670799e-01, 7.06984522e-01])

        def run(gf_flag):
            lat2 = lattice.Lattice2D(2, 2, gf_flag)
            cfg = system.Z2System2D_Config(
                lat2,
                1.0,
                1.0,
                0.0,
                0.0,
                [],
                ncopy=1,
                num_pg_layer=1,
                num_fermionic_layer=0,
                mod_link_inds=(0,),
                unitcell_size=1,
                enforce_u1_symmetry=True,
            )
            cfg.paramvec = np.reshape(
                utils.translate_legacy_z2_parameter_order(cfg, explosive_params), cfg.param_shape()
            )
            cfg.make_pure_gauge()
            cfg.enforce_parameter_conditions(cfg.paramvec)
            sys_ = system.Z2System2D(cfg)
            ec = ExactEvaluatorConfig(tracker_refresh_interval=0)
            ec.compute_grads = False
            ev = exacteval.ExactEvaluator(ec, sys_)
            ev.evaluate()
            return float(ev.obsdict["el_energy"])

        el_no_gf = run(0)  # no gauge fixing (link 0 summed over)
        el_gf = run(-1)  # maximal-tree gauge fixing (link 0 pinned)

        # The physical electric energy is bounded below by 0 (offset convention).
        self.assertGreaterEqual(el_gf, -1e-6, msg=f"gauge-fixed electric energy is unphysical: {el_gf}")
        # Gauge fixing must reproduce the no-gauge-fixing value.
        self.assertAlmostEqual(el_gf, el_no_gf, places=4)

    def test_modified_norm_no_incremental_drift(self):
        """The modified open-link norm must equal a from-scratch recomputation after any
        sequence of gauge updates.

        The modified-norm quantity (norm_mod_vec) is a pure function of the current gauge
        configuration, so a system driven incrementally through a traversal must agree with a
        freshly built system set to the same configuration. A previous implementation ofs
        incrementalWoodbury/IncDet tracking of the modified objects violated this:
        when the measured link is pinned by gauge fixing it
        drifted catastrophically (norm_mod off by ~37 -> exp(37)
         in the modnorm/norm ratio).
        """
        explosive_params = np.array([0.0, 2.79433214e-04, -7.06773559e-01, 0.0, 9.99670799e-01, 7.06984522e-01])

        def build():
            lat2 = lattice.Lattice2D(2, 2, -1)  # maximal tree: the measured link 0 is pinned
            cfg = system.Z2System2D_Config(
                lat2,
                1.0,
                1.0,
                0.0,
                0.0,
                [],
                ncopy=1,
                num_pg_layer=1,
                num_fermionic_layer=0,
                mod_link_inds=(0,),
                unitcell_size=1,
                enforce_u1_symmetry=True,
            )
            cfg.paramvec = np.reshape(
                utils.translate_legacy_z2_parameter_order(cfg, explosive_params), cfg.param_shape()
            )
            cfg.make_pure_gauge()
            cfg.enforce_parameter_conditions(cfg.paramvec)
            return system.Z2System2D(cfg)

        persistent = build()  # driven incrementally through the whole traversal
        evaluator = exacteval.ExactEvaluator(ExactEvaluatorConfig(tracker_refresh_interval=0), persistent)

        max_drift = 0.0
        for config in evaluator.generate_config_vec():
            persistent.update_gauge_full_system(config, tracker_interval=0)
            fresh = build()
            fresh.update_gauge_full_system(config, tracker_interval=1)
            drift = np.max(
                np.abs(np.asarray(persistent.norm_mod_vec, dtype=float) - np.asarray(fresh.norm_mod_vec, dtype=float))
            )
            max_drift = max(max_drift, float(drift))

        self.assertLess(max_drift, 1e-8, msg=f"modified norm drifted by {max_drift} along the traversal")

    @staticmethod
    def _build_explosive_pure_gauge_system(gf_flag=-1):
        """Build the 1-copy, 1-PG-layer Z2 system with the explosive parameters and the measured
        link (0) pinned by maximal-tree gauge fixing. Shared by the tracker-refresh tests."""
        explosive_params = np.array([0.0, 2.79433214e-04, -7.06773559e-01, 0.0, 9.99670799e-01, 7.06984522e-01])
        lat2 = lattice.Lattice2D(2, 2, gf_flag)
        cfg = system.Z2System2D_Config(
            lat2,
            1.0,
            1.0,
            0.0,
            0.0,
            [],
            ncopy=1,
            num_pg_layer=1,
            num_fermionic_layer=0,
            mod_link_inds=(0,),
            unitcell_size=1,
            enforce_u1_symmetry=True,
        )
        cfg.paramvec = np.reshape(utils.translate_legacy_z2_parameter_order(cfg, explosive_params), cfg.param_shape())
        cfg.make_pure_gauge()
        cfg.enforce_parameter_conditions(cfg.paramvec)
        return system.Z2System2D(cfg)

    def test_mod_trackers_incremental_match_fresh_with_refresh(self):
        """With incremental modified-tracker tracking (plus periodic + magnitude-guard refresh),
        BOTH the modified covariance (covmat_out_mod_vec) and the modified norm (norm_mod_vec) must
        still equal a from-scratch system at every config along the gauge-fixed traversal.

        This is the stronger form of test_modified_norm_no_incremental_drift: it also pins the
        modified covariance matrix, not just its norm.
        """
        persistent = self._build_explosive_pure_gauge_system(-1)
        evaluator = exacteval.ExactEvaluator(ExactEvaluatorConfig(tracker_refresh_interval=0), persistent)

        max_norm_drift = 0.0
        max_cov_drift = 0.0
        for config in evaluator.generate_config_vec():
            persistent.update_gauge_full_system(config, tracker_interval=0)
            fresh = self._build_explosive_pure_gauge_system(-1)
            fresh.update_gauge_full_system(config, tracker_interval=1)

            max_norm_drift = max(
                max_norm_drift,
                float(
                    np.max(
                        np.abs(
                            np.asarray(persistent.norm_mod_vec, dtype=float)
                            - np.asarray(fresh.norm_mod_vec, dtype=float)
                        )
                    )
                ),
            )
            max_cov_drift = max(
                max_cov_drift,
                float(
                    np.max(np.abs(np.asarray(persistent.covmat_out_mod_vec) - np.asarray(fresh.covmat_out_mod_vec)))
                ),
            )

        self.assertLess(max_norm_drift, 1e-8, msg=f"modified norm drifted by {max_norm_drift}")
        self.assertLess(max_cov_drift, 1e-8, msg=f"modified covmat drifted by {max_cov_drift}")

    def test_closed_trackers_match_fresh_over_chain(self):
        """The closed (full-system) Woodbury inverses and incremental determinant, maintained
        incrementally with periodic + magnitude-guard refresh, must equal a from-scratch
        recomputation over the long incremental chain of a full traversal.
        """
        persistent = self._build_explosive_pure_gauge_system(-1)
        evaluator = exacteval.ExactEvaluator(ExactEvaluatorConfig(tracker_refresh_interval=0), persistent)

        max_incdet_drift = 0.0
        max_wi_in_drift = 0.0
        max_wi_out_drift = 0.0
        for config in evaluator.generate_config_vec():
            persistent.update_gauge_full_system(config, tracker_interval=0)
            # Ground truth: rebuild the closed trackers from scratch from the exact gamma_in_sys.
            wi_in_fresh, wi_out_fresh, incdet_fresh = persistent._compute_closed_trackers()

            max_incdet_drift = max(
                max_incdet_drift,
                float(np.max(np.abs(np.asarray(persistent.incdet_vec) - np.asarray(incdet_fresh)))),
            )
            max_wi_in_drift = max(
                max_wi_in_drift,
                float(np.max(np.abs(np.asarray(persistent.wi_gamma_in_vec) - np.asarray(wi_in_fresh)))),
            )
            max_wi_out_drift = max(
                max_wi_out_drift,
                float(np.max(np.abs(np.asarray(persistent.wi_gamma_out_vec) - np.asarray(wi_out_fresh)))),
            )

        self.assertLess(max_incdet_drift, 1e-8, msg=f"closed incdet drifted by {max_incdet_drift}")
        self.assertLess(max_wi_in_drift, 1e-8, msg=f"closed wi_gamma_in drifted by {max_wi_in_drift}")
        self.assertLess(max_wi_out_drift, 1e-8, msg=f"closed wi_gamma_out drifted by {max_wi_out_drift}")

    def test_magnitude_guard_catches_singular_step(self):
        """Even with the periodic refresh effectively disabled (huge interval), the magnitude guard
        must catch the near-singular modified update that the pinned measured link produces, keeping
        norm_mod from drifting.

        With a huge interval and NO guard the modified trackers are pure-incremental and drift
        catastrophically (norm_mod off by ~37) -- exactly the original bug. Passing this test with
        the periodic refresh disabled proves the *magnitude guard* (not the interval) is the safety
        net for the explosive case. (The magnitude of the tracked inverse is a global signal; a local
        capacitance-condition check misses this corner -- at the worst step cond~3e7 < 1e8 while the
        inverse magnitude is ~6e6.)
        """
        persistent = self._build_explosive_pure_gauge_system(-1)
        persistent.tracker_refresh_interval = 10**9  # disable periodic refresh; rely on the guard
        evaluator = exacteval.ExactEvaluator(ExactEvaluatorConfig(tracker_refresh_interval=0), persistent)

        max_drift = 0.0
        for config in evaluator.generate_config_vec():
            persistent.update_gauge_full_system(config, tracker_interval=0)
            fresh = self._build_explosive_pure_gauge_system(-1)
            fresh.update_gauge_full_system(config, tracker_interval=1)
            max_drift = max(
                max_drift,
                float(
                    np.max(
                        np.abs(
                            np.asarray(persistent.norm_mod_vec, dtype=float)
                            - np.asarray(fresh.norm_mod_vec, dtype=float)
                        )
                    )
                ),
            )

        self.assertLess(max_drift, 1e-8, msg=f"magnitude guard failed; modified norm drifted by {max_drift}")

    def test_evaluator_manager_sets_tracker_refresh_interval(self):
        """The tracker_refresh_interval passed to EvaluatorManager must reach the constructed system
        (stamped onto system_cfg, read via the system's getattr hook), so the refresh cadence is
        controllable from the manager / CLI as a single control point."""
        explosive_params = np.array([0.0, 2.79433214e-04, -7.06773559e-01, 0.0, 9.99670799e-01, 7.06984522e-01])
        lat2 = lattice.Lattice2D(2, 2, -1)
        cfg = system.Z2System2D_Config(
            lat2,
            1.0,
            1.0,
            0.0,
            0.0,
            [],
            ncopy=1,
            num_pg_layer=1,
            num_fermionic_layer=0,
            mod_link_inds=(0,),
            unitcell_size=1,
            enforce_u1_symmetry=True,
        )
        cfg.paramvec = np.reshape(utils.translate_legacy_z2_parameter_order(cfg, explosive_params), cfg.param_shape())
        cfg.make_pure_gauge()
        cfg.enforce_parameter_conditions(cfg.paramvec)

        mgr = EvaluatorManager(system.Z2System2D, cfg, ExactEvaluatorConfig(tracker_refresh_interval=7), 0)
        self.assertEqual(mgr.evaluator.cfg.tracker_refresh_interval, 7)

    # ----------------------------------------------- per-config gauge invariance (Z2 & D6)

    @staticmethod
    def _apply_gauge_transform(lat, config, V):
        """Standard pure gauge transformation U(x,k) -> V(x) U(x,k) V(x+k)^dagger.

        config: list of group-element matrices indexed by link.
        V: dict site (x, y) -> group-element matrix.
        Works for any matrix representation.
        """
        out = [None] * lat.nlinks
        for ind in range(lat.nlinks):
            coord, dr = lat.ind2coord_dir(ind)
            nb = lat.get_neighbor(coord, dr, orientation=True)
            out[ind] = V[coord] @ config[ind] @ V[nb].conj().T
        return out

    def _assert_per_config_gauge_invariance(self, make_system, seed, ntrials=4):
        """Assert |psi(G)|^2 and the electric/magnetic operators are invariant under a lattice
        gauge transformation U(x,k) -> V(x) U(x,k) V(x+k)^dagger, config by config.

        Args:
            make_system: zero-arg callable returning a fresh system (gauge field at identity).
            seed: RNG seed for the sampled gauge fields and transformations.
            ntrials: number of (G, V) pairs to test.
        """
        rng = np.random.RandomState(seed)
        probe = make_system()
        lat = probe.cfg.lattice
        vals = probe.cfg.gaugemgr.get_possible_gauge_values()

        def observables(config):
            s = make_system()
            s.update_gauge_full_system(config)
            norm = s.calculate_lognorm(incremental=False, all_factors=True)
            return (
                float(norm),
                float(np.real(s.el_energy_op)),
                float(np.real(s.mag_energy_op)),
            )

        for trial in range(ntrials):
            G = [vals[rng.randint(len(vals))].copy() for _ in range(lat.nlinks)]
            V = {(x, y): vals[rng.randint(len(vals))].copy() for y in range(lat.ny) for x in range(lat.nx)}
            GV = self._apply_gauge_transform(lat, G, V)
            ln_G, el_G, mag_G = observables(G)
            ln_V, el_V, mag_V = observables(GV)
            with self.subTest(trial=trial):
                self.assertAlmostEqual(ln_G, ln_V, places=6, msg="norm not gauge invariant")
                self.assertAlmostEqual(el_G, el_V, places=5, msg="electric operator not gauge invariant")
                self.assertAlmostEqual(mag_G, mag_V, places=6, msg="magnetic operator not gauge invariant")

    def test_z2_norm_and_observables_gauge_invariant(self):
        """Z2 (abelian): the per-config norm and electric/magnetic operators must be gauge
        invariant."""
        rng = np.random.RandomState(20260628)
        lat = lattice.Lattice2D(2, 2)
        paramvec = rng.rand(2, 20)
        cfg = system.Z2System2D_Config(lat, 1, 1, 1, 1, None, ncopy=2)
        cfg.paramvec = paramvec
        cfg.enforce_parameter_conditions(cfg.paramvec)

        self._assert_per_config_gauge_invariance(lambda: system.Z2System2D(cfg), seed=3)

    def test_d6_norm_and_observables_gauge_invariant(self):
        """D6 (non-abelian): the per-config norm and electric/magnetic operators must be gauge
        invariant.
        """
        rng = np.random.RandomState(20260628)
        num_pg_layer = 2
        lat = lattice.Lattice2D(2, 2)
        cfg = system.D6System2D_Config(lat, 1, 1, 0, 0, None, num_pg_layer=num_pg_layer, num_fermionic_layer=0)
        cfg.paramvec = rng.rand(num_pg_layer, 1, 20)
        cfg.enforce_parameter_conditions(cfg.paramvec)

        self._assert_per_config_gauge_invariance(lambda: system.D2nSystem2D(cfg), seed=2)

    @skip("Too long and not precise enough")
    def test_mceval(self):
        """Ensure that MC evaluation gives the same results with and without
        gauge fixing"""

        # MC config
        lat2_with_gf = lattice.Lattice2D(2, 2, -1)  # With gauge fixing
        lat2_without_gf = lattice.Lattice2D(2, 2)  # Without gauge fixing
        paramvec = np.random.rand(2, 20)

        # Configuration
        cfg_with_gf = system.Z2System2D_Config(lat2_with_gf, 1, 1, 1, 1, [0], ncopy=2)
        cfg_with_gf.paramvec = paramvec
        cfg_without_gf = system.Z2System2D_Config(lat2_without_gf, 1, 1, 1, 1, [0], ncopy=2)
        cfg_without_gf.paramvec = paramvec

        system_with_gf = system.Z2System2D(cfg_with_gf)
        system_without_gf = system.Z2System2D(cfg_without_gf)

        # MC evaluation
        mc_config = MonteCarloEvaluatorConfig(warmup_steps=20000, meas_steps=20000, binsize=1, update_size_per_step=2)

        mc_evaluator_with_gf = MonteCarloEvaluator(mc_config, system_with_gf)
        mc_config.gauge_fixing = True
        mc_evaluator_without_gf = MonteCarloEvaluator(mc_config, system_without_gf)
        mc_evaluator_without_gf.evaluate()
        mc_evaluator_with_gf.evaluate()
        no_gauge_fixing_energy = mc_evaluator_without_gf.get_obs_mean("energy")
        gauge_fixing_energy = mc_evaluator_with_gf.get_obs_mean("energy")

        self.assertAlmostEqual(no_gauge_fixing_energy, gauge_fixing_energy, places=0)
