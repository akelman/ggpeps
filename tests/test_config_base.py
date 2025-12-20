import unittest
import numpy as np

import ggpeps.system.config_base as config_base


# ==================== ZN Gauged Projector Terms: Test ====================


class TestZNGaugedProjectorTermsForZ2(unittest.TestCase):
    def _assert_complex_close(self, a, b, rtol=1e-12, atol=1e-12):
        self.assertTrue(np.isclose(a, b, rtol=rtol, atol=atol), f"Complex values differ: {a} vs {b}")

    def _assert_terms_equal(self, got_indecies, got_constant, exp_indecies, exp_constant):
        # constant
        self._assert_complex_close(got_constant, exp_constant)
        # indecies structure and values
        self.assertEqual(len(got_indecies), len(exp_indecies), "Different number of monomials")
        for (g_coef, g_mon), (e_coef, e_mon) in zip(got_indecies, exp_indecies):
            self.assertEqual(g_mon, e_mon, f"Monomial indices differ: {g_mon} vs {e_mon}")
            self._assert_complex_close(g_coef, e_coef)

    def test_generate_gauged_projector_terms_small_cases(self):
        # Expected outputs for ncopy in {1,2}, layer in {pure_gauge, physical},
        # orientation in {horizontal, vertical}, group_order = 2.
        expected = {
            (1, "pure_gauge", "horizontal"): (
                (
                    (0.25, (0, 1)),
                    (-0.25j, (2, 0)),
                    (0.25, (2, 3)),
                    (0.25j, (3, 1)),
                ),
                0.0,
            ),
            (1, "pure_gauge", "vertical"): (
                (
                    (0.25, (0, 1)),
                    (0.25j, (2, 1)),
                    (0.25, (2, 3)),
                    (0.25j, (3, 0)),
                ),
                0.0,
            ),
            (1, "physical", "horizontal"): (
                (
                    (0.25, (0, 1)),
                    (-0.25j, (2, 0)),
                    (0.25, (2, 3)),
                    (0.25j, (3, 1)),
                ),
                0.0,
            ),
            (1, "physical", "vertical"): (
                (
                    (0.25, (0, 1)),
                    (0.25j, (2, 1)),
                    (0.25, (2, 3)),
                    (0.25j, (3, 0)),
                ),
                0.0,
            ),
            (2, "pure_gauge", "horizontal"): (
                (
                    (0.0625, (2, 3, 0, 1)),
                    (-0.0625j, (2, 3, 6, 0)),
                    (0.0625, (2, 3, 6, 7)),
                    (0.0625j, (2, 3, 7, 1)),
                    (-0.0625j, (2, 4, 0, 1)),
                    (-0.0625, (2, 4, 6, 0)),
                    (-0.0625j, (2, 4, 6, 7)),
                    (0.0625, (2, 4, 7, 1)),
                    (0.0625j, (3, 5, 0, 1)),
                    (0.0625, (3, 5, 6, 0)),
                    (0.0625j, (3, 5, 6, 7)),
                    (-0.0625, (3, 5, 7, 1)),
                    (0.0625, (4, 5, 0, 1)),
                    (-0.0625j, (4, 5, 6, 0)),
                    (0.0625, (4, 5, 6, 7)),
                    (0.0625j, (4, 5, 7, 1)),
                ),
                0.0,
            ),
            (2, "pure_gauge", "vertical"): (
                (
                    (0.0625, (2, 3, 0, 1)),
                    (0.0625j, (2, 3, 6, 1)),
                    (0.0625, (2, 3, 6, 7)),
                    (0.0625j, (2, 3, 7, 0)),
                    (0.0625j, (2, 5, 0, 1)),
                    (-0.0625, (2, 5, 6, 1)),
                    (0.0625j, (2, 5, 6, 7)),
                    (-0.0625, (2, 5, 7, 0)),
                    (0.0625j, (3, 4, 0, 1)),
                    (-0.0625, (3, 4, 6, 1)),
                    (0.0625j, (3, 4, 6, 7)),
                    (-0.0625, (3, 4, 7, 0)),
                    (0.0625, (4, 5, 0, 1)),
                    (0.0625j, (4, 5, 6, 1)),
                    (0.0625, (4, 5, 6, 7)),
                    (0.0625j, (4, 5, 7, 0)),
                ),
                0.0,
            ),
            (2, "physical", "horizontal"): (
                (
                    (0.0625, (0, 1, 4, 5)),
                    (-0.0625j, (0, 1, 6, 4)),
                    (0.0625, (0, 1, 6, 7)),
                    (0.0625j, (0, 1, 7, 5)),
                    (-0.0625j, (2, 0, 4, 5)),
                    (-0.0625, (2, 0, 6, 4)),
                    (-0.0625j, (2, 0, 6, 7)),
                    (0.0625, (2, 0, 7, 5)),
                    (0.0625, (2, 3, 4, 5)),
                    (-0.0625j, (2, 3, 6, 4)),
                    (0.0625, (2, 3, 6, 7)),
                    (0.0625j, (2, 3, 7, 5)),
                    (0.0625j, (3, 1, 4, 5)),
                    (0.0625, (3, 1, 6, 4)),
                    (0.0625j, (3, 1, 6, 7)),
                    (-0.0625, (3, 1, 7, 5)),
                ),
                0.0,
            ),
            (2, "physical", "vertical"): (
                (
                    (0.0625, (0, 1, 4, 5)),
                    (0.0625j, (0, 1, 6, 5)),
                    (0.0625, (0, 1, 6, 7)),
                    (0.0625j, (0, 1, 7, 4)),
                    (0.0625j, (2, 1, 4, 5)),
                    (-0.0625, (2, 1, 6, 5)),
                    (0.0625j, (2, 1, 6, 7)),
                    (-0.0625, (2, 1, 7, 4)),
                    (0.0625, (2, 3, 4, 5)),
                    (0.0625j, (2, 3, 6, 5)),
                    (0.0625, (2, 3, 6, 7)),
                    (0.0625j, (2, 3, 7, 4)),
                    (0.0625j, (3, 0, 4, 5)),
                    (-0.0625, (3, 0, 6, 5)),
                    (0.0625j, (3, 0, 6, 7)),
                    (-0.0625, (3, 0, 7, 4)),
                ),
                0.0,
            ),
        }

        for ncopy in (1, 2):
            for layer in ("pure_gauge", "physical"):
                for orientation in ("horizontal", "vertical"):
                    got_ind, got_const = config_base.generate_gauged_projector_terms(
                        ncopy=ncopy,
                        layer=layer,
                        orientation=orientation,
                        group_order=2,
                    )
                    exp_ind, exp_const = expected[(ncopy, layer, orientation)]
                    self._assert_terms_equal(got_ind, got_const, exp_ind, exp_const)
