import unittest
import numpy as np

from ggpeps import gauge


class TestGauge(unittest.TestCase):

    def setUp(self):
        self.gaugeZ3 = gauge.ZNGauge(3)
        self.gaugeZ8 = gauge.ZNGauge(8)
        self.gaugeD6 = gauge.D2nGauge(3)
        self.gaugeD8 = gauge.D2nGauge(4)

    def test_possible_gauges(self):
        poss_gauges_z3 = self.gaugeZ3.get_possible_gauge_values()
        poss_gauges_z8 = self.gaugeZ8.get_possible_gauge_values()
        poss_gauges_d6 = self.gaugeD6.get_possible_gauge_values()
        poss_gauges_d8 = self.gaugeD8.get_possible_gauge_values()
        self.assertEqual(len(poss_gauges_z3), 3)
        self.assertEqual(len(poss_gauges_z8), 8)
        self.assertEqual(len(poss_gauges_d6), 6)
        self.assertEqual(len(poss_gauges_d8), 8)

    def test_random_values(self):
        seed = np.random.randint(np.iinfo(np.int32).max)
        rng_state = np.random.RandomState(seed)
        for gauge_obj in [self.gaugeZ3, self.gaugeZ8, self.gaugeD6, self.gaugeD8]:
            poss_gauges = gauge_obj.get_possible_gauge_values()
            for _ in range(100):
                val = gauge_obj.get_random_gauge_value(rng_state)
                self.assertTrue(np.any(np.isclose(val, poss_gauges)))

    def test_get_group_elements_and_factors_for_el_energy(self):
        factors = self.gaugeD6.el_mult_factor
        self.assertTrue(np.allclose(factors, 1.0), f"Expected all electric factors to be 1.0, but got: {factors}")

        actual_elements = self.gaugeD6.group_elements_for_el_energy
        expected_elements = [self.gaugeD6.get_representation(p, 1) for p in range(self.gaugeD6.n)]

        # Check lengths match first
        self.assertEqual(
            len(actual_elements),
            len(expected_elements),
            f"Count mismatch: Expected {len(expected_elements)} elements, got {len(actual_elements)}",
        )

        # Check that every actual element exists in the expected list
        for i, act in enumerate(actual_elements):
            found = any(np.allclose(act, exp, atol=1e-8) for exp in expected_elements)
            self.assertTrue(found, f"Element at index {i} was not found in expected representations:\n{act}")
