import unittest
import numpy as np

from ggpeps import gauge


class TestGauge(unittest.TestCase):

    def setUp(self):
        self.gaugeZ3 = gauge.ZNGauge(3)
        self.gaugeZ8 = gauge.ZNGauge(8)

    def test_possble_gauges(self):
        poss_gauges_z3 = self.gaugeZ3.get_possible_gauge_values()
        poss_gauges_z8 = self.gaugeZ8.get_possible_gauge_values()
        self.assertEqual(len(poss_gauges_z3), 3)
        self.assertEqual(len(poss_gauges_z8), 8)

    def test_random_values(self):
        seed = np.random.randint(np.iinfo(np.int32).max)
        rng_state = np.random.RandomState(seed)
        for gauge in [self.gaugeZ3, self.gaugeZ8]:
            poss_gauges = gauge.get_possible_gauge_values()
            for _ in range(100):
                val = gauge.get_random_gauge_value(rng_state)
                self.assertTrue(np.any(np.isclose(val, poss_gauges)))
