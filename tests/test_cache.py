import unittest
import numpy as np

from ggpeps.minimizer import Cache


class TestCache(unittest.TestCase):

    def setUp(self):
        self.cache = Cache("eval-mc")

    def test_keygen(self):
        paramvec = np.array(
            [1.2, 2, 3]
        )  # for some reason this does not work if all params are int's
        key = self.cache.paramvec2key(paramvec)
        restored_paramvec = self.cache.key2paramvec(key)
        np.testing.assert_allclose(paramvec, restored_paramvec)

    def test_cache_keys(self):
        # Test that all cached variables are present
        self.assertEqual(self.cache.cache_data["cache_version"], 0.1)
        self.assertTrue("git_hash" in self.cache.cache_data.keys())

        self.assertEqual(self.cache.cache_data["mode"], "eval-mc")
        self.assertEqual(self.cache.cache_data["evaluator_manager"], None)
        self.assertEqual(self.cache.cache_data["energy"], {})
        self.assertEqual(self.cache.cache_data["energy_grad"], {})
