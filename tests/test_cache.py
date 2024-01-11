import unittest
import numpy as np

from ggpeps.minimizer import Cache

class TestCache(unittest.TestCase):

    def setUp(self):
        self.cache = Cache()

    def test_keygen(self):
        paramvec = np.array([1.2, 2, 3]) # for some reason this does not work if all params are int's
        key = self.cache.paramvec2key(paramvec)
        restored_paramvec = self.cache.key2paramvec(key)
        np.testing.assert_allclose(paramvec, restored_paramvec)