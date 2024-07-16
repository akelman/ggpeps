import os
import importlib

import unittest 
from unittest import skip

import numpy as np
import jax.numpy as jnp

from ggpeps import utils
from ggpeps import lattice
from ggpeps import system, exacteval
from ggpeps.system.global_funcs import *

from tests import TEST_BACKEND

# ======================= Z2 fermionic system (4 copies) =========================================

class TestBackends(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        # Reload ggpeps to ensure that the backend is set correctly for the rest of the tests
        os.environ["GGPEPS_BACKEND"] = TEST_BACKEND
        importlib.reload(ggpeps)
    
    def test_reloading_backend(self):
        os.environ["GGPEPS_BACKEND"] = "numpy"
        importlib.reload(ggpeps)
        self.assertTrue(ggpeps.PREFERRED_BACKEND == "numpy")

        os.environ["GGPEPS_BACKEND"] = "jax"
        importlib.reload(ggpeps)
        self.assertTrue(ggpeps.PREFERRED_BACKEND == "jax")

    def test_slicing(self):
        """Ensure that numpy and jax slice arrays in the same way.
        """
        np_mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        jax_mat = jnp.array(np_mat)

        a,b,c,d = 1,2,1,2 # bounds
        np_slice = slice_matrix_numpy(np_mat, a, b, c, d)
        jax_slice = slice_matrix_jax(jax_mat, a, b, c, d)
        self.assertTrue(np.allclose(np_slice, np.array(jax_slice) ))


