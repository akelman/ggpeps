import unittest
import numpy as np
import jax.numpy as jnp

from ggpeps import utils
from ggpeps.system.backend import backend
from ggpeps.system.backend_numpy import BackendNumpy
from ggpeps.system.backend_jax import BackendJax


############## SELECT APPROPRIATE VERSION ##############


class TestBackends(unittest.TestCase):
    def setUp(self):
        self.jax_backend = BackendJax()
        self.numpy_backend = BackendNumpy()

    def test_pfaffians(self):
        """Test that the pfaffian function gives the same results for both backends."""

        # This mat gives an error in jax if nan's are not replaced by 0's (as now done in the jax backend)
        mat1_np = np.zeros((4, 4))
        mat1_jax = jnp.asarray(mat1_np)
        pfaval_np1 = self.numpy_backend.pfaffian(mat1_np)
        pfaval_jax1 = self.jax_backend.pfaffian(mat1_jax)
        self.assertAlmostEqual(pfaval_np1, np.asarray(pfaval_jax1))

        # This mat gives an error in jax if nan's are not replaced by 0's (as now done in the jax backend)
        mat2_np = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=float)
        mat2_jax = jnp.asarray(mat2_np)
        pfaval_np2 = self.numpy_backend.pfaffian(mat2_np)
        pfaval_jax2 = self.jax_backend.pfaffian(mat2_jax)
        self.assertAlmostEqual(pfaval_np2, np.asarray(pfaval_jax2))

        # This mat did not give an error, even before replacing nan's with 0's
        mat3_np = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
        mat3_jax = jnp.asarray(mat3_np)
        pfaval_np3 = self.numpy_backend.pfaffian(mat3_np)
        pfaval_jax3 = self.jax_backend.pfaffian(mat3_jax)
        self.assertAlmostEqual(pfaval_np3, np.asarray(pfaval_jax3))

        # An extra test with a random matrix
        mat4_np = np.random.rand(4, 4)
        mat4_np = utils.anti_symmetrize(mat4_np)
        mat4_jax = jnp.asarray(mat4_np)
        pfaval_np4 = self.numpy_backend.pfaffian(mat4_np)
        pfaval_jax4 = self.jax_backend.pfaffian(mat4_jax)
        self.assertAlmostEqual(pfaval_np4, np.asarray(pfaval_jax4))

    def test_vectorized_pfaffians(self):
        """Test that the vectorized pfaffian function gives the same results for as the non-vectorized version."""

        # Test with a batch of 3 random matrices
        mats = np.random.rand(3, 4, 4)
        mats = utils.anti_symmetrize(mats)

        pfavals_vectorized = np.asarray(backend.pfaffian_vectorized(mats))
        pfavals = np.array([backend.pfaffian(mat) for mat in mats])

        self.assertTrue(np.allclose(pfavals_vectorized, pfavals))
