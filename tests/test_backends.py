import unittest
import numpy as np
import jax
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

    def test_dynamic_update_slice(self):
        """dynamic_update_slice writes a block at the given per-axis start indices with lax
        semantics (negative inds wrap, then inds clamp so the block fits). Both backends
        must agree."""
        rng = np.random.RandomState(3)
        cases = [
            (0, 1, 2, 2),  # interior
            (0, 2, -4, -4),  # negative: wraps to n - 4
            (0, 0, 5, 5),  # above range: clamps to n - size
        ]
        for starts in cases:
            mat = rng.rand(2, 3, 6, 6)
            val = rng.rand(2, 1, 3, 3)

            expected = mat.copy()
            index = []
            for ax, start in enumerate(starts):
                if start < 0:
                    start = start + mat.shape[ax]
                start = int(np.clip(start, 0, mat.shape[ax] - val.shape[ax]))
                index.append(slice(start, start + val.shape[ax]))
            expected[tuple(index)] = val

            res_np = np.asarray(self.numpy_backend.dynamic_update_slice(mat.copy(), starts, val))
            res_jax = np.asarray(self.jax_backend.dynamic_update_slice(jnp.asarray(mat), starts, jnp.asarray(val)))
            with self.subTest(starts=starts):
                self.assertTrue(np.array_equal(res_np, expected))
                self.assertTrue(np.array_equal(res_jax, expected))

    def test_dynamic_update_slice_jit_compiles_once(self):
        """dynamic_update_slice inds may be traced: a jitted caller must not recompile per value."""
        mat = jnp.zeros((2, 3, 6, 6))
        val = jnp.ones((2, 1, 3, 3))
        jitted = jax.jit(self.jax_backend.dynamic_update_slice)
        jitted(mat, (0, 1, 2, 2), val)
        jitted(mat, (0, 2, 0, 0), val)
        self.assertEqual(jitted._cache_size(), 1)

    def test_vectorized_pfaffians(self):
        """Test that the vectorized pfaffian function gives the same results for as the non-vectorized version."""

        # Test with a batch of 3 random matrices
        mats = np.random.rand(3, 4, 4)
        mats = utils.anti_symmetrize(mats)

        pfavals_vectorized = np.asarray(backend.pfaffian_vectorized(mats))
        pfavals = np.array([backend.pfaffian(mat) for mat in mats])

        self.assertTrue(np.allclose(pfavals_vectorized, pfavals))
