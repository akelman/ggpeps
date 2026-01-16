import unittest
from unittest import skip

import numpy as np
import pandas as pd
import jax.numpy as jnp
from pfapack import pfaffian as pf

from ggpeps import utils, lattice, system
from ggpeps import xnp as xnp

import py_pfaffian.jax


class TestUtils(unittest.TestCase):

    def test_couplings_from_foldername(self):
        """Test that couplings are correctly extracted from folder names."""

        dirname = "L_2_g_1.3_int_1.4_mass_1.5_chem_1.0_2.0_3.0"
        couplings = utils.get_couplings_from_foldername(dirname)
        self.assertEqual(couplings, "g_1.3_int_1.4_mass_1.5_chem_1.0_2.0_3.0")

    def test_generate_smat(self):
        N = 10
        smat = utils.generate_smat(N)
        m, n = smat.shape
        self.assertEqual(m, n)
        self.assertEqual(m, N)
        res = smat @ np.conjugate(np.transpose(smat))
        ref = 2.0 * np.eye(N)
        self.assertTrue(np.allclose(ref, res))

    def test_select_except(self):
        arr = np.array([1, 2, 3, 4])
        arr_ref = np.array([1, 3, 4])
        arr_exc = utils.select_except(arr, 1)
        self.assertTrue(np.allclose(arr_ref, arr_exc))

    def test_select_except_list(self):
        arr = [1, 2, 3, 4]
        arr_ref = [1, 3, 4]
        arr_exc = utils.select_except(arr, 1)
        self.assertTrue(np.allclose(arr_ref, arr_exc))

    def test_multiply_except(self):
        arr = np.array([1, 2, 3, 4])
        dest = utils.multiply_except(arr, 3)
        self.assertEqual(6, dest)

    def test_anti_symmetrize(self):
        mat = np.random.rand(10)
        mat_as = utils.anti_symmetrize(mat)
        self.assertTrue(utils.is_antisymmetric(mat_as))

    def test_derivative_pfaffian_zero(self):
        zero_mat = np.zeros((4, 4))
        self.assertEqual(utils.derivative_pfaffian(zero_mat, zero_mat), 0)

    def test_derivative_pfaffian(self):
        matvec = [
            np.array(
                [
                    [0.0, 0.03656259, 0.27166934, -0.30600668],
                    [-0.03656259, 0.0, -0.04027417, 0.39463847],
                    [-0.27166934, 0.04027417, 0.0, -0.15850552],
                    [0.30600668, -0.39463847, 0.15850552, 0.0],
                ]
            ),
            np.array(
                [
                    [0.0, -0.03656259, -0.27166934, -0.30600668],
                    [0.03656259, 0.0, -0.04027417, 0.39463847],
                    [0.27166934, 0.04027417, 0.0, -0.15850552],
                    [0.30600668, -0.39463847, 0.15850552, 0.0],
                ]
            ),
        ]
        eps = 1e-6
        deriv_mat = np.zeros((4, 4))
        deriv_mat[0, 1] = 1
        deriv_mat[1, 0] = -1
        for mat in matvec:
            derivative_ana = utils.derivative_pfaffian(mat, deriv_mat)
            mat_rand_right = mat.copy()
            mat_rand_right[0, 1] += eps
            mat_rand_right[1, 0] -= eps
            mat_rand_left = mat.copy()
            mat_rand_left[0, 1] -= eps
            mat_rand_left[1, 0] += eps
            derivative_numeric = (pf.pfaffian(mat_rand_right) - pf.pfaffian(mat_rand_left)) / (2 * eps)
            self.assertAlmostEqual(derivative_numeric, derivative_ana)

    def test_derivative_pfaffian_rnd(self):
        eps = 1e-6
        deriv_mat = np.zeros((4, 4))
        deriv_mat[0, 1] = 1
        deriv_mat[1, 0] = -1
        for i in range(10):
            mat_rand = utils.anti_symmetrize(np.random.rand(4, 4))
            mat_rand_right = mat_rand.copy()
            mat_rand_right[0, 1] += eps
            mat_rand_right[1, 0] -= eps
            mat_rand_left = mat_rand.copy()
            mat_rand_left[0, 1] -= eps
            mat_rand_left[1, 0] += eps

            mat_rand = xnp.array(mat_rand)
            mat_rand_right = xnp.array(mat_rand_right)
            mat_rand_left = xnp.array(mat_rand_left)

            derivative_ana = utils.derivative_pfaffian(mat_rand, deriv_mat)
            derivative_numeric = (pf.pfaffian(mat_rand_right) - pf.pfaffian(mat_rand_left)) / (2 * eps)
            self.assertAlmostEqual(derivative_numeric, derivative_ana)

    def test_derivative_pfaffian_vectorized(self):
        # hard code a simple derivative matrix
        deriv_mat = np.zeros((4, 4))
        deriv_mat[0, 0] = 0.5
        deriv_mat[0, 1] = 1
        deriv_mat[1, 0] = -1
        deriv_mat[1, 1] = 1.3

        # create a stack of random antisymmetric matrices
        mat_list = []
        for i in range(5):
            mat_rand = utils.anti_symmetrize(np.random.rand(4, 4))
            mat_list.append(mat_rand)

        mat_stack = xnp.array(xnp.stack(mat_list, axis=0))
        pfavals = xnp.array([pf.pfaffian(mat) for mat in mat_list])

        # compute the derivates of the pfaffians
        derivative_ana_vec = utils.derivative_pfaffian_vectorized(mat_stack, deriv_mat, pfavals=pfavals)

        # compare to unvectorized calculation
        for i in range(5):
            deriv = utils.derivative_pfaffian(mat_stack[i], deriv_mat, pfaval=pfavals[i])
            self.assertAlmostEqual(deriv, derivative_ana_vec[i])

        # make sure the vectorized version also works on a single matrix
        derivative_ana_single = utils.derivative_pfaffian_vectorized(mat_stack[0], deriv_mat, pfavals=pfavals[0])
        self.assertAlmostEqual(derivative_ana_single, derivative_ana_vec[0])

    @staticmethod
    def _make_summary_df() -> pd.DataFrame:
        """Create a small 'summary-like' DataFrame that contains both scalars and ndarray objects in cells."""
        return pd.DataFrame(
            {
                "name": ["obs0", "obs1", "obs2"],
                "mean": [1.0, 2.0, 3.0],
                "err": [0.1, 0.2, 0.3],
                # Store ndarrays in object-dtype cells (this is what deepcopy_summary_df is meant to handle)
                "paramvec": [
                    np.array([1.0, 2.0, 3.0]),
                    np.array([], dtype=float),
                    np.array([[1, 2], [3, 4]]),
                ],
                "meta": ["a", "b", "c"],
            },
            index=["row0", "row1", "row2"],
        )

    def test_df_copy_deep_true_does_not_deepcopy_ndarray_cells(self):
        """pandas.DataFrame.copy(deep=True) does not deepcopy Python objects stored in cells (e.g. ndarrays)."""
        df = self._make_summary_df()

        df_copy = df.copy(deep=True)

        # The ndarray objects are the *same* objects (shared references)
        self.assertIs(df.at["row0", "paramvec"], df_copy.at["row0", "paramvec"])
        self.assertIs(df.at["row1", "paramvec"], df_copy.at["row1", "paramvec"])
        self.assertIs(df.at["row2", "paramvec"], df_copy.at["row2", "paramvec"])

        # Mutating the array inside the copy mutates the original as well (evidence of shared reference)
        df_copy.at["row0", "paramvec"][0] = 999.0
        self.assertEqual(df.at["row0", "paramvec"][0], 999.0)

    def test_deepcopy_summary_df_deepcopies_ndarray_cells(self):
        """utils.deepcopy_summary_df should make the DataFrame independent, including ndarray-valued cells."""
        df = self._make_summary_df()

        df_deep = utils.deepcopy_summary_df(df)

        # Scalars are equal
        self.assertTrue(df[["mean", "err"]].equals(df_deep[["mean", "err"]]))
        self.assertTrue(df[["name", "meta"]].equals(df_deep[["name", "meta"]]))

        # ndarray objects are *not* the same objects anymore
        self.assertIsNot(df.at["row0", "paramvec"], df_deep.at["row0", "paramvec"])
        self.assertIsNot(df.at["row1", "paramvec"], df_deep.at["row1", "paramvec"])
        self.assertIsNot(df.at["row2", "paramvec"], df_deep.at["row2", "paramvec"])

        # Contents are identical
        self.assertTrue(np.array_equal(df.at["row0", "paramvec"], df_deep.at["row0", "paramvec"]))
        self.assertTrue(np.array_equal(df.at["row1", "paramvec"], df_deep.at["row1", "paramvec"]))
        self.assertTrue(np.array_equal(df.at["row2", "paramvec"], df_deep.at["row2", "paramvec"]))

        # Mutating the deep-copied array does NOT affect the original
        df_deep.at["row0", "paramvec"][0] = 123.0
        self.assertNotEqual(df.at["row0", "paramvec"][0], 123.0)
        self.assertEqual(df_deep.at["row0", "paramvec"][0], 123.0)

    def test_trace_of_product(self):

        mat1 = np.random.rand(4, 4)
        mat2 = np.random.rand(4, 4)
        mat3 = np.random.rand(4, 4)
        mats = (mat1, mat2, mat3)

        trace_hadamard = utils.trace_of_product(mats, method="hadamard")
        trace_trace = utils.trace_of_product(mats, method="trace")

        # Manually compute the product and its trace
        prod = mat1 @ mat2 @ mat3
        trace_ref = np.trace(prod)

        self.assertEqual(trace_hadamard, trace_ref)
        self.assertEqual(trace_trace, trace_ref)


class TestBGBTransform(unittest.TestCase):
    def setUp(self):
        pass

    def test_cmp_dirac_pure_gauge(self):
        lat = lattice.Lattice2D(2, 2)
        system_u1_cfg = system.U1System2DConfig(lat, 1, 0, 0, 0.0, None)
        system_u1_cfg.paramvec = np.asarray([[0.0, 1.0, 2.0]])
        system_u1 = system.U1System2D(system_u1_cfg)
        lay = 0
        site = 0
        tmat_double = system_u1.tmat_layervec_sitevec[lay][site]
        gamma_dirac = utils.tmat_to_covariance_matrix(tmat_double)
        # Delete the rows and columns belonging to the physical fermions
        gamma_dirac = np.delete(gamma_dirac, [9], axis=1)
        gamma_dirac = np.delete(gamma_dirac, [9], axis=0)
        gamma_dirac = np.delete(gamma_dirac, [0], axis=1)
        gamma_dirac = np.delete(gamma_dirac, [0], axis=0)

        tmat_single = system_u1.eval_tmat_symb_single(system_u1.cfg.paramvec[0][0])
        # Cut the physical mode
        tmat_single = tmat_single[1:, :]
        bgb_trafo = utils.BgbTransform(tmat_single, pure_gauge=True)
        gamma_dirac_svd = bgb_trafo.mat_out

        self.assertTrue(np.allclose(np.real(gamma_dirac), np.real(gamma_dirac_svd)))
        self.assertTrue(np.allclose(np.imag(gamma_dirac), np.imag(gamma_dirac_svd)))

    @skip("The case with fermions is not implemented properly yet.")
    def test_cmp_dirac(self):
        lat = lattice.Lattice2D(2, 2)
        system_u1_cfg = system.U1System2DConfig(lat, 1, 0, 0)
        system_u1_cfg.paramvec = np.asarray([[0.7, 1.0, 2.0]])
        system_u1 = system.U1System2D(system_u1_cfg)
        lay = 0
        site = 0
        tmat_double = system_u1.tmat_layervec_sitevec[lay][site]
        # We use the function explicitly to avoid the permutation matrix
        gamma_dirac = utils.tmat_to_covariance_matrix(tmat_double)
        # gamma_dirac = system_u1.gamma_dirac_vec[0]

        tmat_single = system_u1.eval_tmat_symb_single(system_u1.cfg.paramvec[0][0])
        bgb_trafo = utils.BgbTransform(tmat_single, pure_gauge=False)
        gamma_dirac_svd = bgb_trafo.mat_out

        # utils.show_matrixvec([np.real(gamma_dirac_svd), np.real(gamma_dirac)], title=["Re SVD", "Re T"])
        # utils.show_matrixvec([np.imag(gamma_dirac_svd), np.imag(gamma_dirac)], title=["Im SVD", "Im T"])

        self.assertTrue(np.allclose(np.real(gamma_dirac), np.real(gamma_dirac_svd)))
        self.assertTrue(np.allclose(np.imag(gamma_dirac), np.imag(gamma_dirac_svd)))

    def test_simple_mat(self):
        mat = np.array([[0, 1, 2, 3], [7, 0, 2, 1], [9, 6, 0, 9], [8, 4, 0, 0]])
        mat_zero = np.zeros((4, 4))
        mat_full = np.block([[mat_zero, mat], [-np.transpose(mat), mat_zero]])
        # We have different input conventions for the two functions
        # BgbTransform takes only the single matrix and doubles it for positive and negative modes
        # utils.tmat_to_covariance_matrix takes the full T matrix
        covmat_direct = utils.tmat_to_covariance_matrix(mat_full)
        covmat_bgb = utils.BgbTransform(mat, pure_gauge=True).mat_out
        self.assertTrue(np.allclose(covmat_direct, covmat_bgb))


class TestPfaffian(unittest.TestCase):

    def setUp(self):
        N = 4
        self.mat = np.zeros((N, N), dtype=complex)
        self.mat[0, 1] = 1.0
        self.mat[1, 0] = -1.0
        self.mat[0, 2] = 2.0
        self.mat[2, 0] = -2.0
        self.mat[0, 3] = 3.0
        self.mat[3, 0] = -3.0
        self.mat[1, 2] = 4.0j
        self.mat[2, 1] = -4.0j
        self.mat[1, 3] = 5.0
        self.mat[3, 1] = -5.0
        self.mat[2, 3] = 6.0
        self.mat[3, 2] = -6.0

    def test_pfaffian(self):
        pfaffian = pf.pfaffian(self.mat)
        self.assertAlmostEqual(np.real(pfaffian), -4)
        self.assertAlmostEqual(np.imag(pfaffian), 12)

    def test_pfaffian_explicit(self):
        val = utils.pfaffian_explicit_4x4(self.mat)
        ref = pf.pfaffian(self.mat)
        self.assertAlmostEqual(val, ref)

    def test_pfaffian_explicit_masked(self):
        a = np.random.rand(16, 16)
        a = a - a.T
        indarr = np.array([0, 7, 8, 10])
        val = utils.pfaffian_explicit_4x4_masked(a, indarr)
        a_part = a[np.ix_(indarr, indarr)]
        ref = pf.pfaffian(a_part)
        self.assertAlmostEqual(val, ref)

    def test_pfaffian_LTL_jax_vs_numpy(self):
        """Test pfaffian jax against numpy version for random skew-symmetric matrices."""
        np.random.seed(42)
        for n in [4, 6, 8]:
            # Real case
            mat = np.random.randn(n, n)
            mat = mat - mat.T  # Make skew-symmetric
            pf_np = pf.pfaffian_LTL(mat)
            pf_jax = py_pfaffian.jax.pfaffian(jnp.array(mat))
            # print(f"n={n} real: numpy={pf_np}, jax={pf_jax}")
            self.assertTrue(np.allclose(pf_np, float(pf_jax), rtol=1e-6, atol=1e-8))

            # Complex case
            mat = np.random.randn(n, n) + 1j * np.random.randn(n, n)
            mat = mat - mat.T
            pf_np = pf.pfaffian_LTL(mat)
            pf_jax = py_pfaffian.jax.pfaffian(jnp.array(mat))
            # print(f"n={n} complex: numpy={pf_np}, jax={pf_jax}")
            self.assertTrue(np.allclose(pf_np, pf_jax, rtol=1e-6, atol=1e-8))
