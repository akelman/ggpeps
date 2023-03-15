import unittest
from unittest import skip
from ggpeps import utils, lattice, system
from pfapack import pfaffian as pf
import numpy as np

class TestUtils(unittest.TestCase):

    def test_generate_smat(self):
        N=10
        smat = utils.generate_smat(N)
        m, n = smat.shape
        self.assertEqual(m, n)
        self.assertEqual(m, N)
        res=smat@np.conjugate(np.transpose(smat))
        ref=2.*np.eye(N)
        self.assertTrue(np.allclose(ref,res))

    def test_select_except(self):
        arr=np.array([1,2,3,4])
        arr_ref=np.array([1,3,4])
        arr_exc=utils.select_except(arr,1)
        self.assertTrue(np.allclose(arr_ref,arr_exc))

    def test_select_except_list(self):
        arr=[1,2,3,4]
        arr_ref=[1,3,4]
        arr_exc=utils.select_except(arr,1)
        self.assertTrue(np.allclose(arr_ref,arr_exc))

    def test_multiply_except(self):
        arr=np.array([1,2,3,4])
        dest=utils.multiply_except(arr,3)
        self.assertEqual(6,dest)
    
    def test_anti_symmetrize(self):
        mat = np.random.rand(10)
        mat_as = utils.anti_symmetrize(mat)
        self.assertTrue(utils.is_antisymmetric(mat_as))

    def test_derivative_pfaffian_zero(self):
        zero_mat = np.zeros((4,4))
        self.assertEqual(utils.derivative_pfaffian(zero_mat,zero_mat),0)

    def test_derivative_pfaffian(self):
        matvec = [np.array([[0.,         0.03656259, 0.27166934, -0.30600668],
                            [-0.03656259,  0., -0.04027417,  0.39463847],
                            [-0.27166934,  0.04027417,  0., -0.15850552],
                            [0.30600668, -0.39463847,  0.15850552,  0.]]),
                  np.array([[0.,         -0.03656259, -0.27166934, -0.30600668],
                            [0.03656259,  0., -0.04027417,  0.39463847],
                            [0.27166934,  0.04027417,  0., -0.15850552],
                            [0.30600668, -0.39463847,  0.15850552,  0.]])
                  ]
        eps=1e-6
        deriv_mat = np.zeros((4, 4))
        deriv_mat[0, 1] = 1
        deriv_mat[1, 0] = -1
        for mat in matvec:
            derivative_ana = utils.derivative_pfaffian(mat,deriv_mat)
            mat_rand_right = mat.copy()
            mat_rand_right[0, 1] += eps
            mat_rand_right[1, 0] -= eps
            mat_rand_left = mat.copy()
            mat_rand_left[0, 1] -= eps
            mat_rand_left[1, 0] += eps
            derivative_numeric = (pf.pfaffian(mat_rand_right)-pf.pfaffian(mat_rand_left))/(2*eps)
            self.assertAlmostEqual(derivative_numeric,derivative_ana)

    def test_derivative_pfaffian_rnd(self):
        eps=1e-6
        deriv_mat = np.zeros((4, 4))
        deriv_mat[0, 1] = 1
        deriv_mat[1, 0] = -1
        for i in range(10):
            mat_rand = utils.anti_symmetrize(np.random.rand(4,4))
            derivative_ana = utils.derivative_pfaffian(mat_rand,deriv_mat)
            mat_rand_right = mat_rand.copy()
            mat_rand_right[0, 1] += eps
            mat_rand_right[1, 0] -= eps
            mat_rand_left = mat_rand.copy()
            mat_rand_left[0, 1] -= eps
            mat_rand_left[1, 0] += eps
            derivative_numeric = (pf.pfaffian(mat_rand_right)-pf.pfaffian(mat_rand_left))/(2*eps)
            self.assertAlmostEqual(derivative_numeric,derivative_ana)

class TestBGBTransform(unittest.TestCase):
    def setUp(self):
        pass


    def test_cmp_dirac_pure_gauge(self):
        lat = lattice.Lattice2D(2,2)
        system_u1_cfg = system.U1System2DConfig(lat, 1, 0, 0)
        system_u1_cfg.paramvec = np.asarray([[0., 1., 2.]])
        system_u1 = system.U1System2D(system_u1_cfg)
        tmat_double = system_u1.tmat_vec[0]
        gamma_dirac = utils.tmat_to_covariance_matrix(tmat_double)
        #Delete the rows and columns belonging to the physical fermions
        gamma_dirac = np.delete(gamma_dirac,[9],axis=1)
        gamma_dirac = np.delete(gamma_dirac,[9],axis=0)
        gamma_dirac = np.delete(gamma_dirac,[0],axis=1)
        gamma_dirac = np.delete(gamma_dirac,[0],axis=0)

        tmat_single = system_u1._eval_tmat_symb_single(system_u1.cfg.paramvec[0])
        #Cut the physical mode
        tmat_single = tmat_single[1:,:]
        bgb_trafo = utils.BgbTransform(tmat_single, pure_gauge=True)
        gamma_dirac_svd = bgb_trafo.mat_out

        self.assertTrue(np.allclose(np.real(gamma_dirac), np.real(gamma_dirac_svd)))
        self.assertTrue(np.allclose(np.imag(gamma_dirac), np.imag(gamma_dirac_svd)))


    @skip("The case with fermions is not implemented properly yet.")
    def test_cmp_dirac(self):
        lat = lattice.Lattice2D(2,2)
        system_u1_cfg = system.U1System2DConfig(lat, 1, 0, 0)
        system_u1_cfg.paramvec = np.asarray([[0.7, 1., 2.]])
        system_u1 = system.U1System2D(system_u1_cfg)
        tmat_double = system_u1.tmat_vec[0]
        # We use the function explicitly to avoid the permutation matrix
        gamma_dirac = utils.tmat_to_covariance_matrix(tmat_double)
        #gamma_dirac = system_u1.gamma_dirac_vec[0]

        tmat_single = system_u1._eval_tmat_symb_single(system_u1.cfg.paramvec[0])
        bgb_trafo = utils.BgbTransform(tmat_single, pure_gauge=False)
        gamma_dirac_svd = bgb_trafo.mat_out

        #utils.show_matrixvec([np.real(gamma_dirac_svd), np.real(gamma_dirac)], title=["Re SVD", "Re T"])
        #utils.show_matrixvec([np.imag(gamma_dirac_svd), np.imag(gamma_dirac)], title=["Im SVD", "Im T"])

        self.assertTrue(np.allclose(np.real(gamma_dirac), np.real(gamma_dirac_svd)))
        self.assertTrue(np.allclose(np.imag(gamma_dirac), np.imag(gamma_dirac_svd)))


    def test_simple_mat(self):
        mat = np.array([[0, 1, 2, 3],
                        [7, 0, 2, 1],
                        [9, 6, 0, 9],
                        [8, 4, 0, 0]])
        mat_zero = np.zeros((4,4))
        mat_full = np.block([[mat_zero,mat],[-np.transpose(mat),mat_zero]])
        # We have different input conventions for the two functions
        # BgbTransform takes only the single matrix and doubles it for positive and negative modes
        # uitls.tmat_to_covariance_matrix takes the full T matrix
        covmat_direct = utils.tmat_to_covariance_matrix(mat_full)
        covmat_bgb = utils.BgbTransform(mat, pure_gauge=True).mat_out
        self.assertTrue(np.allclose(covmat_direct, covmat_bgb))

class TestPfaffian(unittest.TestCase):


    def setUp(self):
        N=4
        self.mat=np.zeros((N, N), dtype=complex)
        self.mat[0,1]=1.0
        self.mat[1,0]=-1.0
        self.mat[0,2]=2.0
        self.mat[2,0]=-2.0
        self.mat[0,3]=3.0
        self.mat[3,0]=-3.0
        self.mat[1,2]=4.0j
        self.mat[2,1]=-4.0j
        self.mat[1,3]=5.0
        self.mat[3,1]=-5.0
        self.mat[2,3]=6.0
        self.mat[3,2]=-6.0

    def test_pfaffian(self):
        pfaffian = pf.pfaffian(self.mat)
        self.assertAlmostEqual(np.real(pfaffian), -4)
        self.assertAlmostEqual(np.imag(pfaffian), 12)
    
    def test_pfaffian_explicit(self):
        val = utils.pfaffian_explicit_4x4(self.mat)
        ref = pf.pfaffian(self.mat)
        self.assertAlmostEqual(val, ref)

    def test_pfaffian_explicit_masked(self):
        a = np.random.rand(16,16)
        a = a-a.T
        indarr =  np.array([0,7,8,10])
        val = utils.pfaffian_explicit_4x4_masked(a,indarr)
        a_part = a[np.ix_(indarr,indarr)]
        ref = pf.pfaffian(a_part)
        self.assertAlmostEqual(val, ref)