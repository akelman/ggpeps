import unittest 
from unittest import skip
import numpy as np
from ggpeps import lattice
from ggpeps.lattice import Direction
from ggpeps import system
from ggpeps import utils
from ggpeps.mc import MonteCarloEvaluatorConfig, MonteCarloEvaluator
from ggpeps.utils import compare_array_elementwise

class TestU1SystemMethods(unittest.TestCase):

    def setUp(self):
        lat = lattice.Lattice2D(2, 2)
        paramvec = [[0.1, 0.3, 0.4]]
        cfg = system.U1System2DConfig(lat, 1.0, 1.0, 0.0)
        cfg.paramvec = paramvec
        self.system_u1_2_2 = system.U1System2D(cfg)

    def test_tmat_antisymmetric(self):
        tmat = self.system_u1_2_2.tmat_vec[0]
        m,n = tmat.shape
        self.assertEqual(m,9)
        self.assertEqual(n,9)
        self.assertTrue(utils.is_antisymmetric(np.real(tmat)))

    def test_gamma_dirac_covariance(self):
        gamma_dirac = self.system_u1_2_2.gamma_dirac_vec[0]
        m, n = gamma_dirac.shape
        res = gamma_dirac @ np.transpose(np.conjugate(gamma_dirac))
        ref = 0.25 * np.eye(gamma_dirac.shape[0])
        self.assertTrue(np.allclose(ref, res))
        self.assertEqual(m, n)
        self.assertTrue(utils.is_antisymmetric(gamma_dirac))

    def test_gamma_maj_covariance(self):
        gamma_maj = self.system_u1_2_2.gamma_maj_vec[0]
        m, n = gamma_maj.shape
        self.assertEqual(m, n)
        self.assertTrue(utils.is_antisymmetric(gamma_maj))
        self.assertTrue(np.allclose(gamma_maj @ gamma_maj, -np.eye(m)))
        self.assertTrue(np.allclose(gamma_maj@np.transpose(gamma_maj),np.eye(m)))

    def test_gamma_maj_sys_covariance(self):
        gamma_maj = self.system_u1_2_2.gamma_maj_sys_vec[0]
        m, n = gamma_maj.shape
        self.assertEqual(m, n)
        self.assertTrue(utils.is_antisymmetric(gamma_maj))
        self.assertTrue(np.allclose(gamma_maj @ gamma_maj, -np.eye(m)))
        self.assertTrue(np.allclose(gamma_maj @ np.transpose(gamma_maj), np.eye(m)))

    def test_gamma_in_sys_covariance(self):
        gamma_in = self.system_u1_2_2.gamma_in_sys
        m, n = gamma_in.shape
        self.assertEqual(m, n)
        self.assertTrue(utils.is_antisymmetric(gamma_in))
        self.assertTrue(np.allclose(gamma_in@gamma_in,-np.eye(m)))
        self.assertTrue(np.allclose(gamma_in@np.transpose(gamma_in),np.eye(m)))

    def test_norm_minimal(self):
        # This update is a nullop since we initialize the gauge-field with 0
        zeroarr = np.zeros((1, 1))
        #The factor of 2 compensates for the
        logdet_inc = 2*self.system_u1_2_2.update_lognorm_inc(0, zeroarr, all_factors=False)
        # This is equivalent to
        #logdet_inc = self.system_u1_2_2.incdet.det()
        diff = self.system_u1_2_2.mat_d_inv_vec[0] - self.system_u1_2_2.gamma_in_sys
        sign, logdet = np.linalg.slogdet(diff)
        self.assertGreater(sign,0)
        self.assertAlmostEqual(logdet_inc, logdet)

    def test_norm_incremental(self):
        # Test that the incremental update is equivalent to the re-calculation of the norm
        # This update is a nullop since we initialize the gauge-field with 0
        zeroarr = np.zeros((1, 1))
        weight_inc = self.system_u1_2_2.update_lognorm_inc(0, zeroarr, all_factors=True)
        weight_recalc = self.system_u1_2_2.calculate_lognorm(all_factors=True)
        self.assertAlmostEqual(weight_inc, weight_recalc)

    def test_norm_incremental_update(self):
        # Test that the incremental update is equivalent to the re-calculation of the norm
        ind = 0
        theta = np.pi
        weight_inc = self.system_u1_2_2.calculate_weight_attempt(ind, theta, all_factors=True)
        self.system_u1_2_2.update_gauge_ind(ind, theta)
        weight_recalc = self.system_u1_2_2.calculate_lognorm(all_factors=True)
        self.assertAlmostEqual(weight_inc, weight_recalc)


    def test_compare_gauge_gamma_dirac(self):
        lat = lattice.Lattice2D(2, 2)
        paramvec = [[0.1, 0.4, 0.2]]
        cfg = system.U1System2DConfig(lat, 1.0, 0.0, 1.0)
        cfg.paramvec = paramvec
        system_u1_2_2 = system.U1System2D(cfg)
        gamma_dirac = system_u1_2_2.gamma_dirac_vec[0]
        gamma_dirac_cpp = utils.load_matrix_dat_fmt("misc/gamma_dirac_cpp_t_0.1_y_0.4_z_0.2.dat")
        self.assertTrue(np.allclose(gamma_dirac,gamma_dirac_cpp))


    def test_compare_cpp_gamma_maj(self):
        lat = lattice.Lattice2D(2, 2)
        paramvec = [[0.1, 0.4, 0.2]]
        cfg = system.U1System2DConfig(lat, 1.0, 0.0, 1.0)
        cfg.paramvec = paramvec
        system_u1_2_2 = system.U1System2D(cfg)
        gamma_maj = system_u1_2_2.gamma_maj_vec[0]
        gamma_maj_cpp = utils.load_matrix_dat_fmt("misc/gamma_maj_cpp_t_0.1_y_0.4_z_0.2.dat",is_complex=False)
        self.assertTrue(np.allclose(gamma_maj,gamma_maj_cpp))


    def test_compare_cpp_pure_gauge_gamma_dirac(self):
        lat = lattice.Lattice2D(2, 2)
        paramvec = [[0.0, 0.4, 0.2]]
        cfg = system.U1System2DConfig(lat, 1.0, 0.0, 1.0)
        cfg.paramvec = paramvec
        system_u1_2_2 = system.U1System2D(cfg)
        gamma_dirac = system_u1_2_2.gamma_dirac_vec[0]
        gamma_dirac_cpp = utils.load_matrix_dat_fmt("misc/gamma_dirac_cpp_t_0.0_y_0.4_z_0.2.dat")
        self.assertTrue(np.allclose(gamma_dirac,gamma_dirac_cpp))


    def test_compare_cpp_pure_gauge_gamma_maj(self):
        lat = lattice.Lattice2D(2, 2)
        paramvec = [[0.0, 0.4, 0.2]]
        cfg = system.U1System2DConfig(lat, 1.0, 0.0, 1.0)
        cfg.paramvec = paramvec
        system_u1_2_2 = system.U1System2D(cfg)
        gamma_maj = system_u1_2_2.gamma_maj_vec[0]
        gamma_maj_cpp = utils.load_matrix_dat_fmt("misc/gamma_maj_cpp_t_0.0_y_0.4_z_0.2.dat",is_complex=False)
        self.assertTrue(np.allclose(gamma_maj,gamma_maj_cpp))

    @skip("This method cannot work correctly since the electric energy is not computed correctly. The pre-factor is different for the U1 case in contrast to the Z2 case.")
    def test_electric_energy_L_2_empty(self):
        # Calculate the electric energy of an empty system.
        paramvec = np.zeros((1,3))
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.U1System2DConfig(lat_2x2, 1.0, None, None)
        system_cfg.paramvec = paramvec
        system_u1_2_2_pf = system.U1System2D(system_cfg)
        system_u1_2_2_pf.use_pfaffian = True
        system_u1_2_2_link = system.U1System2D(system_cfg)
        el_energy_pf = system_u1_2_2_pf.el_energy
        el_energy_link = system_u1_2_2_link.el_energy
        self.assertAlmostEqual(el_energy_pf, 0.0)
        self.assertAlmostEqual(el_energy_link, 0.0)
        self.assertAlmostEqual(el_energy_link, el_energy_pf)

    @skip("This method cannot work correctly since the electric energy is not computed correctly. The pre-factor is different for the U1 case in contrast to the Z2 case.")
    def test_electric_energy_L_4_empty(self):
        paramvec = np.zeros((1,3))
        lat_4x4 = lattice.Lattice2D(4, 4)
        system_cfg = system.U1System2DConfig(lat_4x4, 1.0, None, None)
        system_cfg.paramvec = paramvec
        system_u1_pf = system.U1System2D(system_cfg)
        system_u1_pf.use_pfaffian = True
        system_u1_link = system.U1System2D(system_cfg)
        el_energy_pf = system_u1_pf.el_energy
        el_energy_link = system_u1_link.el_energy
        self.assertAlmostEqual(el_energy_pf, 0.0)
        self.assertAlmostEqual(el_energy_link, 0.0)
        self.assertAlmostEqual(el_energy_link, el_energy_pf)

    def test_electric_energy_L_2_ring_even_pfaffian(self):
        paramvec = np.zeros((1,3))
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.U1System2DConfig(lat_2x2, 1.0, None, None)
        system_cfg.paramvec = paramvec
        system_u1 = system.U1System2D(system_cfg)
        system_u1.use_pfaffian = True
        #Modify the matrix D by hand
        # [[0, -1],[1, 0]]
        filled=np.zeros((2,2))
        filled[0,1]=-1
        filled[1,0]=1
        gamma_maj_sys_vec=system_u1.gamma_maj_sys_vec
        gamma_maj_sys=np.copy(gamma_maj_sys_vec[0])
        offset=lat_2x2.size*2
        #Setting mode l(0,1)-
        gamma_maj_sys[offset + 2:offset+4,offset + 2:offset+4] = filled
        #Setting mode r(0,0)+
        gamma_maj_sys[offset + 4:offset+6, offset + 4:offset+6] = filled
        #Setting mode l(0,0)+
        gamma_maj_sys[offset + 8:offset+10, offset + 8:offset+10] = filled
        #Setting mode r(0,1)-
        gamma_maj_sys[offset + 14:offset+16, offset + 14:offset+16] = filled
        system_u1.gamma_maj_sys_vec[0]=gamma_maj_sys
        mat_d = gamma_maj_sys[offset:gamma_maj_sys.shape[0],
                              offset:gamma_maj_sys.shape[1]]
        system_u1.mat_d_inv_vec[0]=np.linalg.inv(mat_d)
        system_u1.det_mat_d_vec[0]=np.linalg.slogdet(mat_d)[1]
        # We are checking the value for a single link
        el_energy_link_bare, _ = system_u1._compute_el_energy_op_and_grad_pfaffian(True)
        el_energy = 2 - 2 * np.prod(el_energy_link_bare)
        self.assertAlmostEqual(el_energy, 3.)

    def test_electric_energy_L_2_ring_odd(self):
        paramvec = np.zeros((1,3))
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.U1System2DConfig(lat_2x2, 1.0, None, None)
        system_cfg.paramvec = paramvec
        system_u1 = system.U1System2D(system_cfg)
        system_u1.use_pfaffian = True
        #Modify the matrix D by han
        # [[0, -1],[1, 0]
        filled=np.zeros((2,2))
        filled[0,1]=-1
        filled[1,0]=1
        gamma_maj_sys_vec=system_u1.gamma_maj_sys_vec
        gamma_maj_sys=np.copy(gamma_maj_sys_vec[0])
        offset=lat_2x2.size*2
        #Setting mode l(0,1)+
        gamma_maj_sys[offset + 0:offset+2,offset + 0:offset+2] = filled
        #Setting mode r(0,0)-
        gamma_maj_sys[offset + 6:offset+8,offset + 6:offset+8] = filled
        # Setting mode l(0,0)-
        gamma_maj_sys[offset + 10:offset+12,offset + 10:offset+12] = filled
        # Setting mode r(0,1)+
        gamma_maj_sys[offset + 12:offset+14,offset + 12:offset+14] = filled
        system_u1.gamma_maj_sys_vec[0]=gamma_maj_sys
        mat_d = gamma_maj_sys[offset:gamma_maj_sys.shape[0],
                              offset:gamma_maj_sys.shape[1]]
        system_u1.mat_d_inv_vec[0]=np.linalg.inv(mat_d)
        system_u1.det_mat_d_vec[0]=np.linalg.slogdet(mat_d)[1]
        # We are checking the value for a single link
        el_energy_link_bare, _ = system_u1._compute_el_energy_op_and_grad_pfaffian(True)
        el_energy = 2 - 2 * np.prod(el_energy_link_bare)
        self.assertAlmostEqual(el_energy, 3.)