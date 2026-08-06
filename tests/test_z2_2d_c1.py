import unittest
from unittest import skip

import numpy as np

from ggpeps import utils
from ggpeps import system
from ggpeps import lattice
from ggpeps.lattice import Direction

from ggpeps.evaluator_manager import EvaluatorManager
from ggpeps.exacteval import ExactEvaluatorConfig, ExactEvaluator
from ggpeps.mc import MonteCarloEvaluatorConfig, MonteCarloEvaluator
from ggpeps.utils import compare_array_elementwise


class TestZ2SystemMethods(unittest.TestCase):
    def setUp(self):
        lat = lattice.Lattice2D(2, 2)
        paramvec_real = [[0.3, 0.5, 0.8, 0.0, 0.0, 0.0]]
        cfg = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg.paramvec = paramvec_real
        self.system_z2_2_2_real = system.Z2System2D(cfg)

        paramvec = [[0.3, 0.5, 0.8, 0.2, 0.1, 0.9]]
        cfg = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg.paramvec = paramvec
        self.system_z2_2_2 = system.Z2System2D(cfg)

    def test_tmat_numeric(self):
        # NOTE: This test compares against Mathematica data generated at a nonzero
        # t parameter. Use the generic ncopy=1 config in an unconstrained single-layer
        # convention so the parameter point is not projected to the pure-gauge t=0 sector.
        lat = lattice.Lattice2D(2, 2)
        paramvec_real = [[0.3, 0.5, 0.8, 0.0, 0.0, 0.0]]
        cfg = system.Z2System2D_Config(
            lat,
            0,
            0,
            0,
            0,
            None,
            ncopy=1,
            num_pg_layer=0,
            num_fermionic_layer=1,
            enforce_u1_symmetry=False,
        )
        cfg.paramvec = paramvec_real
        system_z2_2_2_real = system.Z2System2D(cfg)
        lay = 0
        site = 0
        tmat = system_z2_2_2_real.tmat_layervec_sitevec[lay][site]
        ref = np.array(
            [
                [0.0, -0.3 * 1.0j, 0.3 * 1.0j, 0.3, -0.3],
                [0.3 * 1.0j, 0, 0.5 * 1.0j, 0.8, 0.8 * 1.0j],
                [-0.3 * 1.0j, -0.5 * 1.0j, 0.0, -0.8 * 1.0j, -0.8],
                [-0.3, -0.8, 0.8 * 1.0j, 0, -0.5],
                [0.3, -0.8 * 1.0j, 0.8, 0.5, 0],
            ]
        )
        self.assertTrue(np.allclose(tmat, ref))

    def test_tmat_deriv_symb_real(self):
        eps = 1e-5
        t = 0.83
        y = 0.39
        z = 0.93
        lat = lattice.Lattice2D(2, 2)
        paramvec = [[t, y, z, 0, 0, 0]]
        paramvec_left = [[t, y - eps, z, 0, 0, 0]]
        paramvec_right = [[t, y + eps, z, 0, 0, 0]]

        cfg = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg.paramvec = paramvec
        cfg_left = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg_left.paramvec = paramvec_left
        cfg_right = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg_right.paramvec = paramvec_right

        system_z2_2_2 = system.Z2System2D(cfg)
        system_z2_2_2_left = system.Z2System2D(cfg_left)
        system_z2_2_2_right = system.Z2System2D(cfg_right)

        symbvec = system_z2_2_2.symbolvec
        # Derivative wrt y
        deriv_ana = system_z2_2_2.compute_tmat_deriv(symbvec[1])
        lay = 0
        site = 0
        tmat_left = system_z2_2_2_left.tmat_layervec_sitevec[lay][site]
        tmat_right = system_z2_2_2_right.tmat_layervec_sitevec[lay][site]
        deriv_num = (tmat_right - tmat_left) / (2 * eps)
        self.assertTrue(np.allclose(deriv_ana, deriv_num))

    def test_tmat_antisymmetric_real(self):
        lay = 0
        site = 0
        tmat = self.system_z2_2_2_real.tmat_layervec_sitevec[lay][site]
        self.assertTrue(utils.is_antisymmetric(tmat))

    def test_tmat_antisymmetric(self):
        lay = 0
        site = 0
        tmat = self.system_z2_2_2.tmat_layervec_sitevec[lay][site]
        self.assertTrue(utils.is_antisymmetric(tmat))

    def test_gamma_dirac_covariance_real(self):
        lay = 0
        site = 0
        gamma_dirac = self.system_z2_2_2_real.gamma_dirac_layervec_sitevec[lay][site]
        m, n = gamma_dirac.shape
        res = gamma_dirac @ np.transpose(np.conjugate(gamma_dirac))
        ref = 0.25 * np.eye(gamma_dirac.shape[0])
        self.assertTrue(np.allclose(ref, res))
        self.assertEqual(m, n)
        self.assertTrue(utils.is_antisymmetric(gamma_dirac))

    def test_gamma_dirac_covariance(self):
        lay = 0
        site = 0
        gamma_dirac = self.system_z2_2_2.gamma_dirac_layervec_sitevec[lay][site]
        m, n = gamma_dirac.shape
        res = gamma_dirac @ np.transpose(np.conjugate(gamma_dirac))
        ref = 0.25 * np.eye(gamma_dirac.shape[0])
        self.assertTrue(np.allclose(ref, res))
        self.assertEqual(m, n)
        self.assertTrue(utils.is_antisymmetric(gamma_dirac))

    def test_gamma_dirac_numeric(self):
        # We are comparing to Mathematica.
        # This check is useful for the mode-ordering.
        # Independent of the order, gamma_maj is always a covariance matrix
        lat = lattice.Lattice2D(2, 2)
        # NOTE: This test compares against Mathematica data generated at a nonzero
        # t parameter. Use the generic ncopy=1 config in an unconstrained single-layer
        # convention so the parameter point is not projected to the pure-gauge t=0 sector.
        paramvec = [[0.17, 0.35, 0.56, 0, 0, 0]]
        cfg = system.Z2System2D_Config(
            lat,
            0,
            0,
            0,
            0,
            None,
            ncopy=1,
            num_pg_layer=0,
            num_fermionic_layer=1,
            enforce_u1_symmetry=False,
        )
        cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(cfg)
        lay = 0
        site = 0
        gamma_dirac = system_z2_2_2.gamma_dirac_layervec_sitevec[lay][site]
        ref = np.asarray(
            [
                [
                    -1.73472 * 10**-18 - 1.73472 * 10**-18 * 1.0j,
                    -0.0753112 + 6.93889 * 10**-18 * 1.0j,
                    0.0753112 - 6.93889 * 10**-18 * 1.0j,
                    -1.04083 * 10**-17 - 0.0753112 * 1.0j,
                    0.0 + 0.0753112 * 1.0j,
                    -2.01965 * 10**-18 + 0.448788 * 1.0j,
                    0.0421743 - 0.0685332 * 1.0j,
                    0.0421743 - 0.0685332 * 1.0j,
                    0.0421743 - 0.0685332 * 1.0j,
                    0.0421743 - 0.0685332 * 1.0j,
                ],
                [
                    0.0753112 + 0.0 * 1.0j,
                    0.0 - 6.93889 * 10**-18 * 1.0j,
                    0.124232 + 0.0821891 * 1.0j,
                    -0.0256841 - 0.304589 * 1.0j,
                    0.304589 - 0.0256841 * 1.0j,
                    -0.0421743 - 0.0685332 * 1.0j,
                    -7.58942 * 10**-19 + 0.102576 * 1.0j,
                    3.46945 * 10**-18 + 0.0128029 * 1.0j,
                    -0.0733832 - 0.0605802 * 1.0j,
                    0.0733832 - 0.0605802 * 1.0j,
                ],
                [
                    -0.0753112 + 0.0 * 1.0j,
                    -0.124232 - 0.0821891 * 1.0j,
                    -6.93889 * 10**-18 + 0.0 * 1.0j,
                    -0.304589 + 0.0256841 * 1.0j,
                    0.0256841 + 0.304589 * 1.0j,
                    -0.0421743 - 0.0685332 * 1.0j,
                    -3.46945 * 10**-18 + 0.0128029 * 1.0j,
                    3.79471 * 10**-19 + 0.102576 * 1.0j,
                    0.0733832 - 0.0605802 * 1.0j,
                    -0.0733832 - 0.0605802 * 1.0j,
                ],
                [
                    8.15433 * 10**-20 + 0.0753112 * 1.0j,
                    0.0256841 + 0.304589 * 1.0j,
                    0.304589 - 0.0256841 * 1.0j,
                    6.93687 * 10**-18 + 4.33681 * 10**-18 * 1.0j,
                    -0.0821891 + 0.124232 * 1.0j,
                    -0.0421743 - 0.0685332 * 1.0j,
                    0.0733832 - 0.0605802 * 1.0j,
                    -0.0733832 - 0.0605802 * 1.0j,
                    -3.79858 * 10**-19 + 0.102576 * 1.0j,
                    7.65456 * 10**-21 + 0.0128029 * 1.0j,
                ],
                [
                    3.72404 * 10**-20 - 0.0753112 * 1.0j,
                    -0.304589 + 0.0256841 * 1.0j,
                    -0.0256841 - 0.304589 * 1.0j,
                    0.0821891 - 0.124232 * 1.0j,
                    -6.93687 * 10**-18 - 4.33681 * 10**-18 * 1.0j,
                    -0.0421743 - 0.0685332 * 1.0j,
                    -0.0733832 - 0.0605802 * 1.0j,
                    0.0733832 - 0.0605802 * 1.0j,
                    -2.44205 * 10**-21 + 0.0128029 * 1.0j,
                    -2.35267 * 10**-20 + 0.102576 * 1.0j,
                ],
                [
                    -2.01965 * 10**-18 - 0.448788 * 1.0j,
                    0.0421743 + 0.0685332 * 1.0j,
                    0.0421743 + 0.0685332 * 1.0j,
                    0.0421743 + 0.0685332 * 1.0j,
                    0.0421743 + 0.0685332 * 1.0j,
                    -1.73472 * 10**-18 + 1.73472 * 10**-18 * 1.0j,
                    -0.0753112 - 6.93889 * 10**-18 * 1.0j,
                    0.0753112 + 6.93889 * 10**-18 * 1.0j,
                    -1.04083 * 10**-17 + 0.0753112 * 1.0j,
                    0.0 - 0.0753112 * 1.0j,
                ],
                [
                    -0.0421743 + 0.0685332 * 1.0j,
                    -7.58942 * 10**-19 - 0.102576 * 1.0j,
                    3.46945 * 10**-18 - 0.0128029 * 1.0j,
                    -0.0733832 + 0.0605802 * 1.0j,
                    0.0733832 + 0.0605802 * 1.0j,
                    0.0753112 + 0.0 * 1.0j,
                    0.0 + 6.93889 * 10**-18 * 1.0j,
                    0.124232 - 0.0821891 * 1.0j,
                    -0.0256841 + 0.304589 * 1.0j,
                    0.304589 + 0.0256841 * 1.0j,
                ],
                [
                    -0.0421743 + 0.0685332 * 1.0j,
                    -3.46945 * 10**-18 - 0.0128029 * 1.0j,
                    3.79471 * 10**-19 - 0.102576 * 1.0j,
                    0.0733832 + 0.0605802 * 1.0j,
                    -0.0733832 + 0.0605802 * 1.0j,
                    -0.0753112 + 0.0 * 1.0j,
                    -0.124232 + 0.0821891 * 1.0j,
                    -6.93889 * 10**-18 + 0.0 * 1.0j,
                    -0.304589 - 0.0256841 * 1.0j,
                    0.0256841 - 0.304589 * 1.0j,
                ],
                [
                    -0.0421743 + 0.0685332 * 1.0j,
                    0.0733832 + 0.0605802 * 1.0j,
                    -0.0733832 + 0.0605802 * 1.0j,
                    -3.79858 * 10**-19 - 0.102576 * 1.0j,
                    7.65456 * 10**-21 - 0.0128029 * 1.0j,
                    8.15433 * 10**-20 - 0.0753112 * 1.0j,
                    0.0256841 - 0.304589 * 1.0j,
                    0.304589 + 0.0256841 * 1.0j,
                    6.93687 * 10**-18 - 4.33681 * 10**-18 * 1.0j,
                    -0.0821891 - 0.124232 * 1.0j,
                ],
                [
                    -0.0421743 + 0.0685332 * 1.0j,
                    -0.0733832 + 0.0605802 * 1.0j,
                    0.0733832 + 0.0605802 * 1.0j,
                    -2.44205 * 10**-21 - 0.0128029 * 1.0j,
                    -2.35267 * 10**-20 - 0.102576 * 1.0j,
                    3.72404 * 10**-20 + 0.0753112 * 1.0j,
                    -0.304589 - 0.0256841 * 1.0j,
                    -0.0256841 + 0.304589 * 1.0j,
                    0.0821891 + 0.124232 * 1.0j,
                    -6.93687 * 10**-18 + 4.33681 * 10**-18 * 1.0j,
                ],
            ]
        )
        compare_array_elementwise(self, np.real(ref), np.real(gamma_dirac))
        compare_array_elementwise(self, np.imag(ref), np.imag(gamma_dirac))

    def test_gamma_dirac_deriv_symb_real(self):
        eps = 1e-5
        t = 0.83
        y = 0.39
        z = 0.93
        lat = lattice.Lattice2D(2, 2)
        paramvec = [[t, y, z, 0, 0, 0]]
        paramvec_left = [[t, y - eps, z, 0, 0, 0]]
        paramvec_right = [[t, y + eps, z, 0, 0, 0]]

        cfg = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg_left = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg_right = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg.paramvec = paramvec
        cfg_left.paramvec = paramvec_left
        cfg_right.paramvec = paramvec_right

        system_z2_2_2 = system.Z2System2D(cfg)
        system_z2_2_2_left = system.Z2System2D(cfg_left)
        system_z2_2_2_right = system.Z2System2D(cfg_right)

        lay = 0
        site = 0
        uc_ind = 0
        symbvec = system_z2_2_2.symbolvec
        # Derivative wrt y
        deriv_ana = system_z2_2_2.compute_gamma_dirac_deriv(symbvec[1], 0, uc_ind)
        gamma_left = system_z2_2_2_left.gamma_dirac_layervec_sitevec[lay][site]
        gamma_right = system_z2_2_2_right.gamma_dirac_layervec_sitevec[lay][site]
        deriv_num = (gamma_right - gamma_left) / (2 * eps)
        self.assertTrue(np.allclose(deriv_ana, deriv_num))

    def test_gamma_maj_covariance_real(self):
        lay = 0
        site = 0
        gamma_maj = self.system_z2_2_2_real.gamma_maj_layervec_sitevec[lay][site]
        m, n = gamma_maj.shape
        self.assertEqual(m, n)
        self.assertEqual(m, 10)
        self.assertTrue(np.allclose(np.imag(gamma_maj), 0))
        self.assertTrue(utils.is_antisymmetric(gamma_maj))
        self.assertTrue(np.allclose(gamma_maj @ gamma_maj, -np.eye(m)))
        self.assertTrue(np.allclose(gamma_maj @ np.transpose(gamma_maj), np.eye(m)))

    def test_gamma_maj_covariance(self):
        lay = 0
        site = 0
        gamma_maj = self.system_z2_2_2.gamma_maj_layervec_sitevec[lay][site]
        m, n = gamma_maj.shape
        self.assertEqual(m, n)
        self.assertEqual(m, 10)
        self.assertTrue(np.allclose(np.imag(gamma_maj), 0))
        self.assertTrue(utils.is_antisymmetric(gamma_maj))
        self.assertTrue(np.allclose(gamma_maj @ gamma_maj, -np.eye(m)))
        self.assertTrue(np.allclose(gamma_maj @ np.transpose(gamma_maj), np.eye(m)))

    def test_gamma_maj_numeric_real(self):
        # We are comparing to Mathematica.
        # This check is useful for the mode-ordering.
        # Independent of the order, gamma_maj is always a covariance matrix
        lat = lattice.Lattice2D(2, 2)
        # NOTE: This test compares against Mathematica data generated at a nonzero
        # t parameter. Use the generic ncopy=1 config in an unconstrained single-layer
        # convention so the parameter point is not projected to the pure-gauge t=0 sector.
        paramvec = [[0.17, 0.35, 0.56, 0, 0, 0]]
        cfg = system.Z2System2D_Config(
            lat,
            0,
            0,
            0,
            0,
            None,
            ncopy=1,
            num_pg_layer=0,
            num_fermionic_layer=1,
            enforce_u1_symmetry=False,
        )
        cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(cfg)
        lay = 0
        site = 0
        gamma_maj = system_z2_2_2.gamma_maj_layervec_sitevec[lay][site]
        ref = np.asarray(
            [
                [
                    0,
                    0.8975767509856909,
                    -0.06627386700925886,
                    -0.137066406769149,
                    0.23497098303282687,
                    -0.137066406769149,
                    0.08434855801178401,
                    0.013556018251893853,
                    0.08434855801178401,
                    -0.2876888317901919,
                ],
                [
                    -0.8975767509856909,
                    0,
                    0.137066406769149,
                    0.23497098303282687,
                    0.137066406769149,
                    -0.06627386700925886,
                    0.2876888317901919,
                    0.08434855801178401,
                    -0.013556018251893853,
                    0.08434855801178401,
                ],
                [
                    0.06627386700925886,
                    -0.137066406769149,
                    0,
                    0.2051526809871836,
                    0.2484631458705544,
                    -0.13877244593263055,
                    -0.1981345076351612,
                    0.4880175511092357,
                    0.7559443427596008,
                    -0.06979228401520404,
                ],
                [
                    0.137066406769149,
                    -0.23497098303282687,
                    -0.2051526809871836,
                    0,
                    -0.18998407043978507,
                    -0.2484631458705544,
                    0.7303385305060236,
                    -0.09539809626878132,
                    0.17252869538158389,
                    -0.46241173885565845,
                ],
                [
                    -0.23497098303282687,
                    -0.137066406769149,
                    -0.2484631458705544,
                    0.18998407043978507,
                    0,
                    0.2051526809871836,
                    -0.46241173885565845,
                    -0.17252869538158389,
                    -0.09539809626878132,
                    -0.7303385305060236,
                ],
                [
                    0.137066406769149,
                    0.06627386700925886,
                    0.13877244593263055,
                    0.2484631458705544,
                    -0.2051526809871836,
                    0,
                    0.06979228401520404,
                    0.7559443427596008,
                    -0.4880175511092357,
                    -0.1981345076351612,
                ],
                [
                    -0.08434855801178401,
                    -0.2876888317901919,
                    0.1981345076351612,
                    -0.7303385305060236,
                    0.46241173885565845,
                    -0.06979228401520404,
                    0,
                    0.2051526809871836,
                    -0.16437825818620777,
                    -0.22285733361697713,
                ],
                [
                    -0.013556018251893853,
                    -0.08434855801178401,
                    -0.4880175511092357,
                    0.09539809626878132,
                    0.17252869538158389,
                    -0.755944342759601,
                    -0.2051526809871836,
                    0,
                    -0.2740689581241317,
                    0.16437825818620777,
                ],
                [
                    -0.08434855801178401,
                    0.013556018251893853,
                    -0.755944342759601,
                    -0.17252869538158389,
                    0.09539809626878132,
                    0.4880175511092357,
                    0.16437825818620777,
                    0.2740689581241317,
                    0,
                    0.2051526809871836,
                ],
                [
                    0.2876888317901919,
                    -0.08434855801178401,
                    0.06979228401520404,
                    0.46241173885565845,
                    0.7303385305060236,
                    0.1981345076351612,
                    0.22285733361697713,
                    -0.16437825818620777,
                    -0.2051526809871836,
                    0,
                ],
            ]
        )

        compare_array_elementwise(self, ref, gamma_maj, print_vals=False)

    def test_gamma_maj_sys_covariance_real(self):
        gamma_maj = self.system_z2_2_2_real.gamma_maj_sys_vec[0]
        m, n = gamma_maj.shape
        self.assertEqual(m, n)
        self.assertTrue(np.allclose(np.imag(gamma_maj), 0))
        self.assertTrue(utils.is_antisymmetric(gamma_maj))
        self.assertTrue(np.allclose(gamma_maj @ gamma_maj, -np.eye(m)))
        self.assertTrue(np.allclose(gamma_maj @ np.transpose(gamma_maj), np.eye(m)))

    def test_gamma_maj_sys_covariance(self):
        gamma_maj = self.system_z2_2_2.gamma_maj_sys_vec[0]
        m, n = gamma_maj.shape
        self.assertEqual(m, n)
        self.assertTrue(np.allclose(np.imag(gamma_maj), 0))
        self.assertTrue(utils.is_antisymmetric(gamma_maj))
        self.assertTrue(np.allclose(gamma_maj @ gamma_maj, -np.eye(m)))
        self.assertTrue(np.allclose(gamma_maj @ np.transpose(gamma_maj), np.eye(m)))

    def test_gamma_ungauged_site(self):
        # check single site ungauged covariance matrix
        gamma_X = self.system_z2_2_2_real.gamma_gauge_neutral_vec[0][Direction.X]
        gamma_Y = self.system_z2_2_2_real.gamma_gauge_neutral_vec[0][Direction.Y]
        self.assertTrue(
            np.allclose(
                gamma_X,
                np.array(
                    [
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, -1, 0, 0],
                        [-1, 0, 0, 0],
                    ]
                ),
            )
        )
        self.assertTrue(
            np.allclose(
                gamma_Y,
                np.array(
                    [
                        [0, 0, 1, 0],
                        [0, 0, 0, -1],
                        [-1, 0, 0, 0],
                        [0, 1, 0, 0],
                    ]
                ),
            )
        )

        for gamma in [gamma_X, gamma_Y]:
            m, n = gamma.shape
            self.assertTrue(utils.is_antisymmetric(gamma))
            self.assertTrue(np.allclose(gamma @ gamma, -np.eye(m)))
            self.assertTrue(np.allclose(gamma @ np.transpose(gamma), np.eye(m)))

    def test_gamma_in_sys_covariance(self):
        gamma_in = self.system_z2_2_2_real.gamma_in_sys_vec[0]
        m, n = gamma_in.shape
        self.assertEqual(m, n)
        self.assertTrue(utils.is_antisymmetric(gamma_in))
        self.assertTrue(np.allclose(gamma_in @ gamma_in, -np.eye(m)))
        self.assertTrue(np.allclose(gamma_in @ np.transpose(gamma_in), np.eye(m)))

    def test_part_d_covmat_real(self):
        # If t=0, then mat_d must be a valid covariance matrix
        lat = lattice.Lattice2D(2, 2)
        paramvec = [[0.0, 0.5, 0.8, 0, 0, 0]]
        cfg = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(cfg)
        mat = system_z2_2_2.mat_d_vec[0]
        m, _ = mat.shape
        self.assertTrue(utils.is_antisymmetric(mat))
        self.assertTrue(np.allclose(mat @ mat, -np.eye(m)))
        self.assertTrue(np.allclose(mat @ np.transpose(mat), np.eye(m)))

    def test_derivative_y_real(self):

        # This is comparison with Mathematica
        ref = np.asarray(
            [
                [
                    0,
                    0.08258109830595964,
                    0.05343482831562092,
                    -0.040109484641008664,
                    -0.1894507549372015,
                    -0.040109484641008664,
                    -0.06800796331079029,
                    -0.16155227626741986,
                    -0.06800796331079029,
                    0.08133330698540256,
                ],
                [
                    -0.08258109830595964,
                    0,
                    0.040109484641008664,
                    -0.1894507549372015,
                    0.040109484641008664,
                    0.05343482831562092,
                    -0.08133330698540256,
                    -0.06800796331079029,
                    0.16155227626741986,
                    -0.06800796331079029,
                ],
                [
                    -0.05343482831562092,
                    -0.040109484641008664,
                    0,
                    -0.149983826896498,
                    0.8066611217161592,
                    -0.34826252828193405,
                    -0.5416620988290691,
                    -0.6347369627047637,
                    -0.029060877939980495,
                    -0.06401398593571395,
                ],
                [
                    0.040109484641008664,
                    0.1894507549372015,
                    0.14998382689649795,
                    0,
                    -0.3069719791289543,
                    -0.8066611217161592,
                    -0.008415603363490592,
                    -0.043368711359223974,
                    0.5623073734055589,
                    0.6140916881282736,
                ],
                [
                    0.1894507549372015,
                    -0.040109484641008664,
                    -0.8066611217161592,
                    0.3069719791289543,
                    0,
                    -0.149983826896498,
                    0.6140916881282736,
                    -0.562307373405559,
                    -0.043368711359223974,
                    0.008415603363490565,
                ],
                [
                    0.040109484641008664,
                    -0.05343482831562092,
                    0.3482625282819341,
                    0.8066611217161592,
                    0.14998382689649795,
                    0,
                    0.06401398593571395,
                    -0.029060877939980495,
                    0.6347369627047635,
                    -0.5416620988290691,
                ],
                [
                    0.06800796331079029,
                    0.08133330698540256,
                    0.5416620988290691,
                    0.008415603363490565,
                    -0.6140916881282736,
                    -0.06401398593571395,
                    0,
                    -0.149983826896498,
                    -0.3276172537054442,
                    -0.8273063962926491,
                ],
                [
                    0.16155227626741986,
                    0.06800796331079029,
                    0.6347369627047635,
                    0.043368711359223974,
                    0.5623073734055589,
                    0.02906087793998055,
                    0.14998382689649795,
                    0,
                    -0.7860158471396694,
                    0.3276172537054442,
                ],
                [
                    0.06800796331079029,
                    -0.16155227626741986,
                    0.02906087793998055,
                    -0.562307373405559,
                    0.043368711359223974,
                    -0.6347369627047637,
                    0.3276172537054442,
                    0.7860158471396694,
                    0,
                    -0.149983826896498,
                ],
                [
                    -0.08133330698540256,
                    0.06800796331079029,
                    0.06401398593571395,
                    -0.6140916881282736,
                    -0.008415603363490592,
                    0.5416620988290691,
                    0.8273063962926492,
                    -0.3276172537054442,
                    0.14998382689649795,
                    0,
                ],
            ]
        )
        lay = 0
        uc_ind = 0
        lat = lattice.Lattice2D(2, 2)
        # NOTE: This test compares against Mathematica data generated at a nonzero
        # t parameter. Use the generic ncopy=1 config in an unconstrained single-layer
        # convention so the parameter point is not projected to the pure-gauge t=0 sector.
        paramvec = [[0.17, 0.35, 0.56, 0, 0, 0]]
        cfg = system.Z2System2D_Config(
            lat,
            0,
            0,
            0,
            0,
            None,
            ncopy=1,
            num_pg_layer=0,
            num_fermionic_layer=1,
            enforce_u1_symmetry=False,
        )
        cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(cfg)
        res = system_z2_2_2.compute_gamma_maj_deriv(system_z2_2_2.symbolvec[1], lay, uc_ind)
        self.assertTrue(np.allclose(ref, res))

    def test_derivative_z(self):
        # This is comparison with Mathematica
        ref = np.asarray(
            [
                [
                    0,
                    0.13340023572501172,
                    0.23694022460781514,
                    0.027898478669781623,
                    -0.15541340987751348,
                    0.027898478669781623,
                    0.04076340736515084,
                    -0.16827833857288266,
                    0.04076340736515084,
                    0.224075295912446,
                ],
                [
                    -0.13340023572501172,
                    0,
                    -0.027898478669781623,
                    -0.15541340987751348,
                    -0.027898478669781623,
                    0.23694022460781514,
                    -0.224075295912446,
                    0.04076340736515084,
                    0.16827833857288266,
                    0.04076340736515084,
                ],
                [
                    -0.23694022460781514,
                    0.027898478669781623,
                    0,
                    -1.4457765039721213,
                    -0.643152566068254,
                    -0.1821533692352031,
                    0.17445048782141942,
                    0.5296884884008861,
                    0.3045933801892958,
                    0.050644620390171174,
                ],
                [
                    -0.027898478669781623,
                    0.15541340987751348,
                    1.4457765039721213,
                    0,
                    -0.1154532513726973,
                    0.643152566068254,
                    0.3379434391205487,
                    0.08399467932142413,
                    -0.1411004288901665,
                    -0.5630385473321391,
                ],
                [
                    0.15541340987751348,
                    0.027898478669781623,
                    0.643152566068254,
                    0.1154532513726973,
                    0,
                    -1.4457765039721213,
                    -0.5630385473321391,
                    0.14110042889016644,
                    0.08399467932142413,
                    -0.3379434391205487,
                ],
                [
                    -0.027898478669781623,
                    -0.23694022460781514,
                    0.1821533692352031,
                    -0.643152566068254,
                    1.4457765039721213,
                    0,
                    -0.050644620390171174,
                    0.3045933801892958,
                    -0.5296884884008861,
                    0.17445048782141942,
                ],
                [
                    -0.04076340736515084,
                    0.224075295912446,
                    -0.17445048782141942,
                    -0.3379434391205487,
                    0.5630385473321391,
                    0.050644620390171174,
                    0,
                    -1.4457765039721213,
                    -0.1488033103039502,
                    0.609802507137001,
                ],
                [
                    0.16827833857288266,
                    -0.04076340736515084,
                    -0.5296884884008861,
                    -0.08399467932142413,
                    -0.1411004288901665,
                    -0.3045933801892954,
                    1.4457765039721213,
                    0,
                    0.676502624999507,
                    0.1488033103039502,
                ],
                [
                    -0.04076340736515084,
                    -0.16827833857288266,
                    -0.3045933801892954,
                    0.14110042889016644,
                    -0.08399467932142413,
                    0.5296884884008861,
                    0.1488033103039502,
                    -0.676502624999507,
                    0,
                    -1.4457765039721213,
                ],
                [
                    -0.224075295912446,
                    -0.04076340736515084,
                    -0.050644620390171174,
                    0.5630385473321391,
                    0.3379434391205487,
                    -0.17445048782141942,
                    -0.6098025071370011,
                    -0.1488033103039502,
                    1.4457765039721213,
                    0,
                ],
            ]
        )

        lay = 0
        uc_ind = 0
        lat = lattice.Lattice2D(2, 2)
        # NOTE: This test compares against Mathematica data generated at a nonzero
        # t parameter. Use the generic ncopy=1 config in an unconstrained single-layer
        # convention so the parameter point is not projected to the pure-gauge t=0 sector.
        paramvec = [[0.17, 0.35, 0.56, 0, 0, 0]]
        cfg = system.Z2System2D_Config(
            lat,
            0,
            0,
            0,
            0,
            None,
            ncopy=1,
            num_pg_layer=0,
            num_fermionic_layer=1,
            enforce_u1_symmetry=False,
        )
        cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(cfg)
        res = system_z2_2_2.compute_gamma_maj_deriv(system_z2_2_2.symbolvec[2], lay, uc_ind)
        compare_array_elementwise(self, ref, res)

    def test_derivative_t_real(self):
        # This is comparison with Mathematica
        ref = np.array(
            [
                [
                    0,
                    -1.1432704475880655,
                    -0.3499169542672254,
                    -0.7236918826890345,
                    1.2406146560383449,
                    -0.7236918826890345,
                    0.4453488508855597,
                    0.07157392246375063,
                    0.4453488508855597,
                    -1.5189576878418198,
                ],
                [
                    1.1432704475880655,
                    0,
                    0.7236918826890345,
                    1.2406146560383449,
                    0.7236918826890345,
                    -0.3499169542672254,
                    1.5189576878418198,
                    0.4453488508855597,
                    -0.07157392246375063,
                    0.4453488508855597,
                ],
                [
                    0.3499169542672254,
                    -0.7236918826890345,
                    0,
                    0.01891037755325542,
                    -0.24288558325282242,
                    0.13634956066451026,
                    -0.18016238318203856,
                    -0.04381282251752838,
                    -0.06272320007078382,
                    0.19907276073529406,
                ],
                [
                    0.7236918826890345,
                    -1.2406146560383449,
                    -0.01891037755325542,
                    0,
                    -0.4352856631295225,
                    0.24288558325282242,
                    -0.3485408119678002,
                    -0.08674485116172226,
                    -0.10565522871497779,
                    0.32963043441454465,
                ],
                [
                    -1.2406146560383449,
                    -0.7236918826890345,
                    0.24288558325282242,
                    0.4352856631295225,
                    0,
                    0.01891037755325542,
                    0.32963043441454465,
                    0.10565522871497779,
                    -0.08674485116172226,
                    0.3485408119678002,
                ],
                [
                    0.7236918826890345,
                    0.3499169542672254,
                    -0.13634956066451026,
                    -0.24288558325282242,
                    -0.01891037755325542,
                    0,
                    -0.19907276073529406,
                    -0.06272320007078382,
                    0.04381282251752838,
                    -0.18016238318203856,
                ],
                [
                    -0.4453488508855597,
                    -1.5189576878418198,
                    0.18016238318203856,
                    0.3485408119678002,
                    -0.32963043441454476,
                    0.19907276073529406,
                    0,
                    0.01891037755325542,
                    -0.14946805123250612,
                    0.5287031951498388,
                ],
                [
                    -0.07157392246375063,
                    -0.4453488508855597,
                    0.04381282251752838,
                    0.08674485116172226,
                    -0.10565522871497779,
                    0.06272320007078391,
                    -0.01891037755325542,
                    0,
                    -0.04293202864419393,
                    0.14946805123250612,
                ],
                [
                    -0.4453488508855597,
                    0.07157392246375063,
                    0.06272320007078391,
                    0.10565522871497779,
                    0.08674485116172226,
                    -0.04381282251752838,
                    0.14946805123250612,
                    0.04293202864419393,
                    0,
                    0.01891037755325542,
                ],
                [
                    1.5189576878418198,
                    -0.4453488508855597,
                    -0.19907276073529406,
                    -0.32963043441454476,
                    -0.3485408119678002,
                    0.18016238318203856,
                    -0.5287031951498388,
                    -0.14946805123250612,
                    -0.01891037755325542,
                    0,
                ],
            ]
        )

        lay = 0
        uc_ind = 0
        lat = lattice.Lattice2D(2, 2)
        # NOTE: This test compares against Mathematica data generated at a nonzero
        # t parameter. Use the generic ncopy=1 config in an unconstrained single-layer
        # convention so the parameter point is not projected to the pure-gauge t=0 sector.
        paramvec = [[0.17, 0.35, 0.56, 0, 0, 0]]
        cfg = system.Z2System2D_Config(
            lat,
            0,
            0,
            0,
            0,
            None,
            ncopy=1,
            num_pg_layer=0,
            num_fermionic_layer=1,
            enforce_u1_symmetry=False,
        )
        cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(cfg)
        res = system_z2_2_2.compute_gamma_maj_deriv(system_z2_2_2.symbolvec[0], lay, uc_ind)
        compare_array_elementwise(self, ref, res)

    def test_gamma_maj_deriv_symb_y_real(self):
        eps = 1e-5
        t = 0.83
        y = 0.39
        z = 0.93
        lat = lattice.Lattice2D(2, 2)
        paramvec = [[t, y, z, 0, 0, 0]]
        paramvec_left = [[t, y - eps, z, 0, 0, 0]]
        paramvec_right = [[t, y + eps, z, 0, 0, 0]]

        cfg = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg_left = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg_right = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg.paramvec = paramvec
        cfg_left.paramvec = paramvec_left
        cfg_right.paramvec = paramvec_right

        system_z2_2_2 = system.Z2System2D(cfg)
        system_z2_2_2_left = system.Z2System2D(cfg_left)
        system_z2_2_2_right = system.Z2System2D(cfg_right)

        lay = 0
        site = 0
        uc_ind = 0
        symbvec = system_z2_2_2.symbolvec
        # Derivative wrt y
        deriv_ana = system_z2_2_2.compute_gamma_maj_deriv(symbvec[1], lay, uc_ind)
        gamma_left = system_z2_2_2_left.gamma_maj_layervec_sitevec[lay][site]
        gamma_right = system_z2_2_2_right.gamma_maj_layervec_sitevec[lay][site]
        deriv_num = (gamma_right - gamma_left) / (2 * eps)
        self.assertTrue(np.allclose(deriv_ana, deriv_num))

    def test_derivative_t_numeric_real(self):
        # This is comparison of the analytic derivative against the numeric derivative
        eps = 1e-5
        lat = lattice.Lattice2D(2, 2)
        paramvec = [[0.17, 0.35, 0.56, 0, 0, 0]]
        paramvec_left = [[0.17 - eps, 0.35, 0.56, 0, 0, 0]]
        paramvec_right = [[0.17 + eps, 0.35, 0.56, 0, 0, 0]]

        # NOTE: This test compares against Mathematica data generated at a nonzero
        # t parameter. Use the generic ncopy=1 config in an unconstrained single-layer
        # convention so the parameter point is not projected to the pure-gauge t=0 sector.
        cfg = system.Z2System2D_Config(
            lat,
            0,
            0,
            0,
            0,
            None,
            ncopy=1,
            num_pg_layer=0,
            num_fermionic_layer=1,
            enforce_u1_symmetry=False,
        )
        cfg_left = system.Z2System2D_Config(
            lat,
            0,
            0,
            0,
            0,
            None,
            ncopy=1,
            num_pg_layer=0,
            num_fermionic_layer=1,
            enforce_u1_symmetry=False,
        )
        cfg_right = system.Z2System2D_Config(
            lat,
            0,
            0,
            0,
            0,
            None,
            ncopy=1,
            num_pg_layer=0,
            num_fermionic_layer=1,
            enforce_u1_symmetry=False,
        )

        cfg.paramvec = paramvec
        cfg_left.paramvec = paramvec_left
        cfg_right.paramvec = paramvec_right

        system_z2_2_2 = system.Z2System2D(cfg)
        system_z2_2_2_left = system.Z2System2D(cfg_left)
        system_z2_2_2_right = system.Z2System2D(cfg_right)

        lay = 0
        site = 0
        uc_ind = 0
        deriv_ana = system_z2_2_2.compute_gamma_maj_deriv(system_z2_2_2.symbolvec[0], lay, uc_ind)
        gamma_left = system_z2_2_2_left.gamma_maj_layervec_sitevec[lay][site]
        gamma_right = system_z2_2_2_right.gamma_maj_layervec_sitevec[lay][site]
        deriv_num = (gamma_right - gamma_left) / (2 * eps)

        compare_array_elementwise(self, deriv_num, deriv_ana, print_vals=False)

    def test_derivative_y_numeric_pure_gauge_real(self):
        # This is comparison of the analytic derivative against the numeric derivative
        eps = 1e-5
        lat = lattice.Lattice2D(2, 2)
        paramvec = [[0.17, 0.35, 0.56, 0, 0, 0]]
        paramvec_left = [[0.17, 0.35 - eps, 0.56, 0, 0, 0]]
        paramvec_right = [[0.17, 0.35 + eps, 0.56, 0, 0, 0]]

        cfg = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg_left = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg_right = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)

        cfg.paramvec = paramvec
        cfg_left.paramvec = paramvec_left
        cfg_right.paramvec = paramvec_right

        system_z2_2_2 = system.Z2System2D(cfg)
        system_z2_2_2_left = system.Z2System2D(cfg_left)
        system_z2_2_2_right = system.Z2System2D(cfg_right)

        lay = 0
        site = 0
        uc_ind = 0
        deriv_ana = system_z2_2_2.compute_gamma_maj_deriv(system_z2_2_2.symbolvec[1], lay, uc_ind)
        gamma_left = system_z2_2_2_left.gamma_maj_layervec_sitevec[lay][site]
        gamma_right = system_z2_2_2_right.gamma_maj_layervec_sitevec[lay][site]
        deriv_num = (gamma_right - gamma_left) / (2 * eps)

        compare_array_elementwise(self, deriv_num, deriv_ana, print_vals=False)

    def test_derivative_z_numeric_pure_gauge_real(self):
        # This is comparison of the analytic derivative against the numeric derivative
        eps = 1e-5
        lat = lattice.Lattice2D(2, 2)
        paramvec = [[0.17, 0.35, 0.56, 0, 0, 0]]
        paramvec_left = [[0.17, 0.35, 0.56 - eps, 0, 0, 0]]
        paramvec_right = [[0.17, 0.35, 0.56 + eps, 0, 0, 0]]

        cfg = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg_left = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg_right = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg.paramvec = paramvec
        cfg_left.paramvec = paramvec_left
        cfg_right.paramvec = paramvec_right

        system_z2_2_2 = system.Z2System2D(cfg)
        system_z2_2_2_left = system.Z2System2D(cfg_left)
        system_z2_2_2_right = system.Z2System2D(cfg_right)

        lay = 0
        site = 0
        uc_ind = 0
        deriv_ana = system_z2_2_2.compute_gamma_maj_deriv(system_z2_2_2.symbolvec[2], lay, uc_ind)
        gamma_left = system_z2_2_2_left.gamma_maj_layervec_sitevec[lay][site]
        gamma_right = system_z2_2_2_right.gamma_maj_layervec_sitevec[lay][site]
        deriv_num = (gamma_right - gamma_left) / (2 * eps)

        compare_array_elementwise(self, deriv_num, deriv_ana, print_vals=False)

    def test_derivative_y_sys_real(self):
        lay = 0
        uc_ind = 0
        lat = lattice.Lattice2D(2, 2)
        paramvec = [[0.17, 0.35, 0.56, 0, 0, 0]]
        cfg = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(cfg)
        res = system_z2_2_2.gamma_maj_sys_deriv_vec(system_z2_2_2.symbolvec[1])[lay, uc_ind]
        self.assertTrue(utils.is_antisymmetric(res))

    def test_derivative_z_sys_real(self):
        lay = 0
        uc_ind = 0
        lat = lattice.Lattice2D(2, 2)
        paramvec = [[0.17, 0.35, 0.56, 0, 0, 0]]
        cfg = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(cfg)
        res = system_z2_2_2.gamma_maj_sys_deriv_vec(system_z2_2_2.symbolvec[2])[lay, uc_ind]
        self.assertTrue(utils.is_antisymmetric(res))

    def test_derivative_t_sys_real(self):
        lay = 0
        uc_ind = 0
        lat = lattice.Lattice2D(2, 2)
        # NOTE: This test compares against Mathematica data generated at a nonzero
        # t parameter. Use the generic ncopy=1 config in an unconstrained single-layer
        # convention so the parameter point is not projected to the pure-gauge t=0 sector.
        paramvec = [[0.17, 0.35, 0.56, 0, 0, 0]]
        cfg = system.Z2System2D_Config(
            lat,
            0,
            0,
            0,
            0,
            None,
            ncopy=1,
            num_pg_layer=0,
            num_fermionic_layer=1,
            enforce_u1_symmetry=False,
        )
        cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(cfg)
        res = system_z2_2_2.gamma_maj_sys_deriv_vec(system_z2_2_2.symbolvec[0])[lay, uc_ind]
        self.assertTrue(utils.is_antisymmetric(res))

    def test_derivative_gamma_sys_finite_diff(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # There is no sampling involved here. The gauge field is 0.
        eps = 1e-5

        # These numbers are arbitrary
        tr = 0.2
        yr = 0.3
        zr = 0.8
        ti = 0.1
        yi = 0.39
        zi = 0.74
        paramvec = [[tr, yr, zr, ti, yi, zi]]
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_Config(lat_2x2, 1.0, None, None, 0, None, ncopy=1, num_fermionic_layer=0)
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(system_cfg)
        symbolvec = system_z2_2_2.symbolvec
        zeroed_param_inds = {param_ind for _, _, param_ind in system_cfg.zeroed_params}
        for ind in range(len(symbolvec)):
            # The generic pure-gauge ncopy=1 config explicitly removes the t
            # parameters from the variational manifold by zeroing them. A finite
            # difference in a zeroed parameter is therefore projected out, while
            # compute_gamma_maj_deriv returns the raw analytic derivative of the
            # symbolic T matrix. Skip such parameters here and test only genuine
            # variational directions.
            if ind in zeroed_param_inds:
                continue
            with self.subTest(symbol=symbolvec[ind]):
                paramvec_left = np.copy(paramvec)
                paramvec_right = np.copy(paramvec)
                # We are only modifying the first layer (there is only one)
                paramvec_left[0, ind] -= eps
                paramvec_right[0, ind] += eps
                system_cfg_left = system.Z2System2D_Config(
                    lat_2x2, 1.0, None, None, 0, None, ncopy=1, num_fermionic_layer=0
                )
                system_cfg_right = system.Z2System2D_Config(
                    lat_2x2, 1.0, None, None, 0, None, ncopy=1, num_fermionic_layer=0
                )
                system_cfg_left.paramvec = paramvec_left
                system_cfg_right.paramvec = paramvec_right

                system_z2_2_2_left = system.Z2System2D(system_cfg_left)
                system_z2_2_2_right = system.Z2System2D(system_cfg_right)

                deriv_maj_sys = system_z2_2_2.gamma_maj_sys_deriv_vec(symbolvec[ind])[0]
                deriv_maj_sys_left = system_z2_2_2_left.gamma_maj_sys_vec[0]
                deriv_maj_sys_right = system_z2_2_2_right.gamma_maj_sys_vec[0]

                deriv_maj_sys_num = (deriv_maj_sys_right - deriv_maj_sys_left) / (2 * eps)

                self.assertTrue(np.allclose(deriv_maj_sys_num, deriv_maj_sys))

    def test_norm_minimal(self):
        # This update is a nullop since we initialize the gauge-field with 0
        zeroarr = (np.zeros((1, 1)),)
        sys = self.system_z2_2_2_real
        logdet_inc = 2 * sys.update_lognorm_inc(
            sys.incdet_vec,
            sys.wi_gamma_in_vec,
            sys.gamma_in_sys_vec,
            sys.mat_d_vec,
            0,
            np.asarray(zeroarr),
            all_factors=False,
        )
        # This is equivalent to
        # logdet_inc = sys.incdet.det()
        diff = sys.mat_d_inv_vec[0] - sys.gamma_in_sys_vec[0]
        sign, logdet = np.linalg.slogdet(diff)
        self.assertGreater(sign, 0)
        self.assertAlmostEqual(logdet_inc, logdet)

    def test_norm_incremental(self):
        # Test that the incremental update is equivalent to re-calculation of the norm
        # This update is a nullop since we initialize the gauge-field with 0
        zeroarr = (np.zeros((1, 1)),)
        sys = self.system_z2_2_2_real
        weight_inc = sys.update_lognorm_inc(
            sys.incdet_vec,
            sys.wi_gamma_in_vec,
            sys.gamma_in_sys_vec,
            sys.mat_d_vec,
            0,
            np.asarray(zeroarr),
            all_factors=True,
        )
        weight_recalc = sys.calculate_lognorm(incremental=False, all_factors=True)
        self.assertAlmostEqual(weight_inc, weight_recalc)

    def test_norm_incremental_update(self):
        # Test that the incremental update is equivalent to re-calculation of the norm
        ind = 0
        theta = np.array([[-1.0]])
        weight_inc = self.system_z2_2_2_real.calculate_weight_attempt(ind, theta, all_factors=True)
        self.system_z2_2_2_real.update_gauge_ind(ind, theta)
        weight_recalc = self.system_z2_2_2_real.calculate_lognorm(incremental=False, all_factors=True)
        self.assertAlmostEqual(weight_inc, weight_recalc)

    def test_grad_over_norm_pure_gauge_real(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # There is no sampling involved here. The gauge field is 0.
        eps = 1e-5
        lat = lattice.Lattice2D(2, 2)
        t = 0.0
        y = 0.1
        z = 0.2
        paramvec = [[t, y, z, 0, 0, 0]]
        paramvec_left = [[t, y - eps, z, 0, 0, 0]]
        paramvec_right = [[t, y + eps, z, 0, 0, 0]]

        cfg = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg_left = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)
        cfg_right = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=1, num_fermionic_layer=0)

        cfg.paramvec = paramvec
        cfg_left.paramvec = paramvec_left
        cfg_right.paramvec = paramvec_right

        system_z2_2_2 = system.Z2System2D(cfg)
        system_z2_2_2_left = system.Z2System2D(cfg_left)
        system_z2_2_2_right = system.Z2System2D(cfg_right)

        # We are using here that the gradient of the d/dx log(f(x)) is [d/dx f(x)]/f(x).
        # Thus, the d/dx log(norm(x)) = [d/dx norm(x)]/ norm(x) which is
        # exactly the function grad_over_norm.
        # The second symbol is y
        lay = 0
        uc_ind = 0
        deriv_ana = system_z2_2_2.grad_over_norm_vec[lay, uc_ind, 1]
        lognorm_left = system_z2_2_2_left.calculate_lognorm(incremental=True, all_factors=True)
        lognorm_right = system_z2_2_2_right.calculate_lognorm(incremental=True, all_factors=True)
        deriv_num = (lognorm_right - lognorm_left) / (2 * eps)
        # print("Analytical",deriv_ana)
        # print("Numerical",deriv_num)
        self.assertAlmostEqual(deriv_ana, deriv_num)

    def test_grad_over_norm(self):
        # This is comparison of the analytic derivative against the numeric derivative
        eps = 1e-5
        tr = 0.17
        yr = 0.35
        zr = 0.56
        ti = 0.25
        yi = 0.21
        zi = 0.65
        lay = 0
        uc_ind = 0
        paramvec = [[tr, yr, zr, ti, yi, zi]]
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_Config(lat_2x2, 1.0, None, None, None, None, ncopy=1, num_fermionic_layer=0)
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(system_cfg)
        symbolvec = self.system_z2_2_2_real.symbolvec
        for ind in range(len(symbolvec)):
            with self.subTest(ind=ind):
                paramvec_left = np.copy(paramvec)
                paramvec_right = np.copy(paramvec)
                paramvec_left[0, ind] -= eps
                paramvec_right[0, ind] += eps
                system_cfg_left = system.Z2System2D_Config(
                    lat_2x2, 1.0, None, None, None, None, ncopy=1, num_fermionic_layer=0
                )
                system_cfg_right = system.Z2System2D_Config(
                    lat_2x2, 1.0, None, None, None, None, ncopy=1, num_fermionic_layer=0
                )
                system_cfg_left.paramvec = paramvec_left
                system_cfg_right.paramvec = paramvec_right

                system_z2_2_2_left = system.Z2System2D(system_cfg_left)
                system_z2_2_2_right = system.Z2System2D(system_cfg_right)

                # This is a single layer construction, we always use layer 0 to test.
                deriv_ana = system_z2_2_2.grad_over_norm_vec[lay, uc_ind, ind]
                norm_left = system_z2_2_2_left.calculate_lognorm(incremental=True, all_factors=True)
                norm_right = system_z2_2_2_right.calculate_lognorm(incremental=True, all_factors=True)
                deriv_num = (norm_right - norm_left) / (2 * eps)

                self.assertAlmostEqual(deriv_ana, deriv_num, places=6)

    def test_grad_el_energy_1_layer(self):
        # This is comparison of the analytic derivative against the numeric derivative
        eps = 1e-5
        tr = 0.17
        yr = 0.35
        zr = 0.56
        ti = 0.10
        yi = 0.46
        zi = 0.41
        paramvec = [[tr, yr, zr, ti, yi, zi]]
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_Config(lat_2x2, 1.0, None, None, None, None, ncopy=1, num_fermionic_layer=0)
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(system_cfg)
        deriv_ana = system_z2_2_2.el_energy_op_grad_vec
        symbolvec = system_z2_2_2.symbolvec

        uc_ind = 0

        for ind in range(len(symbolvec)):
            with self.subTest(symbol=symbolvec[ind]):
                paramvec_left = np.copy(paramvec)
                paramvec_right = np.copy(paramvec)
                paramvec_left[0, ind] -= eps
                paramvec_right[0, ind] += eps
                system_cfg_left = system.Z2System2D_Config(
                    lat_2x2, 1.0, None, None, None, None, ncopy=1, num_fermionic_layer=0
                )
                system_cfg_right = system.Z2System2D_Config(
                    lat_2x2, 1.0, None, None, None, None, ncopy=1, num_fermionic_layer=0
                )
                system_cfg_left.paramvec = paramvec_left
                system_cfg_right.paramvec = paramvec_right

                system_z2_2_2_left = system.Z2System2D(system_cfg_left)
                system_z2_2_2_right = system.Z2System2D(system_cfg_right)

                val_left = system_z2_2_2_left.el_energy_op
                val_right = system_z2_2_2_right.el_energy_op
                deriv_num = (val_right - val_left) / (2 * eps)

                self.assertAlmostEqual(deriv_ana[0, uc_ind, ind], deriv_num, places=5)

    def test_grad_el_energy_2_layer(self):
        # This is comparison of the analytic derivative against the numeric derivative
        eps = 1e-5
        nlayer = 2
        paramvec = np.asarray([[0.17, 0.35, 0.56, 0.39, 0.42, 0.12], [0.3, 0.2, 0.8, 0.68, 0.32, 0.19]])
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_Config(
            lat_2x2,
            1.0,
            None,
            None,
            None,
            None,
            num_pg_layer=2,
            num_fermionic_layer=0,
            ncopy=1,
        )
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(system_cfg)
        deriv_ana = system_z2_2_2.el_energy_op_grad_vec
        symbolvec = self.system_z2_2_2_real.symbolvec

        uc_ind = 0

        for layerind in range(nlayer):
            for ind in range(len(symbolvec)):
                with self.subTest(symbol=symbolvec[ind], layerind=layerind):
                    paramvec_left = np.copy(paramvec)
                    paramvec_right = np.copy(paramvec)
                    paramvec_left[layerind, ind] -= eps
                    paramvec_right[layerind, ind] += eps
                    system_cfg_left = system.Z2System2D_Config(
                        lat_2x2,
                        1.0,
                        None,
                        None,
                        0,
                        None,
                        num_pg_layer=2,
                        num_fermionic_layer=0,
                        ncopy=1,
                    )
                    system_cfg_right = system.Z2System2D_Config(
                        lat_2x2,
                        1.0,
                        None,
                        None,
                        0,
                        None,
                        num_pg_layer=2,
                        num_fermionic_layer=0,
                        ncopy=1,
                    )
                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    system_z2_2_2_left = system.Z2System2D(system_cfg_left)
                    system_z2_2_2_right = system.Z2System2D(system_cfg_right)

                    val_left = system_z2_2_2_left.el_energy_op
                    val_right = system_z2_2_2_right.el_energy_op
                    deriv_num = (val_right - val_left) / (2 * eps)

                    self.assertAlmostEqual(deriv_ana[layerind, uc_ind, ind], deriv_num, places=5)

    def test_grad_el_energy_3_layer(self):
        # This is a comparison of the analytic derivative against the numeric derivative
        eps = 1e-5
        nlayer = 3
        paramvec = np.random.rand(3, 6)
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_Config(
            lat_2x2, 1.0, None, None, 0, None, num_pg_layer=3, num_fermionic_layer=0, ncopy=1
        )
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(system_cfg)
        deriv_ana = system_z2_2_2.el_energy_op_grad_vec
        symbolvec = self.system_z2_2_2_real.symbolvec

        uc_ind = 0

        for layerind in range(nlayer):
            for ind in range(len(symbolvec)):
                with self.subTest(symbol=symbolvec[ind], layerind=layerind):
                    paramvec_left = np.copy(paramvec)
                    paramvec_right = np.copy(paramvec)
                    paramvec_left[layerind, ind] -= eps
                    paramvec_right[layerind, ind] += eps
                    system_cfg_left = system.Z2System2D_Config(
                        lat_2x2,
                        1.0,
                        None,
                        None,
                        0,
                        None,
                        num_pg_layer=3,
                        num_fermionic_layer=0,
                        ncopy=1,
                    )
                    system_cfg_right = system.Z2System2D_Config(
                        lat_2x2,
                        1.0,
                        None,
                        None,
                        0,
                        None,
                        num_pg_layer=3,
                        num_fermionic_layer=0,
                        ncopy=1,
                    )
                    system_cfg_left.paramvec = paramvec_left
                    system_cfg_right.paramvec = paramvec_right

                    system_z2_2_2_left = system.Z2System2D(system_cfg_left)
                    system_z2_2_2_right = system.Z2System2D(system_cfg_right)

                    val_left = system_z2_2_2_left.el_energy_op
                    val_right = system_z2_2_2_right.el_energy_op
                    deriv_num = (val_right - val_left) / (2 * eps)

                    self.assertAlmostEqual(deriv_ana[layerind, uc_ind, ind], deriv_num, places=4)

    def test_el_energy_1_layer_single_eval(self):
        # Calculate the electric energy of an empty system.
        paramvec = [[0, 0, 0, 0, 0, 0]]
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_Config(
            lat_2x2,
            1.0,
            None,
            None,
            None,
            None,
            num_pg_layer=1,
            num_fermionic_layer=0,
            ncopy=1,
        )
        system_cfg.paramvec = paramvec
        system_z2_2_2 = system.Z2System2D(system_cfg)
        el_energy = system_z2_2_2.el_energy
        self.assertAlmostEqual(el_energy, 0.0)

    def test_el_energy_1_layer_mc(self):
        # Calculate the electric energy of an empty system.
        paramvec = [[0, 0, 0, 0, 0, 0]]
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_Config(
            lat_2x2, 1.0, 0.0, 0.0, 0.0, None, num_pg_layer=1, num_fermionic_layer=0, ncopy=1
        )
        system_cfg.paramvec = paramvec
        mc_config = MonteCarloEvaluatorConfig(warmup_steps=10, meas_steps=10, binsize=1)
        mc_mgr = EvaluatorManager(system.Z2System2D, system_cfg, mc_config, 0)
        mc_result = mc_mgr.simulate()
        el_energy = utils.get_obs_mean_df(mc_result, "el_energy")
        self.assertAlmostEqual(el_energy, 0.0)

    @skip("This test is not precise enough")
    def test_wilson_exact_real(self):
        t = 0.17
        y = 0.35
        z = 0.56
        paramvec = [[t, y, z, 0, 0, 0]]

        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2D_Config(
            paramvec, lat_2x2, 1.0, None, None, 0, None, ncopy=1, num_fermionic_layer=0
        )
        sys_exact = system.Z2System2D(system_cfg)
        sys_mc = system.Z2System2D(system_cfg)

        exact_cfg = ExactEvaluatorConfig()
        exact_ev = ExactEvaluator(exact_cfg, sys_exact)
        res = exact_ev.evaluate()

        mc_config = MonteCarloEvaluatorConfig(warmup_steps=10000, meas_steps=40000, binsize=1)
        mc = MonteCarloEvaluator(mc_config, sys_mc)
        mc.evaluate()

        self.assertAlmostEqual(res["wilson_00_11"], mc.get_obs_mean("wilson_00_11"), places=2)
