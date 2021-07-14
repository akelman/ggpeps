from unittest.case import SkipTest, skip
from exacteval import ExactEvaluator
import unittest
import numpy as np
import utils
import system
import lattice
from measurement import Measurement
from mc import MonteCarloEstimatorConfig, MonteCarloEstimator, MonteCarloManager
from minimizer import Minimizer
import gauge

def compare_array_elementwise(testcase,ref,res,print_vals=True):
    testcase.assertEqual(ref.shape,res.shape)
    if print_vals:
        for i in range(ref.shape[0]):
            for j in range(ref.shape[1]):
                if not np.isclose(ref[i, j] , res[i, j]):
                    print("{},{}: ref: {},res:{}".format(i,j,ref[i,j],res[i,j]))
    testcase.assertTrue(np.allclose(ref,res))

class TestGauge(unittest.TestCase):

    def setUp(self):
        self.gaugeZ3=gauge.ZNGauge(3)
        self.gaugeZ8=gauge.ZNGauge(8)

    def test_possble_gauges(self):
        poss_gauges_z3=self.gaugeZ3.get_possible_gauge_values()
        poss_gauges_z8=self.gaugeZ8.get_possible_gauge_values()
        self.assertEqual(len(poss_gauges_z3),3)
        self.assertEqual(len(poss_gauges_z8),8)

    def test_random_values(self):
        for gauge in [self.gaugeZ3, self.gaugeZ8]:
            poss_gauges = gauge.get_possible_gauge_values()
            for _ in range(100):
                val = gauge.get_random_gauge_value()
                self.assertTrue(np.any(np.isclose(val,poss_gauges)))


class TestUtils(unittest.TestCase):

    def generate_smat(self):
        N=10
        smat = utils.generate_smat(N)
        m, n = smat.shape
        self.assertEqual(m, n)
        self.assertEqual(m, N)
        res=smat@np.conjugate(np.transpose(smat))
        ref=2.*np.eye(N)
        self.assertTrue(np.allclose(ref,res))

class TestLattice(unittest.TestCase):

    def setUp(self):
        self.lat2d=lattice.Lattice2D(8,8)
        self.lat3d=lattice.Lattice3D(8,8,8)

    def test_ind2coord_2d(self):
        ref=(3,4)
        ind=self.lat2d.coord2ind(ref)
        coord=self.lat2d.ind2coord(ind)
        self.assertEqual(ref,coord)

    def test_ind2coord_dir_2d(self):
        coord_ref=(2,3)
        for dir_ref in [lattice.Direction.X, lattice.Direction.Y]:
            ind=self.lat2d.coord2ind_dir(coord_ref,dir_ref)
            coord,dir=self.lat2d.ind2coord_dir(ind)
            self.assertEqual(coord_ref,coord)
            self.assertEqual(dir_ref,dir)

    def test_wilson_loop_1x1(self):
        ref=[(((0,0),lattice.Direction.X),False),
            (((1,0),lattice.Direction.Y),False),
            (((0,1),lattice.Direction.X),True),
            (((0,0),lattice.Direction.Y),True)
            ]
        path=self.lat2d.generate_wilson_loop((0,0),(1,1),False)
        self.assertEqual(ref,path)

    def test_wilson_loop_2x1(self):
        ref=[(((0,0),lattice.Direction.X),False),
            (((1,0),lattice.Direction.X),False),
            (((2,0),lattice.Direction.Y),False),
            (((1,1),lattice.Direction.X),True),
            (((0,1),lattice.Direction.X),True),
            (((0,0),lattice.Direction.Y),True)
            ]
        path=self.lat2d.generate_wilson_loop((0,0),(2,1),False)
        self.assertEqual(ref,path)

    def test_wilson_loop_1x1_periodic(self):
        ref=[(((7,7),lattice.Direction.X),False),
            (((0,7),lattice.Direction.Y),False),
            (((7,0),lattice.Direction.X),True),
            (((7,7),lattice.Direction.Y),True)
            ]
        path=self.lat2d.generate_wilson_loop((7,7),(1,1),False)
        self.assertEqual(ref,path)

    def test_polyakov_loop_hor(self):
        ref=[(((0,0),lattice.Direction.X),False),
            (((1,0),lattice.Direction.X),False),
            (((2,0),lattice.Direction.X),False),
            (((3,0),lattice.Direction.X),False),
            (((4,0),lattice.Direction.X),False),
            (((5,0),lattice.Direction.X),False),
            (((6,0),lattice.Direction.X),False),
            (((7,0),lattice.Direction.X),False),
            ]
        path=self.lat2d.generate_polyakov_loop((0,0),lattice.Direction.X, use_indices=False)
        self.assertEqual(ref,path)

    def test_polyakov_loop_vert(self):
        ref=[(((0,0),lattice.Direction.Y),False),
            (((0,1),lattice.Direction.Y),False),
            (((0,2),lattice.Direction.Y),False),
            (((0,3),lattice.Direction.Y),False),
            (((0,4),lattice.Direction.Y),False),
            (((0,5),lattice.Direction.Y),False),
            (((0,6),lattice.Direction.Y),False),
            (((0,7),lattice.Direction.Y),False),
            ]
        path=self.lat2d.generate_polyakov_loop((0,0),lattice.Direction.Y, use_indices=False)
        self.assertEqual(ref,path)

    def test_2d_covering(self):
        nx=13
        ny=7
        lat=lattice.Lattice2D(nx,ny)
        linkvec=np.zeros(lat.nlinks)
        sitevec=np.zeros(lat.size)
        for x in range(nx):
            for y in range(ny):
                ind_site=lat.coord2ind((x,y))
                ind_link_x=lat.coord2ind_dir((x,y),lattice.Direction.X)
                ind_link_y=lat.coord2ind_dir((x,y),lattice.Direction.Y)
                linkvec[ind_link_x]=1
                linkvec[ind_link_y]=1
                sitevec[ind_site]=1
        self.assertEqual(np.sum(linkvec),lat.nlinks)
        self.assertEqual(np.sum(sitevec),lat.size)

    def test_ind2coord_3d(self):
        ref=(3,4,2)
        ind=self.lat3d.coord2ind(ref)
        coord=self.lat3d.ind2coord(ind)
        self.assertEqual(ref,coord)

    def test_ind2coord_dir_3d(self):
        coord_ref=(2,3,3)
        for dir_ref in lattice.Direction:
            ind=self.lat3d.coord2ind_dir(coord_ref,dir_ref)
            coord,dir=self.lat3d.ind2coord_dir(ind)
            self.assertEqual(coord_ref,coord)
            self.assertEqual(dir_ref,dir)

class TestPermutationBuilder2D(unittest.TestCase):

    def setUp(self):
        self.permbuilder_2x3=lattice.PermutationBuilderGMS2D(lattice.Lattice2D(2,3),1)
        self.permbuilder_3x2=lattice.PermutationBuilderGMS2D(lattice.Lattice2D(3,2),1)
        self.permbuilder_4x3=lattice.PermutationBuilderGMS2D(lattice.Lattice2D(4,3),1)

    def test_lr_permutation(self):
        self.assertEqual(self.permbuilder_2x3._perm_lr().shape,(8,16))
        self.assertEqual(self.permbuilder_3x2._perm_lr().shape,(12,24))
        self.assertEqual(self.permbuilder_4x3._perm_lr().shape,(16,32))

    def test_du_permutation(self):
        self.assertEqual(self.permbuilder_2x3._perm_du().shape,(12,40))
        self.assertEqual(self.permbuilder_3x2._perm_du().shape,(8,32))
        self.assertEqual(self.permbuilder_4x3._perm_du().shape,(12,72))

    def test_full_permutation(self):
        self.assertTrue(utils.is_permutation(self.permbuilder_2x3.perm()))
        self.assertTrue(utils.is_permutation(self.permbuilder_3x2.perm()))
        self.assertTrue(utils.is_permutation(self.permbuilder_4x3.perm()))

class TestZ2SystemMethods(unittest.TestCase):
    def setUp(self):
        lat=lattice.Lattice2D(2,2)
        paramdict={"t":0.3,"y":0.5,"z":0.8}
        cfg=system.Z2System2DConfig(paramdict,lat,0,0,0)
        self.system_z2_2_2=system.Z2System2D(cfg)

    def test_tmat_numeric(self):
        #Test numeric equivalent with Mathematica
        tmat=self.system_z2_2_2.tmat
        ref = np.array([[0., -0.3*1.j, 0.3*1.j, 0.3, -0.3],
                        [0.3*1.j, 0, 0.5*1.j, 0.8, 0.8*1.j],
                        [-0.3*1.j, -0.5*1.j, 0., -0.8*1.j, -0.8],
                        [-0.3, -0.8, 0.8*1.j, 0, -0.5],
                        [0.3, -0.8*1.j, 0.8, 0.5, 0]])
        self.assertTrue(np.allclose(tmat,ref))

    def test_tmat_antisymmetric(self):
        tmat=self.system_z2_2_2.tmat
        self.assertTrue(utils.is_antisymmetric(tmat))

    def test_gamma_dirac_covariance(self):
        gamma_dirac=self.system_z2_2_2.gamma_dirac
        m, n = gamma_dirac.shape
        res=gamma_dirac@np.transpose(np.conjugate(gamma_dirac))
        ref=0.25*np.eye(gamma_dirac.shape[0])
        self.assertTrue(np.allclose(ref,res))
        self.assertEqual(m, n)
        self.assertTrue(utils.is_antisymmetric(gamma_dirac))

    def test_gamma_dirac_numeric(self):
        # We are comparing to Mathematica.
        # This check is useful for the mode-ordering.
        # Independent of the order, gamma_maj is always a covariance matrix
        lat=lattice.Lattice2D(2,2)
        paramdict = {"y": 0.35, "z": 0.56, "t": 0.17}
        cfg = system.Z2System2DConfig(paramdict, lat, 0, 0, 0)
        system_z2_2_2 = system.Z2System2D(cfg)
        gamma_dirac=system_z2_2_2.gamma_dirac
        ref = np.asarray([[-1.73472*10**-18 - 1.73472*10**-18 * 1.j, -0.0753112 + 6.93889*10**-18 * 1.j, 0.0753112 - 6.93889*10**-18 * 1.j, -1.04083*10**-17 - 0.0753112 * 1.j, 0. + 0.0753112 * 1.j, -2.01965*10**-18 + 0.448788 * 1.j, 0.0421743 - 0.0685332 * 1.j, 0.0421743 - 0.0685332 * 1.j, 0.0421743 - 0.0685332 * 1.j, 0.0421743 - 0.0685332 * 1.j],
                          [0.0753112 + 0. * 1.j, 0. - 6.93889*10**-18 * 1.j, 0.124232 + 0.0821891 * 1.j, -0.0256841 - 0.304589 * 1.j, 0.304589 - 0.0256841 * 1.j, -0.0421743 -
                              0.0685332 * 1.j, -7.58942*10**-19 + 0.102576 * 1.j, 3.46945*10**-18 + 0.0128029 * 1.j, -0.0733832 - 0.0605802 * 1.j, 0.0733832 - 0.0605802 * 1.j],
                          [-0.0753112 + 0. * 1.j, -0.124232 - 0.0821891 * 1.j, -6.93889*10**-18 + 0. * 1.j, -0.304589 + 0.0256841 * 1.j, 0.0256841 + 0.304589 * 1.j, -0.0421743 -
                              0.0685332 * 1.j, -3.46945*10**-18 + 0.0128029 * 1.j, 3.79471*10**-19 + 0.102576 * 1.j, 0.0733832 - 0.0605802 * 1.j, -0.0733832 - 0.0605802 * 1.j],
                          [8.15433*10**-20 + 0.0753112 * 1.j, 0.0256841 + 0.304589 * 1.j, 0.304589 - 0.0256841 * 1.j, 6.93687*10**-18 + 4.33681*10**-18 * 1.j, -0.0821891 + 0.124232 * 1.j, -
                           0.0421743 - 0.0685332 * 1.j, 0.0733832 - 0.0605802 * 1.j, -0.0733832 - 0.0605802 * 1.j, -3.79858*10**-19 + 0.102576 * 1.j, 7.65456*10**-21 + 0.0128029 * 1.j],
                          [3.72404*10**-20 - 0.0753112 * 1.j, -0.304589 + 0.0256841 * 1.j, -0.0256841 - 0.304589 * 1.j, 0.0821891 - 0.124232 * 1.j, -6.93687*10**-18 - 4.33681*10**-18 *
                           1.j, -0.0421743 - 0.0685332 * 1.j, -0.0733832 - 0.0605802 * 1.j, 0.0733832 - 0.0605802 * 1.j, -2.44205*10**-21 + 0.0128029 * 1.j, -2.35267*10**-20 + 0.102576 * 1.j],
                          [-2.01965*10**-18 - 0.448788 * 1.j, 0.0421743 + 0.0685332 * 1.j, 0.0421743 + 0.0685332 * 1.j, 0.0421743 + 0.0685332 * 1.j, 0.0421743 + 0.0685332 * 1.j, -1.73472 *
                           10**-18 + 1.73472*10**-18 * 1.j, -0.0753112 - 6.93889*10**-18 * 1.j, 0.0753112 + 6.93889*10**-18 * 1.j, -1.04083*10**-17 + 0.0753112 * 1.j, 0. - 0.0753112 * 1.j],
                          [-0.0421743 + 0.0685332 * 1.j, -7.58942*10**-19 - 0.102576 * 1.j, 3.46945*10**-18 - 0.0128029 * 1.j, -0.0733832 + 0.0605802 * 1.j, 0.0733832 +
                           0.0605802 * 1.j, 0.0753112 + 0. * 1.j, 0. + 6.93889*10**-18 * 1.j, 0.124232 - 0.0821891 * 1.j, -0.0256841 + 0.304589 * 1.j, 0.304589 + 0.0256841 * 1.j],
                          [-0.0421743 + 0.0685332 * 1.j, -3.46945*10**-18 - 0.0128029 * 1.j, 3.79471*10**-19 - 0.102576 * 1.j, 0.0733832 + 0.0605802 * 1.j, -0.0733832 + 0.0605802 *
                           1.j, -0.0753112 + 0. * 1.j, -0.124232 + 0.0821891 * 1.j, -6.93889*10**-18 + 0. * 1.j, -0.304589 - 0.0256841 * 1.j, 0.0256841 - 0.304589 * 1.j],
                          [-0.0421743 + 0.0685332 * 1.j, 0.0733832 + 0.0605802 * 1.j, -0.0733832 + 0.0605802 * 1.j, -3.79858*10**-19 - 0.102576 * 1.j, 7.65456*10**-21 - 0.0128029 * 1.j,
                           8.15433*10**-20 - 0.0753112 * 1.j, 0.0256841 - 0.304589 * 1.j, 0.304589 + 0.0256841 * 1.j, 6.93687*10**-18 - 4.33681*10**-18 * 1.j, -0.0821891 - 0.124232 * 1.j],
                          [-0.0421743 + 0.0685332 * 1.j, -0.0733832 + 0.0605802 * 1.j, 0.0733832 + 0.0605802 * 1.j, -2.44205*10**-21 - 0.0128029 * 1.j, -2.35267*10**-20 - 0.102576 * 1.j, 3.72404*10**-20 + 0.0753112 * 1.j, -0.304589 - 0.0256841 * 1.j, -0.0256841 + 0.304589 * 1.j, 0.0821891 + 0.124232 * 1.j, -6.93687*10**-18 + 4.33681*10**-18 * 1.j]])
        compare_array_elementwise(self,np.real(ref),np.real(gamma_dirac))
        compare_array_elementwise(self,np.imag(ref),np.imag(gamma_dirac))

    def test_gamma_maj_covariance(self):
        gamma_maj=self.system_z2_2_2.gamma_maj
        m, n = gamma_maj.shape
        self.assertEqual(m, n)
        self.assertEqual(m, 10)
        self.assertTrue(np.allclose(np.imag(gamma_maj),0))
        self.assertTrue(utils.is_antisymmetric(gamma_maj))
        self.assertTrue(np.allclose(gamma_maj@gamma_maj,-np.eye(m)))
        self.assertTrue(np.allclose(gamma_maj@np.transpose(gamma_maj),np.eye(m)))

    def test_gamma_maj_numeric(self):
        # We are comparing to Mathematica.
        # This check is useful for the mode-ordering.
        # Independent of the order, gamma_maj is always a covariance matrix
        lat=lattice.Lattice2D(2,2)
        paramdict = {"y": 0.35, "z": 0.56, "t": 0.17}
        cfg = system.Z2System2DConfig(paramdict, lat, 0, 0, 0)
        system_z2_2_2 = system.Z2System2D(cfg)
        gamma_maj = system_z2_2_2.gamma_maj
        ref = np.asarray([[0, 0.8975767509856909, -0.06627386700925886, -0.137066406769149, 0.23497098303282687, -0.137066406769149, 0.08434855801178401, 0.013556018251893853, 0.08434855801178401, -0.2876888317901919],
                          [-0.8975767509856909, 0, 0.137066406769149, 0.23497098303282687, 0.137066406769149, -0.06627386700925886,
                           0.2876888317901919, 0.08434855801178401, -0.013556018251893853, 0.08434855801178401],
                          [0.06627386700925886, -0.137066406769149, 0, 0.2051526809871836, 0.2484631458705544, -
                              0.13877244593263055, -0.1981345076351612, 0.4880175511092357, 0.7559443427596008, -0.06979228401520404],
                          [0.137066406769149, -0.23497098303282687, -0.2051526809871836, 0, -0.18998407043978507, -
                           0.2484631458705544, 0.7303385305060236, -0.09539809626878132, 0.17252869538158389, -0.46241173885565845],
                          [-0.23497098303282687, -0.137066406769149, -0.2484631458705544, 0.18998407043978507, 0,
                           0.2051526809871836, -0.46241173885565845, -0.17252869538158389, -0.09539809626878132, -0.7303385305060236],
                          [0.137066406769149, 0.06627386700925886, 0.13877244593263055, 0.2484631458705544, -0.2051526809871836,
                           0, 0.06979228401520404, 0.7559443427596008, -0.4880175511092357, -0.1981345076351612],
                          [-0.08434855801178401, -0.2876888317901919, 0.1981345076351612, -0.7303385305060236, 0.46241173885565845, -
                           0.06979228401520404, 0, 0.2051526809871836, -0.16437825818620777, -0.22285733361697713],
                          [-0.013556018251893853, -0.08434855801178401, -0.4880175511092357, 0.09539809626878132,
                           0.17252869538158389, -0.755944342759601, -0.2051526809871836, 0, -0.2740689581241317, 0.16437825818620777],
                          [-0.08434855801178401, 0.013556018251893853, -0.755944342759601, -0.17252869538158389,
                           0.09539809626878132, 0.4880175511092357, 0.16437825818620777, 0.2740689581241317, 0, 0.2051526809871836],
                          [0.2876888317901919, -0.08434855801178401, 0.06979228401520404, 0.46241173885565845, 0.7303385305060236, 0.1981345076351612, 0.22285733361697713, -0.16437825818620777, -0.2051526809871836, 0]])

        compare_array_elementwise(self,ref,gamma_maj,print_vals=False)


    def test_gamma_maj_sys_covariance(self):
        gamma_maj=self.system_z2_2_2.gamma_maj_sys
        m, n = gamma_maj.shape
        self.assertEqual(m, n)
        self.assertTrue(np.allclose(np.imag(gamma_maj),0))
        self.assertTrue(utils.is_antisymmetric(gamma_maj))
        self.assertTrue(np.allclose(gamma_maj@gamma_maj,-np.eye(m)))
        self.assertTrue(np.allclose(gamma_maj@np.transpose(gamma_maj),np.eye(m)))

    def test_gamma_in_sys_covariance(self):
        gamma_in=self.system_z2_2_2.gamma_in_sys
        m, n = gamma_in.shape
        self.assertEqual(m, n)
        self.assertTrue(utils.is_antisymmetric(gamma_in))
        self.assertTrue(np.allclose(gamma_in@gamma_in,-np.eye(m)))
        self.assertTrue(np.allclose(gamma_in@np.transpose(gamma_in),np.eye(m)))

    def test_part_d_covmat(self):
        #If t=0, then mat_d must be a valid covariance matrix
        lat=lattice.Lattice2D(2,2)
        paramdict={"t":0.0,"y":0.5,"z":0.8}
        cfg=system.Z2System2DConfig(paramdict,lat,0,0,0)
        system_z2_2_2=system.Z2System2D(cfg)
        mat=system_z2_2_2.mat_d
        m, _ = mat.shape
        self.assertTrue(utils.is_antisymmetric(mat))
        self.assertTrue(np.allclose(mat@mat,-np.eye(m)))
        self.assertTrue(np.allclose(mat@np.transpose(mat),np.eye(m)))

    def test_derivative_y(self):

        #This is comparison with Mathematica
        ref = np.asarray([[0, 0.08258109830595964, 0.05343482831562092, -0.040109484641008664, -0.1894507549372015, -0.040109484641008664, -0.06800796331079029, -0.16155227626741986, -0.06800796331079029, 0.08133330698540256],
                          [-0.08258109830595964, 0, 0.040109484641008664, -0.1894507549372015, 0.040109484641008664,
                           0.05343482831562092, -0.08133330698540256, -0.06800796331079029, 0.16155227626741986, -0.06800796331079029],
                          [-0.05343482831562092, -0.040109484641008664, 0, -0.149983826896498, 0.8066611217161592, -
                           0.34826252828193405, -0.5416620988290691, -0.6347369627047637, -0.029060877939980495, -0.06401398593571395],
                          [0.040109484641008664, 0.1894507549372015, 0.14998382689649795, 0, -0.3069719791289543, -
                           0.8066611217161592, -0.008415603363490592, -0.043368711359223974, 0.5623073734055589, 0.6140916881282736],
                          [0.1894507549372015, -0.040109484641008664, -0.8066611217161592, 0.3069719791289543, 0, -
                           0.149983826896498, 0.6140916881282736, -0.562307373405559, -0.043368711359223974, 0.008415603363490565],
                          [0.040109484641008664, -0.05343482831562092, 0.3482625282819341, 0.8066611217161592, 0.14998382689649795,
                           0, 0.06401398593571395, -0.029060877939980495, 0.6347369627047635, -0.5416620988290691],
                          [0.06800796331079029, 0.08133330698540256, 0.5416620988290691, 0.008415603363490565, -
                           0.6140916881282736, -0.06401398593571395, 0, -0.149983826896498, -0.3276172537054442, -0.8273063962926491],
                          [0.16155227626741986, 0.06800796331079029, 0.6347369627047635, 0.043368711359223974, 0.5623073734055589,
                           0.02906087793998055, 0.14998382689649795, 0, -0.7860158471396694, 0.3276172537054442],
                          [0.06800796331079029, -0.16155227626741986, 0.02906087793998055, -0.562307373405559,
                           0.043368711359223974, -0.6347369627047637, 0.3276172537054442, 0.7860158471396694, 0, -0.149983826896498],
                          [-0.08133330698540256, 0.06800796331079029, 0.06401398593571395, -0.6140916881282736, -0.008415603363490592, 0.5416620988290691, 0.8273063962926492, -0.3276172537054442, 0.14998382689649795, 0]])
        lat = lattice.Lattice2D(2, 2)
        paramdict = {"y": 0.35, "z": 0.56, "t": 0.17}
        cfg = system.Z2System2DConfig(paramdict, lat, 0, 0, 0)
        system_z2_2_2 = system.Z2System2D(cfg)
        res = system_z2_2_2.compute_gamma_maj_deriv_y()
        self.assertTrue(np.allclose(ref, res))

    def test_derivative_z(self):
        #This is comparison with Mathematica
        ref = np.asarray([[0, 0.13340023572501172, 0.23694022460781514, 0.027898478669781623, -0.15541340987751348, 0.027898478669781623, 0.04076340736515084, -0.16827833857288266, 0.04076340736515084, 0.224075295912446],
                          [-0.13340023572501172, 0, -0.027898478669781623, -0.15541340987751348, -0.027898478669781623,
                           0.23694022460781514, -0.224075295912446, 0.04076340736515084, 0.16827833857288266, 0.04076340736515084],
                          [-0.23694022460781514, 0.027898478669781623, 0, -1.4457765039721213, -0.643152566068254, -
                           0.1821533692352031, 0.17445048782141942, 0.5296884884008861, 0.3045933801892958, 0.050644620390171174],
                          [-0.027898478669781623, 0.15541340987751348, 1.4457765039721213, 0, -0.1154532513726973,
                           0.643152566068254, 0.3379434391205487, 0.08399467932142413, -0.1411004288901665, -0.5630385473321391],
                          [0.15541340987751348, 0.027898478669781623, 0.643152566068254, 0.1154532513726973, 0, -
                           1.4457765039721213, -0.5630385473321391, 0.14110042889016644, 0.08399467932142413, -0.3379434391205487],
                          [-0.027898478669781623, -0.23694022460781514, 0.1821533692352031, -0.643152566068254, 1.4457765039721213,
                           0, -0.050644620390171174, 0.3045933801892958, -0.5296884884008861, 0.17445048782141942],
                          [-0.04076340736515084, 0.224075295912446, -0.17445048782141942, -0.3379434391205487, 0.5630385473321391,
                           0.050644620390171174, 0, -1.4457765039721213, -0.1488033103039502, 0.609802507137001],
                          [0.16827833857288266, -0.04076340736515084, -0.5296884884008861, -0.08399467932142413, -
                           0.1411004288901665, -0.3045933801892954, 1.4457765039721213, 0, 0.676502624999507, 0.1488033103039502],
                          [-0.04076340736515084, -0.16827833857288266, -0.3045933801892954, 0.14110042889016644, -
                           0.08399467932142413, 0.5296884884008861, 0.1488033103039502, -0.676502624999507, 0, -1.4457765039721213],
                          [-0.224075295912446, -0.04076340736515084, -0.050644620390171174, 0.5630385473321391, 0.3379434391205487, -0.17445048782141942, -0.6098025071370011, -0.1488033103039502, 1.4457765039721213, 0]])

        lat = lattice.Lattice2D(2, 2)
        paramdict = {"y": 0.35, "z": 0.56, "t": 0.17}
        cfg = system.Z2System2DConfig(paramdict, lat, 0, 0, 0)
        system_z2_2_2 = system.Z2System2D(cfg)
        res = system_z2_2_2.compute_gamma_maj_deriv_z()
        compare_array_elementwise(self,ref,res)

    def test_derivative_t(self):
        #This is comparison with Mathematica
        ref = np.array([[0, -1.1432704475880655, -0.3499169542672254, -0.7236918826890345, 1.2406146560383449, -0.7236918826890345, 0.4453488508855597, 0.07157392246375063, 0.4453488508855597, -1.5189576878418198],
                       [1.1432704475880655, 0, 0.7236918826890345, 1.2406146560383449, 0.7236918826890345, -
                           0.3499169542672254, 1.5189576878418198, 0.4453488508855597, -0.07157392246375063, 0.4453488508855597],
                       [0.3499169542672254, -0.7236918826890345, 0, 0.01891037755325542, -0.24288558325282242,
                           0.13634956066451026, -0.18016238318203856, -0.04381282251752838, -0.06272320007078382, 0.19907276073529406],
                       [0.7236918826890345, -1.2406146560383449, -0.01891037755325542, 0, -0.4352856631295225,
                        0.24288558325282242, -0.3485408119678002, -0.08674485116172226, -0.10565522871497779, 0.32963043441454465],
                       [-1.2406146560383449, -0.7236918826890345, 0.24288558325282242, 0.4352856631295225, 0,
                        0.01891037755325542, 0.32963043441454465, 0.10565522871497779, -0.08674485116172226, 0.3485408119678002],
                       [0.7236918826890345, 0.3499169542672254, -0.13634956066451026, -0.24288558325282242, -0.01891037755325542,
                        0, -0.19907276073529406, -0.06272320007078382, 0.04381282251752838, -0.18016238318203856],
                       [-0.4453488508855597, -1.5189576878418198, 0.18016238318203856, 0.3485408119678002, -0.32963043441454476,
                        0.19907276073529406, 0, 0.01891037755325542, -0.14946805123250612, 0.5287031951498388],
                       [-0.07157392246375063, -0.4453488508855597, 0.04381282251752838, 0.08674485116172226, -0.10565522871497779,
                        0.06272320007078391, -0.01891037755325542, 0, -0.04293202864419393, 0.14946805123250612],
                       [-0.4453488508855597, 0.07157392246375063, 0.06272320007078391, 0.10565522871497779, 0.08674485116172226, -
                        0.04381282251752838, 0.14946805123250612, 0.04293202864419393, 0, 0.01891037755325542],
                       [1.5189576878418198, -0.4453488508855597, -0.19907276073529406, -0.32963043441454476, -0.3485408119678002, 0.18016238318203856, -0.5287031951498388, -0.14946805123250612, -0.01891037755325542, 0]])

        lat = lattice.Lattice2D(2, 2)
        paramdict = {"y": 0.35, "z": 0.56, "t": 0.17}
        cfg = system.Z2System2DConfig(paramdict, lat, 0, 0, 0)
        system_z2_2_2 = system.Z2System2D(cfg)
        res = system_z2_2_2.compute_gamma_maj_deriv_t()
        compare_array_elementwise(self,ref,res)

    def test_derivative_t_numeric(self):
        #This is comparison of the analytic derivative against the numeric derivative
        eps=1e-5
        lat = lattice.Lattice2D(2, 2)
        paramdict = {"y": 0.35, "z": 0.56, "t": 0.17}
        paramdict_left = {"y": 0.35, "z": 0.56, "t": 0.17-eps}
        paramdict_right = {"y": 0.35, "z": 0.56, "t": 0.17+eps}

        cfg = system.Z2System2DConfig(paramdict, lat, 0, 0, 0)
        cfg_left = system.Z2System2DConfig(paramdict_left, lat, 0, 0, 0)
        cfg_right = system.Z2System2DConfig(paramdict_right, lat, 0, 0, 0)

        system_z2_2_2 = system.Z2System2D(cfg)
        system_z2_2_2_left= system.Z2System2D(cfg_left)
        system_z2_2_2_right = system.Z2System2D(cfg_right)

        deriv_ana=system_z2_2_2.compute_gamma_maj_deriv_t()
        gamma_left=system_z2_2_2_left.gamma_maj
        gamma_right=system_z2_2_2_right.gamma_maj
        deriv_num=(gamma_right-gamma_left)/(2*eps)

        compare_array_elementwise(self,deriv_num,deriv_ana,print_vals=True)


    def test_derivative_y_numeric_pure_gauge(self):
        #This is comparison of the analytic derivative against the numeric derivative
        eps=1e-5
        lat = lattice.Lattice2D(2, 2)
        paramdict = {"y": 0.35, "z": 0.56, "t": 0.0}
        paramdict_left = {"y": 0.35-eps, "z": 0.56, "t": 0.0}
        paramdict_right = {"y": 0.35+eps, "z": 0.56, "t": 0.0}

        cfg = system.Z2System2DConfig(paramdict, lat, 0, 0, 0)
        cfg_left = system.Z2System2DConfig(paramdict_left, lat, 0, 0, 0)
        cfg_right = system.Z2System2DConfig(paramdict_right, lat, 0, 0, 0)

        system_z2_2_2 = system.Z2System2D(cfg)
        system_z2_2_2_left= system.Z2System2D(cfg_left)
        system_z2_2_2_right = system.Z2System2D(cfg_right)

        deriv_ana=system_z2_2_2.compute_gamma_maj_deriv_y()
        gamma_left=system_z2_2_2_left.gamma_maj
        gamma_right=system_z2_2_2_right.gamma_maj
        deriv_num=(gamma_right-gamma_left)/(2*eps)

        compare_array_elementwise(self,deriv_num,deriv_ana,print_vals=False)

    def test_derivative_z_numeric_pure_gauge(self):
        #This is comparison of the analytic derivative against the numeric derivative
        eps=1e-5
        lat = lattice.Lattice2D(2, 2)
        paramdict = {"y": 0.35, "z": 0.56, "t": 0.0}
        paramdict_left = {"y": 0.35, "z": 0.56-eps, "t": 0.0}
        paramdict_right = {"y": 0.35, "z": 0.56+eps, "t": 0.0}

        cfg = system.Z2System2DConfig(paramdict, lat, 0, 0, 0)
        cfg_left = system.Z2System2DConfig(paramdict_left, lat, 0, 0, 0)
        cfg_right = system.Z2System2DConfig(paramdict_right, lat, 0, 0, 0)

        system_z2_2_2 = system.Z2System2D(cfg)
        system_z2_2_2_left= system.Z2System2D(cfg_left)
        system_z2_2_2_right = system.Z2System2D(cfg_right)

        deriv_ana=system_z2_2_2.compute_gamma_maj_deriv_z()
        gamma_left=system_z2_2_2_left.gamma_maj
        gamma_right=system_z2_2_2_right.gamma_maj
        deriv_num=(gamma_right-gamma_left)/(2*eps)

        compare_array_elementwise(self,deriv_num,deriv_ana,print_vals=True)

    def test_derivative_y_sys(self):
        lat = lattice.Lattice2D(2, 2)
        paramdict = {"y": 0.35, "z": 0.56, "t": 0.17}
        cfg = system.Z2System2DConfig(paramdict, lat, 0, 0, 0)
        system_z2_2_2 = system.Z2System2D(cfg)
        res = system_z2_2_2.gamma_maj_sys_deriv("y")
        self.assertTrue(utils.is_antisymmetric(res))

    def test_derivative_z_sys(self):
        lat = lattice.Lattice2D(2, 2)
        paramdict = {"y": 0.35, "z": 0.56, "t": 0.17}
        cfg = system.Z2System2DConfig(paramdict, lat, 0, 0, 0)
        system_z2_2_2 = system.Z2System2D(cfg)
        res = system_z2_2_2.gamma_maj_sys_deriv("z")
        self.assertTrue(utils.is_antisymmetric(res))

    def test_derivative_t_sys(self):
        lat = lattice.Lattice2D(2, 2)
        paramdict = {"y": 0.35, "z": 0.56, "t": 0.17}
        cfg = system.Z2System2DConfig(paramdict, lat, 0, 0, 0)
        system_z2_2_2 = system.Z2System2D(cfg)
        res = system_z2_2_2.gamma_maj_sys_deriv("t")
        self.assertTrue(utils.is_antisymmetric(res))

    def test_derivative_gamma_sys_finite_diff(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # There is no sampling involved here. The gauge field is 0.
        eps=1e-5
        lat = lattice.Lattice2D(2, 2)
        t=0.2
        y=0.3
        z=0.8
        paramdict = {"y": y, "z": z, "t": t}
        param_namevec=["t","y","z"]
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2DConfig(paramdict, lat_2x2, 1.0, None, None)
        for ind in range(3):
            with self.subTest(ind=ind):
                paramvec=system_cfg.paramvec
                paramvec_left=np.copy(paramvec)
                paramvec_right=np.copy(paramvec)
                paramvec_left[ind]-=eps
                paramvec_right[ind]+=eps
                system_cfg_left = system.Z2System2DConfig(paramvec_left, lat_2x2, 1.0,
                                                        None, None)
                system_cfg_right = system.Z2System2DConfig(paramvec_right, lat_2x2,
                                                        1.0, None, None)

                system_z2_2_2 = system.Z2System2D(system_cfg)
                system_z2_2_2_left= system.Z2System2D(system_cfg_left)
                system_z2_2_2_right = system.Z2System2D(system_cfg_right)

                deriv_maj_sys=system_z2_2_2.gamma_maj_sys_deriv(param_namevec[ind])
                deriv_maj_sys_left=system_z2_2_2_left.gamma_maj_sys
                deriv_maj_sys_right=system_z2_2_2_right.gamma_maj_sys

                deriv_maj_sys_num=(deriv_maj_sys_right-deriv_maj_sys_left)/(2*eps)

                self.assertTrue(np.allclose(deriv_maj_sys_num, deriv_maj_sys))
        
    def test_norm_minimal(self):
        # This update is a nullop since we initialize the gauge-field with 0
        zeroarr = np.zeros((1, 1))
        #The factor of 2 compensates for the
        logdet_inc = 2*self.system_z2_2_2.calculate_lognorm_inc(
            0, zeroarr, all_factors=False)
        # This is equivalent to
        #logdet_inc = self.system_z2_2_2.incdet.det()
        diff = self.system_z2_2_2.mat_d_inv - self.system_z2_2_2.gamma_in_sys
        sign, logdet = np.linalg.slogdet(diff)
        self.assertGreater(sign,0)
        self.assertAlmostEqual(logdet_inc, logdet)

    def test_norm_incremental(self):
        # Test that the incremental update is equivalent to the re-calculation of the norm
        # This update is a nullop since we initialize the gauge-field with 0
        zeroarr = np.zeros((1, 1))
        weight_inc = self.system_z2_2_2.calculate_lognorm_inc(0,
                                                              zeroarr,
                                                              all_factors=True)
        weight_recalc = self.system_z2_2_2.calculate_lognorm(all_factors=True)
        self.assertAlmostEqual(weight_inc, weight_recalc)

    def test_norm_incremental_update(self):
        # Test that the incremental update is equivalent to the re-calculation of the norm
        ind = 0
        theta = np.pi
        weight_inc = self.system_z2_2_2.calculate_weight_attempt(
            ind, theta, all_factors=True)
        self.system_z2_2_2.update_gauge_ind(ind, theta)
        weight_recalc = self.system_z2_2_2.calculate_lognorm(all_factors=True)
        self.assertAlmostEqual(weight_inc, weight_recalc)

    def test_grad_over_norm_pure_gauge(self):
        # This is comparison of the analytic derivative against the numeric derivative
        # There is no sampling involved here. The gauge field is 0.
        eps=1e-5
        lat = lattice.Lattice2D(2, 2)
        t=0.0
        y=0.1
        z=0.2
        paramdict = {"y": y, "z": z, "t": t}
        paramdict_left = {"y": y-eps, "z": z, "t": t}
        paramdict_right = {"y": y+eps, "z": z, "t": t}

        cfg = system.Z2System2DConfig(paramdict, lat, 0, 0, 0)
        cfg_left = system.Z2System2DConfig(paramdict_left, lat, 0, 0, 0)
        cfg_right = system.Z2System2DConfig(paramdict_right, lat, 0, 0, 0)

        system_z2_2_2 = system.Z2System2D(cfg)
        system_z2_2_2_left= system.Z2System2D(cfg_left)
        system_z2_2_2_right = system.Z2System2D(cfg_right)

        # We are using here that the gradient of the d/dx log(f(x)) is [d/dx f(x)]/f(x).
        # Thus, the d/dx log(norm(x))= [d/dx norm(x)]/ norm(x) which is exactly the function grad_over_norm
        deriv_ana = system_z2_2_2.compute_grad_over_norm("y")
        lognorm_left = system_z2_2_2_left.calculate_lognorm(all_factors=True)
        lognorm_right = system_z2_2_2_right.calculate_lognorm(all_factors=True)
        deriv_num = (lognorm_right - lognorm_left) / (2 * eps)
        #print("Analytical",deriv_ana)
        #print("Numerical",deriv_num)
        self.assertAlmostEqual(deriv_ana, deriv_num)

    def test_grad_over_norm(self):
        #This is comparison of the analytic derivative against the numeric derivative
        eps=1e-5
        lat = lattice.Lattice2D(2, 2)
        t=0.17
        y=0.35
        z=0.56
        paramdict = {"y": y, "z": z, "t": t}
        param_namevec=["t","y","z"]
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2DConfig(paramdict, lat_2x2, 1.0, None, None)
        for ind in range(3):
            with self.subTest(ind=ind):
                paramvec=system_cfg.paramvec
                paramvec_left=np.copy(paramvec)
                paramvec_right=np.copy(paramvec)
                paramvec_left[ind]-=eps
                paramvec_right[ind]+=eps
                system_cfg_left = system.Z2System2DConfig(paramvec_left, lat_2x2, 1.0,
                                                        None, None)
                system_cfg_right = system.Z2System2DConfig(paramvec_right, lat_2x2,
                                                        1.0, None, None)

                cfg = system.Z2System2DConfig(paramdict, lat, 0, 0, 0)

                system_z2_2_2 = system.Z2System2D(cfg)
                system_z2_2_2_left= system.Z2System2D(system_cfg_left)
                system_z2_2_2_right = system.Z2System2D(system_cfg_right)

                deriv_ana = system_z2_2_2.compute_grad_over_norm(param_namevec[ind]) 
                norm_left = system_z2_2_2_left.calculate_lognorm(all_factors=True)
                norm_right = system_z2_2_2_right.calculate_lognorm(all_factors=True)
                deriv_num = (norm_right - norm_left) / (2 * eps)

                self.assertAlmostEqual(deriv_ana, deriv_num)

    @skip("This test is not precise enough")
    def test_wilson_exact(self):
        t=0.17
        y=0.35
        z=0.56
        paramdict = {"y": y, "z": z, "t": t}

        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2DConfig(paramdict, lat_2x2, 1.0, None,
                                            None)
        sys_exact = system.Z2System2D(system_cfg)
        sys_mc = system.Z2System2D(system_cfg)

        exact_ev=ExactEvaluator(sys_exact)
        res=exact_ev.evaluate()

        mc_config=MonteCarloEstimatorConfig()
        mc_config.binsize=1
        mc_config.meas_steps=40000
        mc_config.warmup_steps=10000
        mc=MonteCarloEstimator(mc_config,sys_mc)
        mc.simulate()

        self.assertAlmostEqual(res["wilson_00_11"], mc.get_obs_mean("wilson_00_11"),places=2)

class TestU1MultilayerSystemMethods(unittest.TestCase):

    def setUp(self):
        lat=lattice.Lattice2D(2,2)
        paramdict={"t":[0.3,0.4],"y":[0.5,0.2],"z":[0.8,0.3]}
        cfg=system.U1MultilayerSystem2DConfig(paramdict,lat)
        self.system_u1_2_2=system.U1MultilayerSystem2D(cfg)

    def test_tmat_numeric(self):
        #Test numeric equivalent with Mathematica
        tmat=self.system_u1_2_2.tmat
        pass

    def test_tmat_antisymmetric(self):
        tmat=self.system_u1_2_2.tmat
        ncopy,m,n=tmat.shape
        self.assertEqual(m,5)
        self.assertEqual(n,4)
        #for ind in range(ncopy):
        #self.assertTrue(utils.is_antisymmetric(tmat[ind,1:,:]))

#    def test_gamma_dirac_covariance(self):
#        gamma_dirac=self.system_u1_2_2.gamma_dirac
#        m, n = gamma_dirac.shape
#        res=gamma_dirac@np.transpose(np.conjugate(gamma_dirac))
#        ref=0.25*np.eye(gamma_dirac.shape[0])
#        self.assertTrue(np.allclose(ref,res))
#        self.assertEqual(m, n)
#        self.assertTrue(utils.is_antisymmetric(gamma_dirac))
#
#    def test_gamma_maj_covariance(self):
#        gamma_maj=self.system_u1_2_2.gamma_maj
#        m, n = gamma_maj.shape
#        self.assertEqual(m, n)
#        self.assertTrue(utils.is_antisymmetric(gamma_maj))
#        self.assertTrue(np.allclose(gamma_maj@gamma_maj,-np.eye(m)))
#        self.assertTrue(np.allclose(gamma_maj@np.transpose(gamma_maj),np.eye(m)))
#
#    def test_gamma_maj_sys_covariance(self):
#        gamma_maj=self.system_u1_2_2.gamma_maj_sys
#        m, n = gamma_maj.shape
#        self.assertEqual(m, n)
#        self.assertTrue(utils.is_antisymmetric(gamma_maj))
#        self.assertTrue(np.allclose(gamma_maj@gamma_maj,-np.eye(m)))
#        self.assertTrue(np.allclose(gamma_maj@np.transpose(gamma_maj),np.eye(m)))
#
#    def test_gamma_in_sys_covariance(self):
#        gamma_in=self.system_u1_2_2.gamma_in_sys
#        m, n = gamma_in.shape
#        self.assertEqual(m, n)
#        self.assertTrue(utils.is_antisymmetric(gamma_in))
#        self.assertTrue(np.allclose(gamma_in@gamma_in,-np.eye(m)))
#        self.assertTrue(np.allclose(gamma_in@np.transpose(gamma_in),np.eye(m)))

class TestMinimizerZ2(unittest.TestCase):

    def test_derivative_mag_energy_exact(self):
        eps = 1e-5
        paramdict = {"t": 0.0, "y": 0.5, "z": 0.5}
        for ind in range(3):
            with self.subTest(ind=ind):
                lat_2x2 = lattice.Lattice2D(2, 2)
                system_cfg = system.Z2System2DConfig(paramdict, lat_2x2, 1.0, None,
                                                    None)
                paramvec=system_cfg.paramvec
                paramvec_left=np.copy(paramvec)
                paramvec_right=np.copy(paramvec)
                paramvec_left[ind]-=eps
                paramvec_right[ind]+=eps
                system_cfg_left = system.Z2System2DConfig(paramvec_left, lat_2x2, 1.0,
                                                        None, None)
                system_cfg_right = system.Z2System2DConfig(paramvec_right, lat_2x2,
                                                        1.0, None, None)

                sys = system.Z2System2D(system_cfg)
                sys_left = system.Z2System2D(system_cfg_left)
                sys_right = system.Z2System2D(system_cfg_right)

                exact_ev=ExactEvaluator(sys)
                exact_ev_left=ExactEvaluator(sys_left)
                exact_ev_right=ExactEvaluator(sys_right)

                res=exact_ev.evaluate()
                res_left=exact_ev_left.evaluate()
                res_right=exact_ev_right.evaluate()

                mag_energy_deriv_num = (res_right["mag_energy"] - res_left["mag_energy"]) / (2 * eps)
                mag_energy_deriv_ana = res["mag_energy_grad"][ind]

                self.assertAlmostEqual(mag_energy_deriv_num, mag_energy_deriv_ana)

    def test_derivative_el_energy_exact(self):
        eps = 1e-5
        paramdict = {"t": 0.2, "y": 0.5, "z": 0.5}
        for ind in range(3):
            with self.subTest(ind=ind):
                lat_2x2 = lattice.Lattice2D(2, 2)
                system_cfg = system.Z2System2DConfig(paramdict, lat_2x2, 1.0, None,
                                                    None)
                paramvec=system_cfg.paramvec
                paramvec_left=np.copy(paramvec)
                paramvec_right=np.copy(paramvec)
                paramvec_left[ind]-=eps
                paramvec_right[ind]+=eps
                system_cfg_left = system.Z2System2DConfig(paramvec_left, lat_2x2, 1.0,
                                                        None, None)
                system_cfg_right = system.Z2System2DConfig(paramvec_right, lat_2x2,
                                                        1.0, None, None)

                sys = system.Z2System2D(system_cfg)
                sys_left = system.Z2System2D(system_cfg_left)
                sys_right = system.Z2System2D(system_cfg_right)

                exact_ev=ExactEvaluator(sys)
                exact_ev_left=ExactEvaluator(sys_left)
                exact_ev_right=ExactEvaluator(sys_right)

                res=exact_ev.evaluate()
                res_left=exact_ev_left.evaluate()
                res_right=exact_ev_right.evaluate()

                el_energy_deriv_num = (res_right["el_energy"] - res_left["el_energy"]) / (2 * eps)
                el_energy_deriv_ana = res["el_energy_grad"][ind]

                self.assertAlmostEqual(el_energy_deriv_num, el_energy_deriv_ana)

    @skip("Too long")
    def test_derivative_mag_energy_y(self):
        eps = 1e-4
        paramdict = {"t": 0.1, "y": 0.3, "z": 1.4}
        paramdict_left = {"t": 0.1, "y": 0.3-eps, "z": 1.4}
        paramdict_right = {"t": 0.1, "y": 0.3+eps, "z": 1.4}
        lat_2x2 = lattice.Lattice2D(2, 2)
        system_cfg = system.Z2System2DConfig(paramdict, lat_2x2, 1.0, None,
                                             None)
        system_cfg_left = system.Z2System2DConfig(paramdict_left, lat_2x2, 1.0,
                                                  None, None)
        system_cfg_right = system.Z2System2DConfig(paramdict_right, lat_2x2,
                                                   1.0, None, None)
        sys_left = system.Z2System2D(system_cfg_left)
        sys_right = system.Z2System2D(system_cfg_right)

        mc_config = MonteCarloEstimatorConfig()
        mc_config.warmup_steps = 1000
        mc_config.meas_steps = 10000
        mc_config.binsize = 1

        mc_mgr = MonteCarloManager(mc_config, system.Z2System2D, system_cfg, 0)
        minimizer = Minimizer(mc_mgr)
        mc_left = MonteCarloEstimator(mc_config, sys_left)
        mc_right = MonteCarloEstimator(mc_config, sys_right)

        minimizer.last_mcresult=minimizer.mc_mgr.simulate()
        mc_left.simulate()
        mc_right.simulate()

        mag_energy_deriv = minimizer.energy_gradient(minimizer.last_mcresult)
        mag_energy_left = mc_left.get_obs_mean("mag_energy")
        mag_energy_right = mc_right.get_obs_mean("mag_energy")

        mag_energy_deriv_num = (mag_energy_right - mag_energy_left) / (2 * eps)

        self.assertAlmostEqual(mag_energy_deriv[1],mag_energy_deriv_num)


class TestMeasurements(unittest.TestCase):
    def test_add(self):
        meas1=Measurement("meas1",1)
        meas2=Measurement("meas2",2)
        for i in range(10):
            meas1.append(1)
            meas2.append(2)
        self.assertEqual(meas1.mean(),1)
        self.assertEqual(meas1.var(),0)
        self.assertEqual(meas1.std(),0)
        self.assertEqual(len(meas1),10)
        self.assertEqual(meas2.mean(),2)
        self.assertEqual(meas2.var(),0)
        self.assertEqual(meas2.std(),0)
        self.assertEqual(len(meas2),5)

    def test_random_scalar(self):
        meas1=Measurement("meas1",1)
        meas2=Measurement("meas2",2)
        for _ in range(1000):
            rnd=np.random.rand()
            meas1.append(rnd)
            meas2.append(rnd)
        self.assertAlmostEqual(meas1.mean(),meas2.mean())

    def test_random_array(self):
        meas1=Measurement("meas1",1)
        meas2=Measurement("meas2",2)
        for _ in range(1000):
            rnd=np.random.rand(2,2)
            meas1.append(rnd)
            meas2.append(rnd)
        self.assertTrue(np.allclose(meas1.mean(),meas2.mean()))

    def test_mul_meas(self):
        meas1=Measurement("meas1",1)
        meas2=Measurement("meas2",1)
        for i in range(10):
            meas1.append(1.5)
            meas2.append(2)
        meas3=meas1*meas2
        for i in range(10):
            self.assertAlmostEqual(3,meas3.datavec[i])

    def test_sub_meas(self):
        meas1=Measurement("meas1",1)
        meas2=Measurement("meas2",1)
        for i in range(10):
            meas1.append(1)
            meas2.append(2)
        meas3 = meas1 - meas2
        for i in range(10):
            self.assertAlmostEqual(-1,meas3.datavec[i])

    def test_add_meas(self):
        meas1=Measurement("meas1",1)
        meas2=Measurement("meas2",1)
        for i in range(10):
            meas1.append(1)
            meas2.append(2)
        meas3 = meas1 + meas2
        for i in range(10):
            self.assertAlmostEqual(3,meas3.datavec[i])


# ======================= WoodburyInverter Test =========================================
class TestWoodburyInverter(unittest.TestCase):
    def setUp(self):
        self.n=10
        self.ident=np.eye(self.n)
        self.wi=utils.WoodburyInverter(self.ident)

    def test_identity(self):
        inv_wb=self.wi.update(self.ident,self.ident,self.ident)
        inv=np.linalg.inv(2*self.ident)
        self.assertTrue(np.allclose(inv,inv_wb))

    def test_identity_incr(self):
        for _ in range(10):
            update=0.1*self.ident
            self.wi.update(self.ident,update,self.ident)
        inv_wb=self.wi.inv()
        inv=np.linalg.inv(2*self.ident)
        self.assertTrue(np.allclose(inv,inv_wb))

    def test_random_incr(self):
        mat=np.random.rand(self.n,self.n)
        wi=utils.WoodburyInverter(mat)
        for _ in range(100):
            incr=np.random.rand(self.n,self.n)
            wi.update(self.ident,incr,self.ident)
            mat+=incr
        inv_wb=wi.inv()
        inv=np.linalg.inv(mat)
        self.assertTrue(np.allclose(inv,inv_wb))


    def test_pos_update(self):
        n=10
        n_up=2
        mat=np.random.rand(n,n)
        update_mat=np.random.rand(n_up,n_up)
        padval=n-n_up
        update_padded=np.pad(update_mat,[(0,padval),(0,padval)],'constant',constant_values=(0,0))
        wi=utils.WoodburyInverter(mat)
        inv_wb=wi.update_index(update_mat,0,0)
        inv=np.linalg.inv(mat+update_padded)
        self.assertTrue(np.allclose(inv,inv_wb))

    def test_zero_incr(self):
        mat=np.random.rand(self.n,self.n)
        wi=utils.WoodburyInverter(mat)
        zero=np.zeros((self.n,self.n))
        inv_wb=wi.update(zero,0,0)
        inv=np.linalg.inv(mat)
        self.assertTrue(np.allclose(inv,inv_wb))

# ======================= IncDeterminant Test =========================================
class TestIncDeterminant(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.ident = np.eye(self.n)
        self.incdet = utils.IncDeterminant(self.ident)

    def test_identity(self):
        detval = self.incdet.update(self.ident, self.ident, self.ident, self.ident)
        detval_direct = np.linalg.det(2*self.ident)
        self.assertAlmostEqual(detval, detval_direct)

    def test_identity_incr(self):
        track = np.copy(self.ident)
        for _ in range(10):
            self.incdet.update(np.linalg.inv(track), self.ident, 0.1*self.ident, self.ident)
            track += 0.1*self.ident
        detval = self.incdet.det()
        detval_direct = np.linalg.det(2*self.ident)
        self.assertAlmostEqual(detval, detval_direct)

    def test_random_incr(self):
        mat = np.random.rand(self.n, self.n)
        incdet = utils.IncDeterminant(mat)
        for _ in range(100):
            incr = np.random.rand(self.n, self.n)
            incdet.update(np.linalg.inv(mat),self.ident,incr,self.ident);
            mat+=incr
        detval =incdet.det()
        detval_direct =np.linalg.det(mat)
        self.assertAlmostEqual((detval-detval_direct)/detval_direct,0)

    def test_zero_incr(self):
        mat = np.random.rand(self.n, self.n)
        zero=np.zeros((self.n,self.n))
        incdet = utils.IncDeterminant(mat)
        incdet.update(np.linalg.inv(mat),self.ident,zero,self.ident)
        detval = incdet.det()
        detval_direct = np.linalg.det(mat)
        self.assertAlmostEqual(detval,detval_direct)

# ======================= IncLogAbsDeterminant Test =========================================


def generate_pos_def_matrix(n):
    rand = np.random.rand(n, n)
    rand = 0.5 * (rand + np.transpose(rand))
    rand += n * np.eye(n)
    return rand


class TestIncLogDeterminant(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.ident = np.eye(self.n)
        self.incdet = utils.IncLogAbsDeterminant(self.ident)

    def test_identity(self):
        detval = self.incdet.update(
            self.ident, self.ident, self.ident, self.ident)
        _, detval_direct = np.linalg.slogdet(2*self.ident)
        self.assertAlmostEqual(detval, detval_direct)

    def test_identity_incr(self):
        track = np.copy(self.ident)
        for _ in range(10):
            self.incdet.update(np.linalg.inv(
                track), self.ident, 0.1*self.ident, self.ident)
            track += 0.1*self.ident
        detval = self.incdet.det()
        _, detval_direct = np.linalg.slogdet(2*self.ident)
        self.assertAlmostEqual(detval, detval_direct)

    def test_random_incr(self):
        mat = generate_pos_def_matrix(self.n)
        incdet = utils.IncLogAbsDeterminant(mat)
        for _ in range(100):
            incr = generate_pos_def_matrix(self.n)
            incdet.update(np.linalg.inv(mat), self.ident, incr, self.ident)
            mat += incr
        detval = incdet.det()
        _, detval_direct = np.linalg.slogdet(mat)
        self.assertAlmostEqual(detval, detval_direct)

    def test_zero_incr(self):
        mat = generate_pos_def_matrix(self.n)
        zero = np.zeros((self.n, self.n))
        incdet = utils.IncLogAbsDeterminant(mat)
        incdet.update(np.linalg.inv(mat), self.ident, zero, self.ident)
        detval = incdet.det()
        _, detval_direct = np.linalg.slogdet(mat)
        self.assertAlmostEqual(detval, detval_direct)

    def test_revert(self):
        mat = generate_pos_def_matrix(self.n)
        incr = generate_pos_def_matrix(self.n)
        incdet = utils.IncLogAbsDeterminant(mat)
        incdet.update(np.linalg.inv(mat),self.ident,incr,self.ident)
        incdet.update(np.linalg.inv(mat+incr),self.ident,-incr,self.ident)
        detval = incdet.det()
        _,detval_direct = np.linalg.slogdet(mat)
        self.assertAlmostEqual(detval,detval_direct)

if __name__ == '__main__':
    unittest.main()
