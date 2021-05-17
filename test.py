import unittest
import numpy as np
import utils
import system
import lattice 
from measurement import Measurement
import copy

class TestMCMethods(unittest.TestCase):

    def test(self):
        pass

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
        self.lat=lattice.Lattice2D(8,8)
    
    def test_ind2coord(self):
        ref=(3,4)
        ind=self.lat.coord2ind(ref)
        coord=self.lat.ind2coord(ind)
        self.assertEqual(ref,coord)

    def test_ind2coord_dir(self):
        coord_ref=(2,3)
        for dir_ref in lattice.Direction:
            ind=self.lat.coord2ind_dir(coord_ref,dir_ref)
            coord,dir=self.lat.ind2coord_dir(ind)
            self.assertEqual(coord_ref,coord)
            self.assertEqual(dir_ref,dir)


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

    def test_gamma_maj_covariance(self):
        gamma_maj=self.system_z2_2_2.gamma_maj
        m, n = gamma_maj.shape
        self.assertEqual(m, n)
        self.assertTrue(utils.is_antisymmetric(gamma_maj))
        self.assertTrue(np.allclose(gamma_maj@gamma_maj,-np.eye(m)))
        self.assertTrue(np.allclose(gamma_maj@np.transpose(gamma_maj),np.eye(m)))

    def test_gamma_maj_sys_covariance(self):
        gamma_maj=self.system_z2_2_2.gamma_maj_sys
        m, n = gamma_maj.shape
        self.assertEqual(m, n)
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

class TestMeasurements(unittest.TestCase): 
    def testAdd(self):
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

    def testRandomScalar(self):
        meas1=Measurement("meas1",1)
        meas2=Measurement("meas2",2)
        for _ in range(1000):
            rnd=np.random.rand()
            meas1.append(rnd)
            meas2.append(rnd)
        self.assertAlmostEqual(meas1.mean(),meas2.mean())

    def testRandomArray(self):
        meas1=Measurement("meas1",1)
        meas2=Measurement("meas2",2)
        for _ in range(1000):
            rnd=np.random.rand(2,2)
            meas1.append(rnd)
            meas2.append(rnd)
        self.assertTrue(np.allclose(meas1.mean(),meas2.mean()))

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
