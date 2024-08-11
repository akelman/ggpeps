import os
import importlib

import unittest 
from unittest import skip

import numpy as np
import jax.numpy as jnp

import ggpeps
from ggpeps import lattice
#rom ggpeps import xnp
from ggpeps.system.global_funcs import *

from tests import TEST_BACKEND

# ======================= Z2 fermionic system (4 copies) =========================================

class TestBackends(unittest.TestCase):
    def setUp(self):
        # We don't create system objects, etc. here, 
        # because we want to have explicit control over which backend is used
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

    def test_tmat_equivalence(self):
        print(dir(ggpeps))
        lat = lattice.Lattice2D(2,2)
        paramvec = np.random.rand(2, 20)

        # numpy
        os.environ["GGPEPS_BACKEND"] = "numpy"
        importlib.reload(ggpeps)

        cfg = ggpeps.system.Z2System2D_G2C_F2C_Config(lat, 1,1,1,1)
        cfg.paramvec = paramvec
        system_z2 = ggpeps.system.Z2System2D_G2C_F2C(cfg) 
        system_z2.cfg.enforce_parameter_conditions(system_z2.cfg.paramvec)
        tmat_vec_np = system_z2.tmat_vec

        # jax
        os.environ["GGPEPS_BACKEND"] = "jax"
        importlib.reload(ggpeps)

        cfg = ggpeps.system.Z2System2D_G2C_F2C_Config(lat, 1,1,1,1)
        cfg.paramvec = paramvec
        system_z2 = ggpeps.system.Z2System2D_G2C_F2C(cfg) 
        system_z2.cfg.enforce_parameter_conditions(system_z2.cfg.paramvec)
        tmat_vec_jax = system_z2.tmat_vec

        self.assertTrue(np.allclose(tmat_vec_np, np.array(tmat_vec_jax)))

    def test_gamma_dirac_vec(self):
        #global ggpeps
        #import ggpeps
        lat = ggpeps.lattice.Lattice2D(2,2)
        paramvec = np.random.rand(2, 20)
        
        # jax
        #def test_jax():
        os.environ["GGPEPS_BACKEND"] = "jax"
        #importlib.invalidate_caches()
        importlib.reload(ggpeps)

        cfg = ggpeps.system.Z2System2D_G2C_F2C_Config(lat, 1,1,1,1)
        cfg.paramvec = paramvec
        system_z2 = ggpeps.system.Z2System2D_G2C_F2C(cfg) 
        system_z2.cfg.enforce_parameter_conditions(system_z2.cfg.paramvec)
        gamma_dirac_vec_jax = system_z2.gamma_dirac_vec
            #return gamma_dirac_vec_jax

        # numpy
        os.environ["GGPEPS_BACKEND"] = "numpy"
        importlib.reload(ggpeps)

        cfg = ggpeps.system.Z2System2D_G2C_F2C_Config(lat, 1,1,1,1)
        cfg.paramvec = paramvec
        system_z2 = ggpeps.system.Z2System2D_G2C_F2C(cfg) 
        system_z2.cfg.enforce_parameter_conditions(system_z2.cfg.paramvec)
        gamma_dirac_vec_np = system_z2.gamma_dirac_vec
        print(type(gamma_dirac_vec_jax))
        self.assertTrue(np.allclose(gamma_dirac_vec_np, np.array(gamma_dirac_vec_jax)))


