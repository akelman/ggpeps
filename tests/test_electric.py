import unittest

import numpy as np

from ggpeps import system
from ggpeps import lattice


class TestElectric(unittest.TestCase):
    """The electric energy is a fairly complicated operator.
    This class is for testing its implementation, as well as its helper functions."""

    def setUp(self):

        lat = lattice.Lattice2D(2, 2)
        num_pg_layer = 1
        num_fermionic_layer = 1
        nlayer = num_pg_layer + num_fermionic_layer
        unitcell_size = 1
        paramvec = np.random.rand(nlayer, unitcell_size, 20)
        cfg = system.Z2System2D_G2C_F2C_Config(lat, 1, 1, 1, 1, None, num_pg_layer=1, num_fermionic_layer=1)
        cfg.paramvec = paramvec
        self.system_z2 = system.Z2System2D(cfg)
        self.system_z2.cfg.enforce_parameter_conditions(self.system_z2.cfg.paramvec)

    def test_make_sigma(self):
        # Test for Z2, Zn, Dn
        pass

    def test_bracket_term(self):
        pass

    def test_pfaffian_wick_phase(self):
        pass

    def test_get_cov_matrix_idx(self):
        pass

    def test_simplify_majorana_acc(self):
        pass

    def test_generate_gauged_projector_terms(self):
        pass
