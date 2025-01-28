import unittest
import numpy as np

from ggpeps import lattice, system, utils
from ggpeps import xnp as xnp


class TestSystemBase(unittest.TestCase):

    def setUp(self):
        lat = lattice.Lattice2D(2, 3)

        paramvec = [[0.3, 0.5, 0.8, 0.2, 0.3, 0.9]]
        cfg = system.Z2System2DConfig(lat, 0, 0, 0, 0, None)
        cfg.paramvec = paramvec
        self.system_z2_1c = system.Z2System2D(cfg)

        paramvec2C = np.random.rand(1, 20)
        cfg2C = system.Z2System2D2CConfig(lat, 0, 0, 0, 0, None)
        cfg2C.paramvec = paramvec2C
        self.system_z2_2c = system.Z2System2D(cfg2C)

    def test_link_based_mode_order_1copy(self):

        modes_calc = self.system_z2_1c.get_link_based_mode_order()

        # The following explicit mode ordering was found using pen and paper (well, metaphorically)
        # <mode_letter:maj mode>_<copy>_<link_id>
        modes_manual = [
            "l1_1_0",
            "l2_1_0",
            "r1_1_0",
            "r2_1_0",  # each group of four lines is one link
            "l1_1_1",
            "l2_1_1",
            "r1_1_1",
            "r2_1_1",
            "l1_1_2",
            "l2_1_2",
            "r1_1_2",
            "r2_1_2",
            "l1_1_3",
            "l2_1_3",
            "r1_1_3",
            "r2_1_3",
            "l1_1_4",
            "l2_1_4",
            "r1_1_4",
            "r2_1_4",
            "l1_1_5",
            "l2_1_5",
            "r1_1_5",
            "r2_1_5",
            "d1_1_6",
            "d2_1_6",
            "u1_1_6",
            "u2_1_6",
            "d1_1_7",
            "d2_1_7",
            "u1_1_7",
            "u2_1_7",
            "d1_1_8",
            "d2_1_8",
            "u1_1_8",
            "u2_1_8",
            "d1_1_9",
            "d2_1_9",
            "u1_1_9",
            "u2_1_9",
            "d1_1_10",
            "d2_1_10",
            "u1_1_10",
            "u2_1_10",
            "d1_1_11",
            "d2_1_11",
            "u1_1_11",
            "u2_1_11",
        ]

        self.assertTrue(len(modes_calc) == len(modes_manual))
        for k in range(len(modes_calc)):
            self.assertTrue(modes_calc[k] == modes_manual[k])

    def test_link_based_mode_order_2copy(self):

        modes_calc = self.system_z2_2c.get_link_based_mode_order()

        # The following explicit mode ordering was found using "pen and paper"
        # <mode_letter:maj mode>_<copy>_<link_id>
        modes_manual = [
            "l1_1_0",
            "l2_1_0",
            "r1_1_0",
            "r2_1_0",
            "l1_2_0",
            "l2_2_0",
            "r1_2_0",
            "r2_2_0",  # each group of lines is one link
            "l1_1_1",
            "l2_1_1",
            "r1_1_1",
            "r2_1_1",
            "l1_2_1",
            "l2_2_1",
            "r1_2_1",
            "r2_2_1",
            "l1_1_2",
            "l2_1_2",
            "r1_1_2",
            "r2_1_2",
            "l1_2_2",
            "l2_2_2",
            "r1_2_2",
            "r2_2_2",
            "l1_1_3",
            "l2_1_3",
            "r1_1_3",
            "r2_1_3",
            "l1_2_3",
            "l2_2_3",
            "r1_2_3",
            "r2_2_3",
            "l1_1_4",
            "l2_1_4",
            "r1_1_4",
            "r2_1_4",
            "l1_2_4",
            "l2_2_4",
            "r1_2_4",
            "r2_2_4",
            "l1_1_5",
            "l2_1_5",
            "r1_1_5",
            "r2_1_5",
            "l1_2_5",
            "l2_2_5",
            "r1_2_5",
            "r2_2_5",
            "d1_1_6",
            "d2_1_6",
            "u1_1_6",
            "u2_1_6",
            "d1_2_6",
            "d2_2_6",
            "u1_2_6",
            "u2_2_6",
            "d1_1_7",
            "d2_1_7",
            "u1_1_7",
            "u2_1_7",
            "d1_2_7",
            "d2_2_7",
            "u1_2_7",
            "u2_2_7",
            "d1_1_8",
            "d2_1_8",
            "u1_1_8",
            "u2_1_8",
            "d1_2_8",
            "d2_2_8",
            "u1_2_8",
            "u2_2_8",
            "d1_1_9",
            "d2_1_9",
            "u1_1_9",
            "u2_1_9",
            "d1_2_9",
            "d2_2_9",
            "u1_2_9",
            "u2_2_9",
            "d1_1_10",
            "d2_1_10",
            "u1_1_10",
            "u2_1_10",
            "d1_2_10",
            "d2_2_10",
            "u1_2_10",
            "u2_2_10",
            "d1_1_11",
            "d2_1_11",
            "u1_1_11",
            "u2_1_11",
            "d1_2_11",
            "d2_2_11",
            "u1_2_11",
            "u2_2_11",
        ]

        self.assertTrue(len(modes_calc) == len(modes_manual))
        for k in range(len(modes_calc)):
            self.assertTrue(modes_calc[k] == modes_manual[k])

    def test_site_based_mode_order_1copy(self):

        modes_calc = self.system_z2_1c.get_site_based_mode_order()

        # <mode_letter:maj mode>_<copy>_<link_id>
        modes_manual = [
            "l1_1_1",
            "l2_1_1",
            "r1_1_0",
            "r2_1_0",  # ordered by site
            "d1_1_8",
            "d2_1_8",
            "u1_1_6",
            "u2_1_6",
            "l1_1_0",
            "l2_1_0",
            "r1_1_1",
            "r2_1_1",
            "d1_1_11",
            "d2_1_11",
            "u1_1_9",
            "u2_1_9",
            "l1_1_3",
            "l2_1_3",
            "r1_1_2",
            "r2_1_2",
            "d1_1_6",
            "d2_1_6",
            "u1_1_7",
            "u2_1_7",
            "l1_1_2",
            "l2_1_2",
            "r1_1_3",
            "r2_1_3",
            "d1_1_9",
            "d2_1_9",
            "u1_1_10",
            "u2_1_10",
            "l1_1_5",
            "l2_1_5",
            "r1_1_4",
            "r2_1_4",
            "d1_1_7",
            "d2_1_7",
            "u1_1_8",
            "u2_1_8",
            "l1_1_4",
            "l2_1_4",
            "r1_1_5",
            "r2_1_5",
            "d1_1_10",
            "d2_1_10",
            "u1_1_11",
            "u2_1_11",
        ]

        self.assertTrue(len(modes_calc) == len(modes_manual))
        for k in range(len(modes_calc)):
            self.assertTrue(modes_calc[k] == modes_manual[k])

    def test_site_based_mode_order_2copy(self):

        modes_calc = self.system_z2_2c.get_site_based_mode_order()

        # Every site is 4 lines, first two are the first copy, second two (indented ones) are the second copy
        # <mode_letter:maj mode>_<copy>_<link_id>
        modes_manual = [
            "l1_1_1",
            "l2_1_1",
            "r1_1_0",
            "r2_1_0",
            "d1_1_8",
            "d2_1_8",
            "u1_1_6",
            "u2_1_6",
            "l1_2_1",
            "l2_2_1",
            "r1_2_0",
            "r2_2_0",
            "d1_2_8",
            "d2_2_8",
            "u1_2_6",
            "u2_2_6",
            "l1_1_0",
            "l2_1_0",
            "r1_1_1",
            "r2_1_1",
            "d1_1_11",
            "d2_1_11",
            "u1_1_9",
            "u2_1_9",
            "l1_2_0",
            "l2_2_0",
            "r1_2_1",
            "r2_2_1",
            "d1_2_11",
            "d2_2_11",
            "u1_2_9",
            "u2_2_9",
            "l1_1_3",
            "l2_1_3",
            "r1_1_2",
            "r2_1_2",
            "d1_1_6",
            "d2_1_6",
            "u1_1_7",
            "u2_1_7",
            "l1_2_3",
            "l2_2_3",
            "r1_2_2",
            "r2_2_2",
            "d1_2_6",
            "d2_2_6",
            "u1_2_7",
            "u2_2_7",
            "l1_1_2",
            "l2_1_2",
            "r1_1_3",
            "r2_1_3",
            "d1_1_9",
            "d2_1_9",
            "u1_1_10",
            "u2_1_10",
            "l1_2_2",
            "l2_2_2",
            "r1_2_3",
            "r2_2_3",
            "d1_2_9",
            "d2_2_9",
            "u1_2_10",
            "u2_2_10",
            "l1_1_5",
            "l2_1_5",
            "r1_1_4",
            "r2_1_4",
            "d1_1_7",
            "d2_1_7",
            "u1_1_8",
            "u2_1_8",
            "l1_2_5",
            "l2_2_5",
            "r1_2_4",
            "r2_2_4",
            "d1_2_7",
            "d2_2_7",
            "u1_2_8",
            "u2_2_8",
            "l1_1_4",
            "l2_1_4",
            "r1_1_5",
            "r2_1_5",
            "d1_1_10",
            "d2_1_10",
            "u1_1_11",
            "u2_1_11",
            "l1_2_4",
            "l2_2_4",
            "r1_2_5",
            "r2_2_5",
            "d1_2_10",
            "d2_2_10",
            "u1_2_11",
            "u2_2_11",
        ]

        self.assertTrue(len(modes_calc) == len(modes_manual))
        for k in range(len(modes_calc)):
            self.assertTrue(modes_calc[k] == modes_manual[k])

    def test_gamma_maj_from_dirac(self):
        """Test that gamma_maj is correctly calculated from the Dirac matrices"""
        # We know that the gamma dirac matrices have all the same shape
        site = 0
        m, _ = self.system_z2_1c.gamma_dirac_layervec_sitevec[-1][site].shape
        smat = utils.generate_smat(m)
        gamma_maj_vec = xnp.array(
            [
                xnp.real(smat @ gamma_dirac @ xnp.transpose(smat))
                for gamma_dirac in self.system_z2_1c.gamma_dirac_layervec_sitevec
            ]
        )
        new_calc = xnp.real(
            smat @ self.system_z2_1c.gamma_dirac_layervec_sitevec @ xnp.transpose(smat)
        )
        self.assertTrue((gamma_maj_vec == new_calc).all())


class TestSystemBaseDimensions(unittest.TestCase):
    """Class to test that various attributes of the system have the correct shape."""

    def setUp(self):
        lat = lattice.Lattice2D(2, 2)

        pg_layers = 1
        fermionic_layers = 1
        nlayers = pg_layers + fermionic_layers
        unitcell_size = 2
        paramvec2C = np.random.rand(nlayers, unitcell_size, 20)
        cfg2C = system.Z2System2D_G2C_F2C_Config(
            lat, 0, 0, 0, 0, None, pg_layers, fermionic_layers, unitcell_size
        )
        cfg2C.paramvec = paramvec2C
        self.system_z2_2c = system.Z2System2D(cfg2C)

    def test_paramvec(self):

        paramvec = np.array(self.system_z2_2c.cfg.paramvec)
        actual_shape = paramvec.shape

        target_shape = self.system_z2_2c.cfg.param_shape()
        self.assertTrue(actual_shape == target_shape)

    def test_tmat_layervec_unitcellvec(self):

        tmat_layervec_unitcellvec = np.array(
            self.system_z2_2c.tmat_layervec_unitcellvec
        )
        actual_shape = tmat_layervec_unitcellvec.shape

        num_dirac_modes = 1 + 2 * 4  # (1 physical + 2 copies * 4 links)
        target_shape = (
            self.system_z2_2c.cfg.nlayer,
            self.system_z2_2c.cfg.max_unitcell_size,
            num_dirac_modes,
            num_dirac_modes,
        )
        self.assertTrue(actual_shape == target_shape)

    def test_tmat_layervec_sitevec(self):

        tmat_layervec_sitevec = np.array(self.system_z2_2c.tmat_layervec_sitevec)
        actual_shape = tmat_layervec_sitevec.shape

        num_dirac_modes = 1 + 2 * 4  # 1 physical + 2 copies * 4 links
        target_shape = (
            self.system_z2_2c.cfg.nlayer,
            self.system_z2_2c.cfg.lattice.size,
            num_dirac_modes,
            num_dirac_modes,
        )
        self.assertTrue(actual_shape == target_shape)

    def test_gamma_dirac_layervec_sitevec(self):

        gamma_dirac_layervec_sitevec = np.array(
            self.system_z2_2c.gamma_dirac_layervec_sitevec
        )
        actual_shape = gamma_dirac_layervec_sitevec.shape

        num_dirac_modes = 2 * (1 + 2 * 4)  # 2 modes * (1 physical + 2 copies * 4 links)
        target_shape = (
            self.system_z2_2c.cfg.nlayer,
            self.system_z2_2c.cfg.lattice.size,
            num_dirac_modes,
            num_dirac_modes,
        )
        self.assertTrue(actual_shape == target_shape)

    def test_gamma_maj_layervec_sitevec(self):

        gamma_maj_layervec_sitevec = np.array(
            self.system_z2_2c.gamma_maj_layervec_sitevec
        )
        actual_shape = gamma_maj_layervec_sitevec.shape

        num_maj_modes = 2 * (1 + 2 * 4)  # 2 modes * (1 physical + 2 copies * 4 links)
        target_shape = (
            self.system_z2_2c.cfg.nlayer,
            self.system_z2_2c.cfg.lattice.size,
            num_maj_modes,
            num_maj_modes,
        )
        self.assertTrue(actual_shape == target_shape)

    def test_gamma_maj_sys_vec(self):

        gamma_maj_sys_vec = np.array(self.system_z2_2c.gamma_maj_sys_vec)
        actual_shape = gamma_maj_sys_vec.shape

        num_modes_per_site = 2 * (
            1 + 2 * 4
        )  # 2 modes * (1 physical + 2 copies * 4 links)
        num_modes = num_modes_per_site * self.system_z2_2c.cfg.lattice.size
        target_shape = (
            self.system_z2_2c.cfg.nlayer,
            num_modes,
            num_modes,
        )
        self.assertTrue(actual_shape == target_shape)

    def test_gamma_maj_sys_deriv(self):

        symb = self.system_z2_2c.symbolvec[0]  # arbitrarily chosen symbol
        gamma_maj_sys_deriv_vec = self.system_z2_2c.gamma_maj_sys_deriv_vec(symb)
        actual_shape = gamma_maj_sys_deriv_vec.shape

        num_modes_per_site = 2 * (
            1 + 2 * 4
        )  # 2 modes * (1 physical + 2 copies * 4 links)
        num_modes = num_modes_per_site * self.system_z2_2c.cfg.lattice.size
        target_shape = (
            self.system_z2_2c.cfg.nlayer,
            self.system_z2_2c.cfg.max_unitcell_size,
            num_modes,
            num_modes,
        )
        self.assertTrue(actual_shape == target_shape)

    def test_grad_norm(self):

        lay = 0
        uc_ind = 0
        grad_over_norm = self.system_z2_2c.compute_grad_norm(lay, uc_ind)
        actual_shape = grad_over_norm.shape

        target_shape = (len(self.system_z2_2c.symbolvec),)

        self.assertTrue(actual_shape == target_shape)

    def test_grad_norm_vec(self):

        grad_over_norm_vec = self.system_z2_2c.compute_grad_norm_vec()
        actual_shape = grad_over_norm_vec.shape

        target_shape = self.system_z2_2c.cfg.param_shape()

        self.assertTrue(actual_shape == target_shape)

    def test_grad_over_norm_dict(self):

        grad_over_norm_dict = self.system_z2_2c._grad_over_norm_dict
        key = list(grad_over_norm_dict.keys())[0]  # pick the first key arbitrarily

        self.assertTrue(len(key) == 3)
