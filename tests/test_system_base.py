import unittest
import numpy as np

import ggpeps
from ggpeps import lattice, system, utils
from ggpeps import xnp as xnp
from ggpeps.system.config_base import get_cov_matrix_idx
from ggpeps.system.system_base import System2DBase


class TestSystemBase(unittest.TestCase):

    def setUp(self):
        lat = lattice.Lattice2D(2, 3)

        paramvec = [[0.3, 0.5, 0.8, 0.2, 0.3, 0.9]]
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
        self.system_z2_1c = system.Z2System2D(cfg)

        paramvec2C = np.random.rand(1, 20)
        cfg2C = system.Z2System2D_Config(lat, 0, 0, 0, 0, None, ncopy=2, num_fermionic_layer=0)
        cfg2C.paramvec = paramvec2C
        self.system_z2_2c = system.Z2System2D(cfg2C)

    def test_link_based_mode_order_1copy(self):

        modes_calc = self.system_z2_1c.get_link_based_mode_order()

        # The following explicit mode ordering was found using pen and paper (well, metaphorically)
        # <mode_letter:maj mode>_<copy>_<color>_<link_id>
        # Here we only test with one color, so color is always 1
        modes_manual = [
            "l1_1_1_0",
            "l2_1_1_0",
            "r1_1_1_0",
            "r2_1_1_0",  # each group of four lines is one link
            "l1_1_1_1",
            "l2_1_1_1",
            "r1_1_1_1",
            "r2_1_1_1",
            "l1_1_1_2",
            "l2_1_1_2",
            "r1_1_1_2",
            "r2_1_1_2",
            "l1_1_1_3",
            "l2_1_1_3",
            "r1_1_1_3",
            "r2_1_1_3",
            "l1_1_1_4",
            "l2_1_1_4",
            "r1_1_1_4",
            "r2_1_1_4",
            "l1_1_1_5",
            "l2_1_1_5",
            "r1_1_1_5",
            "r2_1_1_5",
            "d1_1_1_6",
            "d2_1_1_6",
            "u1_1_1_6",
            "u2_1_1_6",
            "d1_1_1_7",
            "d2_1_1_7",
            "u1_1_1_7",
            "u2_1_1_7",
            "d1_1_1_8",
            "d2_1_1_8",
            "u1_1_1_8",
            "u2_1_1_8",
            "d1_1_1_9",
            "d2_1_1_9",
            "u1_1_1_9",
            "u2_1_1_9",
            "d1_1_1_10",
            "d2_1_1_10",
            "u1_1_1_10",
            "u2_1_1_10",
            "d1_1_1_11",
            "d2_1_1_11",
            "u1_1_1_11",
            "u2_1_1_11",
        ]

        self.assertTrue(len(modes_calc) == len(modes_manual))
        for k in range(len(modes_calc)):
            self.assertTrue(modes_calc[k] == modes_manual[k])

    def test_link_based_mode_order_2copy(self):

        modes_calc = self.system_z2_2c.get_link_based_mode_order()

        # The following explicit mode ordering was found using "pen and paper"
        # <mode_letter:maj mode>_<copy>_<color>_<link_id>
        modes_manual = [
            "l1_1_1_0",
            "l2_1_1_0",
            "r1_1_1_0",
            "r2_1_1_0",
            "l1_2_1_0",
            "l2_2_1_0",
            "r1_2_1_0",
            "r2_2_1_0",  # each group of lines is one link
            "l1_1_1_1",
            "l2_1_1_1",
            "r1_1_1_1",
            "r2_1_1_1",
            "l1_2_1_1",
            "l2_2_1_1",
            "r1_2_1_1",
            "r2_2_1_1",
            "l1_1_1_2",
            "l2_1_1_2",
            "r1_1_1_2",
            "r2_1_1_2",
            "l1_2_1_2",
            "l2_2_1_2",
            "r1_2_1_2",
            "r2_2_1_2",
            "l1_1_1_3",
            "l2_1_1_3",
            "r1_1_1_3",
            "r2_1_1_3",
            "l1_2_1_3",
            "l2_2_1_3",
            "r1_2_1_3",
            "r2_2_1_3",
            "l1_1_1_4",
            "l2_1_1_4",
            "r1_1_1_4",
            "r2_1_1_4",
            "l1_2_1_4",
            "l2_2_1_4",
            "r1_2_1_4",
            "r2_2_1_4",
            "l1_1_1_5",
            "l2_1_1_5",
            "r1_1_1_5",
            "r2_1_1_5",
            "l1_2_1_5",
            "l2_2_1_5",
            "r1_2_1_5",
            "r2_2_1_5",
            "d1_1_1_6",
            "d2_1_1_6",
            "u1_1_1_6",
            "u2_1_1_6",
            "d1_2_1_6",
            "d2_2_1_6",
            "u1_2_1_6",
            "u2_2_1_6",
            "d1_1_1_7",
            "d2_1_1_7",
            "u1_1_1_7",
            "u2_1_1_7",
            "d1_2_1_7",
            "d2_2_1_7",
            "u1_2_1_7",
            "u2_2_1_7",
            "d1_1_1_8",
            "d2_1_1_8",
            "u1_1_1_8",
            "u2_1_1_8",
            "d1_2_1_8",
            "d2_2_1_8",
            "u1_2_1_8",
            "u2_2_1_8",
            "d1_1_1_9",
            "d2_1_1_9",
            "u1_1_1_9",
            "u2_1_1_9",
            "d1_2_1_9",
            "d2_2_1_9",
            "u1_2_1_9",
            "u2_2_1_9",
            "d1_1_1_10",
            "d2_1_1_10",
            "u1_1_1_10",
            "u2_1_1_10",
            "d1_2_1_10",
            "d2_2_1_10",
            "u1_2_1_10",
            "u2_2_1_10",
            "d1_1_1_11",
            "d2_1_1_11",
            "u1_1_1_11",
            "u2_1_1_11",
            "d1_2_1_11",
            "d2_2_1_11",
            "u1_2_1_11",
            "u2_2_1_11",
        ]

        self.assertTrue(len(modes_calc) == len(modes_manual))
        for k in range(len(modes_calc)):
            self.assertTrue(modes_calc[k] == modes_manual[k])

    def test_site_based_mode_order_1copy(self):

        modes_calc = self.system_z2_1c.get_site_based_mode_order()

        # <mode_letter:maj mode>_<copy>_<color>_<link_id>
        modes_manual = [
            "l1_1_1_1",
            "l2_1_1_1",
            "r1_1_1_0",
            "r2_1_1_0",  # ordered by site
            "d1_1_1_8",
            "d2_1_1_8",
            "u1_1_1_6",
            "u2_1_1_6",
            "l1_1_1_0",
            "l2_1_1_0",
            "r1_1_1_1",
            "r2_1_1_1",
            "d1_1_1_11",
            "d2_1_1_11",
            "u1_1_1_9",
            "u2_1_1_9",
            "l1_1_1_3",
            "l2_1_1_3",
            "r1_1_1_2",
            "r2_1_1_2",
            "d1_1_1_6",
            "d2_1_1_6",
            "u1_1_1_7",
            "u2_1_1_7",
            "l1_1_1_2",
            "l2_1_1_2",
            "r1_1_1_3",
            "r2_1_1_3",
            "d1_1_1_9",
            "d2_1_1_9",
            "u1_1_1_10",
            "u2_1_1_10",
            "l1_1_1_5",
            "l2_1_1_5",
            "r1_1_1_4",
            "r2_1_1_4",
            "d1_1_1_7",
            "d2_1_1_7",
            "u1_1_1_8",
            "u2_1_1_8",
            "l1_1_1_4",
            "l2_1_1_4",
            "r1_1_1_5",
            "r2_1_1_5",
            "d1_1_1_10",
            "d2_1_1_10",
            "u1_1_1_11",
            "u2_1_1_11",
        ]

        self.assertTrue(len(modes_calc) == len(modes_manual))
        for k in range(len(modes_calc)):
            self.assertTrue(modes_calc[k] == modes_manual[k])

    def test_site_based_mode_order_2copy(self):

        modes_calc = self.system_z2_2c.get_site_based_mode_order()

        # Every site is 4 lines, first two are the first copy, second two are the second copy
        # <mode_letter:maj mode>_<copy>_<color>_<link_id>
        # Here we only test with one color, so color is always 1
        modes_manual = [
            "l1_1_1_1",
            "l2_1_1_1",
            "r1_1_1_0",
            "r2_1_1_0",
            "d1_1_1_8",
            "d2_1_1_8",
            "u1_1_1_6",
            "u2_1_1_6",
            "l1_2_1_1",
            "l2_2_1_1",
            "r1_2_1_0",
            "r2_2_1_0",
            "d1_2_1_8",
            "d2_2_1_8",
            "u1_2_1_6",
            "u2_2_1_6",
            "l1_1_1_0",
            "l2_1_1_0",
            "r1_1_1_1",
            "r2_1_1_1",
            "d1_1_1_11",
            "d2_1_1_11",
            "u1_1_1_9",
            "u2_1_1_9",
            "l1_2_1_0",
            "l2_2_1_0",
            "r1_2_1_1",
            "r2_2_1_1",
            "d1_2_1_11",
            "d2_2_1_11",
            "u1_2_1_9",
            "u2_2_1_9",
            "l1_1_1_3",
            "l2_1_1_3",
            "r1_1_1_2",
            "r2_1_1_2",
            "d1_1_1_6",
            "d2_1_1_6",
            "u1_1_1_7",
            "u2_1_1_7",
            "l1_2_1_3",
            "l2_2_1_3",
            "r1_2_1_2",
            "r2_2_1_2",
            "d1_2_1_6",
            "d2_2_1_6",
            "u1_2_1_7",
            "u2_2_1_7",
            "l1_1_1_2",
            "l2_1_1_2",
            "r1_1_1_3",
            "r2_1_1_3",
            "d1_1_1_9",
            "d2_1_1_9",
            "u1_1_1_10",
            "u2_1_1_10",
            "l1_2_1_2",
            "l2_2_1_2",
            "r1_2_1_3",
            "r2_2_1_3",
            "d1_2_1_9",
            "d2_2_1_9",
            "u1_2_1_10",
            "u2_2_1_10",
            "l1_1_1_5",
            "l2_1_1_5",
            "r1_1_1_4",
            "r2_1_1_4",
            "d1_1_1_7",
            "d2_1_1_7",
            "u1_1_1_8",
            "u2_1_1_8",
            "l1_2_1_5",
            "l2_2_1_5",
            "r1_2_1_4",
            "r2_2_1_4",
            "d1_2_1_7",
            "d2_2_1_7",
            "u1_2_1_8",
            "u2_2_1_8",
            "l1_1_1_4",
            "l2_1_1_4",
            "r1_1_1_5",
            "r2_1_1_5",
            "d1_1_1_10",
            "d2_1_1_10",
            "u1_1_1_11",
            "u2_1_1_11",
            "l1_2_1_4",
            "l2_2_1_4",
            "r1_2_1_5",
            "r2_2_1_5",
            "d1_2_1_10",
            "d2_2_1_10",
            "u1_2_1_11",
            "u2_2_1_11",
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
        new_calc = xnp.real(smat @ self.system_z2_1c.gamma_dirac_layervec_sitevec @ xnp.transpose(smat))
        self.assertTrue((gamma_maj_vec == new_calc).all())


class TestSystemBaseDimensions(unittest.TestCase):
    """Class to test that various attributes of the system have the correct shape."""

    def setUp(self):
        lat = lattice.Lattice2D(2, 2)

        pg_layers = 1
        fermionic_layers = 1
        nlayers = pg_layers + fermionic_layers
        mod_link_inds = (0,)
        unitcell_size = 2
        paramvec2C = np.random.rand(nlayers, unitcell_size, 20)
        cfg2C = system.Z2System2D_Config(
            lat,
            0,
            0,
            0,
            0,
            None,
            num_pg_layer=pg_layers,
            num_fermionic_layer=fermionic_layers,
            mod_link_inds=mod_link_inds,
            unitcell_size=unitcell_size,
            ncopy=2,
        )
        cfg2C.paramvec = paramvec2C
        self.system_z2_2c = system.Z2System2D(cfg2C)

    def test_paramvec(self):

        paramvec = np.array(self.system_z2_2c.cfg.paramvec)
        actual_shape = paramvec.shape

        target_shape = self.system_z2_2c.cfg.param_shape()
        self.assertTrue(actual_shape == target_shape)

    def test_tmat_layervec_unitcellvec(self):

        tmat_layervec_unitcellvec = np.array(self.system_z2_2c.tmat_layervec_unitcellvec)
        actual_shape = tmat_layervec_unitcellvec.shape

        num_dirac_modes = 1 + 2 * 4  # (1 physical + 2 copies * 4 links)
        target_shape = (
            self.system_z2_2c.cfg.nlayer,
            self.system_z2_2c.cfg.unitcell_size,
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

        gamma_dirac_layervec_sitevec = np.array(self.system_z2_2c.gamma_dirac_layervec_sitevec)
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

        gamma_maj_layervec_sitevec = np.array(self.system_z2_2c.gamma_maj_layervec_sitevec)
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

        num_modes_per_site = 2 * (1 + 2 * 4)  # 2 modes * (1 physical + 2 copies * 4 links)
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

        num_modes_per_site = 2 * (1 + 2 * 4)  # 2 modes * (1 physical + 2 copies * 4 links)
        num_modes = num_modes_per_site * self.system_z2_2c.cfg.lattice.size
        target_shape = (
            self.system_z2_2c.cfg.nlayer,
            self.system_z2_2c.cfg.unitcell_size,
            num_modes,
            num_modes,
        )
        self.assertTrue(actual_shape == target_shape)

    def test_grad_norm_vec(self):

        grad_over_norm_vec = self.system_z2_2c.grad_over_norm_vec
        actual_shape = grad_over_norm_vec.shape

        target_shape = self.system_z2_2c.cfg.param_shape()

        self.assertTrue(actual_shape == target_shape)

    def test_grad_over_norm(self):

        grad_over_norm_dict = self.system_z2_2c.grad_over_norm_vec
        shape = grad_over_norm_dict.shape
        target_shape = self.system_z2_2c.cfg.param_shape()

        self.assertTrue(shape == target_shape)


class TestMultiColorModeOrder(unittest.TestCase):
    """Regression tests for the color-outer / copy-inner mode ordering when num_colors > 1.

    The Z2 mode-order tests above only ever exercise num_colors=1 (the "2C" Z2 configs
    have two *copies* but a single color, since the Z2 representation is 1-dimensional).
    With a single color the `for color` loop in get_link_based_mode_order /
    get_site_based_mode_order is trivial, so the color-vs-copy nesting is never checked.
    """

    def setUp(self):
        lat = lattice.Lattice2D(2, 2)
        num_pg_layer = 1
        num_fermionic_layer = 0
        paramvec = np.random.rand(num_pg_layer + num_fermionic_layer, 1, 20)
        cfg = system.D6System2D_Config(
            lat,
            1,
            1,
            0,
            0,
            None,
            ncopy=2,
            num_pg_layer=num_pg_layer,
            num_fermionic_layer=num_fermionic_layer,
        )
        cfg.paramvec = paramvec
        self.system_D6 = system.D2nSystem2D(cfg)
        self.system_D6.cfg.enforce_parameter_conditions(self.system_D6.cfg.paramvec)

        self.ncolor = self.system_D6.cfg.gaugemgr.rep_dim
        self.ncopy = self.system_D6.cfg.ncopy
        # Guard: these tests are only meaningful for a genuinely multi-color, multi-copy system.
        self.assertEqual((self.ncolor, self.ncopy), (2, 2))

    def test_link_based_mode_order_D6_first_link(self):
        """The first horizontal link must list its 16 modes color-outer, copy-inner.

        Mode string format is "<letter><maj>_<copy>_<color>_<link>".
        """
        modes = self.system_D6.get_link_based_mode_order()
        expected_first_link = [
            # color 1, copy 1
            "l1_1_1_0", "l2_1_1_0", "r1_1_1_0", "r2_1_1_0",
            # color 1, copy 2
            "l1_2_1_0", "l2_2_1_0", "r1_2_1_0", "r2_2_1_0",
            # color 2, copy 1
            "l1_1_2_0", "l2_1_2_0", "r1_1_2_0", "r2_1_2_0",
            # color 2, copy 2
            "l1_2_2_0", "l2_2_2_0", "r1_2_2_0", "r2_2_2_0",
        ]  # fmt: skip
        self.assertEqual(modes[: len(expected_first_link)], expected_first_link)

    def test_site_based_mode_order_D6_first_site(self):
        """The first site must list its modes color-outer, copy-inner, with the 8 dir/maj
        modes l1,l2,r1,r2,d1,d2,u1,u2 per (color, copy). Link ids are lattice-specific,
        so only the (letter, copy, color) nesting is asserted here.
        """
        modes = self.system_D6.get_site_based_mode_order()
        modes_per_site = 8 * self.ncolor * self.ncopy
        first_site = modes[:modes_per_site]

        dirmaj = ["l1", "l2", "r1", "r2", "d1", "d2", "u1", "u2"]
        expected = [
            (letter, copy, color)
            for color in range(1, self.ncolor + 1)
            for copy in range(1, self.ncopy + 1)
            for letter in dirmaj
        ]
        actual = []
        for mode in first_site:
            letter, copy_s, color_s, _link = mode.split("_")
            actual.append((letter, int(copy_s), int(color_s)))
        self.assertEqual(actual, expected)

    def test_link_order_consistent_with_get_cov_matrix_idx(self):
        """Within a single link, the position of each mode in get_link_based_mode_order
        must equal get_cov_matrix_idx(...) -- i.e. the mode-order builder and the index
        helper share one convention. This is the direct contract _expand_gamma_maj_to_system
        depends on, checked against an independent function so it is not a loop self-copy.
        """
        modes = self.system_D6.get_link_based_mode_order()
        modes_per_link = 4 * self.ncolor * self.ncopy
        first_link = modes[:modes_per_link]

        # letter -> (direction, majorana); link has 2 directions (1=left, 2=right).
        letter_to_dir_maj = {"l1": (1, 1), "l2": (1, 2), "r1": (2, 1), "r2": (2, 2)}
        for pos, mode in enumerate(first_link):
            letter, copy_s, color_s, _link = mode.split("_")
            direction, majorana = letter_to_dir_maj[letter]
            idx = get_cov_matrix_idx(int(color_s), int(copy_s), direction, majorana, self.ncolor, self.ncopy)
            with self.subTest(mode=mode):
                self.assertEqual(pos, idx)

    def test_color_outer_invariant(self):
        """Anti-transposition guard: within each link, every color-1 mode precedes every
        color-2 mode (the defining property of color-outer ordering). A copy<->color swap
        would interleave the colors and break this.
        """
        modes = self.system_D6.get_link_based_mode_order()
        modes_per_link = 4 * self.ncolor * self.ncopy
        first_link = modes[:modes_per_link]
        colors_in_order = [int(mode.split("_")[2]) for mode in first_link]
        # The color sequence must be non-decreasing (1,1,...,2,2,...), never interleaved.
        self.assertEqual(colors_in_order, sorted(colors_in_order))
        self.assertEqual(set(colors_in_order), {1, 2})

    def test_gamma_maj_sys_vec_D6_is_valid_covariance(self):
        """End-to-end smoke test: the multi-color path through _expand_gamma_maj_to_system
        runs and yields a real, antisymmetric, pure-state (Gamma^2 = -I) covariance matrix.
        """
        gamma_sys = np.array(self.system_D6.gamma_maj_sys_vec)
        for lay in range(gamma_sys.shape[0]):
            g = gamma_sys[lay]
            with self.subTest(layer=lay):
                self.assertTrue(np.allclose(g.imag, 0))
                self.assertTrue(np.allclose(g, -np.transpose(g)))
                self.assertTrue(np.allclose(g @ g, -np.eye(g.shape[0])))


class TestIncrementalModCovmats(unittest.TestCase):
    """The open-link ("mod") family is maintained incrementally across single-link gauge updates:
    gamma_in_sys_mod is patched in place (_patch_gamma_in_sys_mod) instead of re-extracted, and
    during warmup the whole family is deferred and re-anchored once (reanchor_mod_trackers).
    Both must reproduce the from-scratch computation."""

    @staticmethod
    def _build_d6(seed):
        lat = lattice.Lattice2D(2, 2)
        cfg = system.D6System2D_Config(
            lat, 1, 1, 0, 0, None, ncopy=2, num_pg_layer=2, num_fermionic_layer=0, mod_link_inds=(0,)
        )
        rng = np.random.RandomState(seed)
        cfg.paramvec = rng.rand(*cfg.param_shape())
        cfg.enforce_parameter_conditions(cfg.paramvec)
        return cfg, system.D2nSystem2D(cfg), rng

    @staticmethod
    def _build_z2(seed):
        lat = lattice.Lattice2D(2, 2)
        cfg = system.Z2System2D_Config(
            lat, 1.0, 1.0, 0.0, 0.0, np.zeros(0), ncopy=2, num_pg_layer=1, num_fermionic_layer=0, mod_link_inds=(0,)
        )
        rng = np.random.RandomState(seed)
        cfg.paramvec = rng.rand(*cfg.param_shape())
        cfg.enforce_parameter_conditions(cfg.paramvec)
        return cfg, system.Z2System2D(cfg), rng

    @staticmethod
    def _update_sequence(cfg, rng, nsteps=25):
        gvals = cfg.gaugemgr.get_possible_gauge_values()
        return [
            (int(rng.randint(0, cfg.lattice.nlinks)), gvals[int(rng.randint(0, len(gvals)))]) for _ in range(nsteps)
        ]

    def _check_mod_patch(self, cfg, sys_, rng):
        sys_.initialize()
        for link, theta in self._update_sequence(cfg, rng):
            sys_.update_gauge_ind(link, theta)
        patched = np.asarray(sys_.gamma_in_sys_mod_vec)  # maintained by _patch_gamma_in_sys_mod
        fresh = np.asarray(sys_._extract_gamma_in_sys_mod_vec(cfg.mod_link_inds, sys_.gamma_in_sys_vec))
        self.assertLess(np.abs(patched - fresh).max(), 1e-14)

    def _check_defer_reanchor(self, build, seed):
        cfg_a, sys_a, rng = build(seed)
        cfg_b, sys_b, _ = build(seed)
        updates = self._update_sequence(cfg_a, rng)

        sys_a.initialize()
        sys_a.defer_mod_trackers = True  # warmup mode: mod family not maintained
        for link, theta in updates:
            sys_a.update_gauge_ind(link, theta)
        sys_a.defer_mod_trackers = False
        sys_a.reanchor_mod_trackers()

        sys_b.initialize()  # mod family maintained incrementally on every accepted step
        _ = sys_b.gamma_in_sys_mod_vec, sys_b.wi_gamma_in_mod_vec  # force live tracking from the start
        for link, theta in updates:
            sys_b.update_gauge_ind(link, theta)

        for name in ("gamma_in_sys_mod_vec", "wi_gamma_in_mod_vec", "wi_gamma_out_mod_vec", "incdet_mod_vec"):
            a = np.asarray(getattr(sys_a, name))
            b = np.asarray(getattr(sys_b, name))
            scale = max(np.abs(a).max(), 1.0)
            self.assertLess(np.abs(a - b).max() / scale, 1e-9, msg=name)

    def test_mod_patch_matches_fresh_extract_d6(self):
        self._check_mod_patch(*self._build_d6(seed=5))

    def test_patch_matches_reference_for_every_link(self):
        """Direct check of _patch_gamma_in_sys_mod on synthetic data against a plain-numpy
        reference, for every link_ind: below / between / above / equal to the measured links
        (the equal case must leave that measured-link copy untouched)."""
        nlayer, nvirt, nlinks = 2, 2, 6
        mod_link_inds = (0, 3)
        k = 2 * nvirt
        n = k * nlinks
        rng = np.random.RandomState(11)
        gamma = rng.rand(nlayer, n, n)
        old_mod = rng.rand(nlayer, len(mod_link_inds), n - k, n - k)

        for link_ind in range(nlinks):
            reference = old_mod.copy()
            ind_mat = k * link_ind
            for mi, m in enumerate(mod_link_inds):
                if link_ind == m:
                    continue
                pos = ind_mat if link_ind < m else ind_mat - k
                reference[:, mi, pos : pos + k, pos : pos + k] = gamma[:, ind_mat : ind_mat + k, ind_mat : ind_mat + k]

            patched = System2DBase._patch_gamma_in_sys_mod(
                xnp.asarray(gamma), xnp.asarray(old_mod.copy()), link_ind, mod_link_inds, nvirt
            )
            with self.subTest(link_ind=link_ind):
                self.assertTrue(np.array_equal(np.asarray(patched), reference))

    @unittest.skipUnless(ggpeps.PREFERRED_BACKEND == "jax", "compile count is only meaningful under jit")
    def test_update_impl_compiles_once_per_direction(self):
        """link_ind and the gauge value are traced: a run of single-link updates must compile
        the _update_gauge_ind kernel at most once per link direction."""
        cfg, sys_, rng = self._build_d6(seed=3)
        sys_.initialize()
        _ = sys_.gamma_in_sys_mod_vec, sys_.wi_gamma_in_mod_vec  # live mod tracking from the start
        System2DBase._update_gauge_ind.clear_cache()
        for link, theta in self._update_sequence(cfg, rng, nsteps=12):
            sys_.update_gauge_ind(link, theta)
        self.assertLessEqual(System2DBase._update_gauge_ind._cache_size(), 2)

    @unittest.skipUnless(ggpeps.PREFERRED_BACKEND == "jax", "compile count is only meaningful under jit")
    def test_patch_compiles_once_across_link_inds(self):
        """link_ind is traced, not static: changing it between calls must not recompile."""
        nlayer, nvirt, nlinks = 2, 2, 6
        k = 2 * nvirt
        n = k * nlinks
        rng = np.random.RandomState(11)
        gamma = xnp.asarray(rng.rand(nlayer, n, n))
        old_mod = xnp.asarray(rng.rand(nlayer, 2, n - k, n - k))

        patch = System2DBase._patch_gamma_in_sys_mod
        patch.clear_cache()
        for link_ind in range(nlinks):
            patch(gamma, old_mod, link_ind, (0, 3), nvirt)
        self.assertEqual(patch._cache_size(), 1)

    def test_mod_patch_matches_fresh_extract_z2(self):
        self._check_mod_patch(*self._build_z2(seed=13))

    def test_defer_reanchor_matches_live_tracking_d6(self):
        self._check_defer_reanchor(self._build_d6, seed=7)

    def test_defer_reanchor_matches_live_tracking_z2(self):
        self._check_defer_reanchor(self._build_z2, seed=17)
