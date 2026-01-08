import unittest

import ggpeps.utils as utils
from tools import build_slurm


class TestTools(unittest.TestCase):

    def setUp(self):
        pass

    def test_build_slurm_arg_extraction(self):
        """Test that arguments are correctly extracted from folder names."""

        dirname = "L_2_g_1.3_int_1.4_mass_1.5_chem_1.0_2.0_3.0"
        size_str = build_slurm.format_arg(dirname, "L")
        elmag_str = build_slurm.format_arg(dirname, "g")
        int_str = build_slurm.format_arg(dirname, "int")
        mass_str = build_slurm.format_arg(dirname, "mass")
        chem_str = build_slurm.format_arg(dirname, "chem")
        self.assertEqual(size_str, "--L 2")
        self.assertEqual(elmag_str, "--g 1.3")
        self.assertEqual(int_str, "--int 1.4")
        self.assertEqual(mass_str, "--mass 1.5")
        self.assertEqual(chem_str, "--chem 1.0 2.0 3.0")

        dirname = "g_1.3_int_1.4_mass_1.5_chem_1.0_2.0"
        chem_str = build_slurm.format_arg(dirname, "chem")
        self.assertEqual(chem_str, "--chem 1.0 2.0")

        dirname = "g_1.3_int_1.4_mass_1.5_chem_1.0"
        chem_str = build_slurm.format_arg(dirname, "chem")
        self.assertEqual(chem_str, "--chem 1.0")

        dirname = "g_1.3_int_1.4_mass_1_chem_1.0"
        mass_str = build_slurm.format_arg(dirname, "mass")
        self.assertEqual(mass_str, "--mass 1")

        dirname = "g_-1.3_int_1.4_mass_1_chem_1.0_-2.0"
        elmag_str = build_slurm.format_arg(dirname, "g")
        mass_str = build_slurm.format_arg(dirname, "mass")
        chem_str = build_slurm.format_arg(dirname, "chem")
        self.assertEqual(elmag_str, "--g -1.3")
        self.assertEqual(mass_str, "--mass 1")
        self.assertEqual(chem_str, "--chem 1.0 -2.0")

    def test_build_slurm_gf_extraction(self):
        """Test that arguments are correctly extracted from folder names for gauge fixing."""

        dirname = "L_2_g_1.3_int_1.4_mass_1.5_chem_1.0_2.0_3.0_gf_T"
        size_str = build_slurm.format_arg(dirname, "L")
        elmag_str = build_slurm.format_arg(dirname, "g")
        int_str = build_slurm.format_arg(dirname, "int")
        mass_str = build_slurm.format_arg(dirname, "mass")
        chem_str = build_slurm.format_arg(dirname, "chem")
        self.assertEqual(size_str, "--L 2")
        self.assertEqual(elmag_str, "--g 1.3")
        self.assertEqual(int_str, "--int 1.4")
        self.assertEqual(mass_str, "--mass 1.5")
        self.assertEqual(chem_str, "--chem 1.0 2.0 3.0")
        gf_str = build_slurm.format_arg(dirname, "gf")
        self.assertEqual(gf_str, "--gauge_fixing")

        dirname = "L_2_g_1.3_int_1.4_mass_1.5_chem_1.0_2.0_3.0_gf_F"
        gf_str = build_slurm.format_arg(dirname, "gf")
        self.assertEqual(gf_str, "")

        dirname = "L_2_g_1.3_int_1.4_mass_1.5_chem_1.0_2.0_3.0_gf_c"
        gf_str = build_slurm.format_arg(dirname, "gf")
        self.assertEqual(gf_str, "--gauge_fixing -2")
