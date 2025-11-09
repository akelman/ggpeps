import unittest

from tools import build_slurm


class TestTools(unittest.TestCase):

    def setUp(self):
        pass

    def test_build_slurm_arg_extraction(self):
        """Test that arguments are correctly extracted from folder names."""

        dirname = "L_2_g_1.3_int_1.4_mass_1.5_chem_1.0_2.0_3.0"
        size_str = build_slurm.folder2arg(dirname, "L")
        elmag_str = build_slurm.folder2arg(dirname, "g")
        int_str = build_slurm.folder2arg(dirname, "int")
        mass_str = build_slurm.folder2arg(dirname, "mass")
        chem_str = build_slurm.folder2arg(dirname, "chem")
        self.assertEqual(size_str, "--L 2")
        self.assertEqual(elmag_str, "--g 1.3")
        self.assertEqual(int_str, "--int 1.4")
        self.assertEqual(mass_str, "--mass 1.5")
        self.assertEqual(chem_str, "--chem 1.0 2.0 3.0")

        dirname = "g_1.3_int_1.4_mass_1.5_chem_1.0_2.0"
        chem_str = build_slurm.folder2arg(dirname, "chem")
        self.assertEqual(chem_str, "--chem 1.0 2.0")

        dirname = "g_1.3_int_1.4_mass_1.5_chem_1.0"
        chem_str = build_slurm.folder2arg(dirname, "chem")
        self.assertEqual(chem_str, "--chem 1.0")

        dirname = "g_1.3_int_1.4_mass_1_chem_1.0"
        mass_str = build_slurm.folder2arg(dirname, "mass")
        self.assertEqual(mass_str, "--mass 1")

        dirname = "g_-1.3_int_1.4_mass_1_chem_1.0_-2.0"
        elmag_str = build_slurm.folder2arg(dirname, "g")
        mass_str = build_slurm.folder2arg(dirname, "mass")
        chem_str = build_slurm.folder2arg(dirname, "chem")
        self.assertEqual(elmag_str, "--g -1.3")
        self.assertEqual(mass_str, "--mass 1")
        self.assertEqual(chem_str, "--chem 1.0 -2.0")
