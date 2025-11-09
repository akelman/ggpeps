import unittest

from tools import build_slurm


class TestTools(unittest.TestCase):

    def setUp(self):
        pass

    def test_build_slurm_arg_extraction(self):
        """Test that arguments are correctly extracted from folder names."""

        dirname = "g_1.3_int_1.4_mass_1.5_chem_1.0_2.0_3.0"
        mass_str = build_slurm.folder2arg(dirname, "mass")
        chem_str = build_slurm.folder2arg(dirname, "chem")
        self.assertEqual(mass_str, "--mass 1.5")
        # self.assertEqual(chem_str, "--chem 1.0 2.0 3.0")
