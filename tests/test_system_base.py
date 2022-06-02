import unittest
import numpy as np
from ggpeps import lattice, system


class TestSystemBase(unittest.TestCase):

    def setUp(self):
        lat = lattice.Lattice2D(2,3)
        paramvec = [[0.3, 0.5, 0.8]]
        cfg = system.Z2System2DConfig(lat, 0, 0, 0)
        cfg.paramvec = paramvec
        self.system_z2 = system.Z2System2D(cfg)
    
    def test_link_based_mode_order(self):
        # TODO: for now, this only tests the case with 1 copy 
        # (to extend the test, the tested function needs to be updated)

        modes_calc = self.system_z2.get_link_based_mode_order()

        # The following explicit mode ordering was found using pen and paper (well, metaphorically)
        # <mode_letter:maj mode>_<copy>_<link_id>
        modes_manual = [    "l1_1_0", "l2_1_0", "r1_1_0", "r2_1_0",
                            "l1_1_1", "l2_1_1", "r1_1_1", "r2_1_1",
                            "l1_1_2", "l2_1_2", "r1_1_2", "r2_1_2",
                            "l1_1_3", "l2_1_3", "r1_1_3", "r2_1_3",
                            "l1_1_4", "l2_1_4", "r1_1_4", "r2_1_4", 
                            "l1_1_5", "l2_1_5", "r1_1_5", "r2_1_5",
                            "d1_1_6", "d2_1_6", "u1_1_6", "u2_1_6",
                            "d1_1_7", "d2_1_7", "u1_1_7", "u2_1_7",
                            "d1_1_8", "d2_1_8", "u1_1_8", "u2_1_8",
                            "d1_1_9", "d2_1_9", "u1_1_9", "u2_1_9",
                            "d1_1_10", "d2_1_10", "u1_1_10", "u2_1_10",
                            "d1_1_11", "d2_1_11", "u1_1_11", "u2_1_11" ]
        
        self.assertTrue( len(modes_calc) == len(modes_manual))
        for k in range( len(modes_calc) ):
            self.assertTrue( modes_calc[k] == modes_manual[k] )
    
    def test_site_based_mode_order(self):
        # TODO: for now, this only tests the case with 1 copy 
        # (to extend the test, the tested function needs to be updated)

        modes_calc = self.system_z2.get_site_based_mode_order()

        # <mode_letter:maj mode>_<copy>_<link_id>
        modes_manual = [    "l1_1_1", "l2_1_1", "r1_1_0", "r2_1_0", # each two lines is one site
                            "d1_1_8", "d2_1_8", "u1_1_6", "u2_1_6",
                            "l1_1_0", "l2_1_0", "r1_1_1", "r2_1_1",
                            "d1_1_11", "d2_1_11", "u1_1_9", "u2_1_9",
                            "l1_1_3", "l2_1_3", "r1_1_2", "r2_1_2",
                            "d1_1_6", "d2_1_6", "u1_1_7", "u2_1_7",
                            "l1_1_2", "l2_1_2", "r1_1_3", "r2_1_3",
                            "d1_1_9", "d2_1_9", "u1_1_10", "u2_1_10",
                            "l1_1_5", "l2_1_5", "r1_1_4", "r2_1_4", 
                            "d1_1_7", "d2_1_7", "u1_1_8", "u2_1_8",
                            "l1_1_4", "l2_1_4", "r1_1_5", "r2_1_5",
                            "d1_1_10", "d2_1_10", "u1_1_11", "u2_1_11" ]
        
        self.assertTrue( len(modes_calc) == len(modes_manual))
        for k in range( len(modes_calc) ):
            self.assertTrue( modes_calc[k] == modes_manual[k] )
    
    def test_matching_permutations(self):
        # Test that permutation matrices generated using the ModeArray methods matches with previous PermutationBuilder class

        from ggpeps.modearray import generate_permutation_matrix

        modes_link_order = self.system_z2.get_link_based_mode_order()
        modes_site_order = self.system_z2.get_site_based_mode_order()
        new_perm_mat = generate_permutation_matrix(modes_site_order, modes_link_order)

        permbuilder = lattice.PermutationBuilderGMS2D(self.system_z2.cfg.lattice, 1)
        prev_perm_mat = permbuilder.perm()
        prev_perm_mat = prev_perm_mat[12:,12:] # remove the physical modes on the sites

        self.assertTrue(np.allclose(new_perm_mat.view(np.ndarray), prev_perm_mat.T)) # currently the convention is not matching, so we take the transposition of one of them
        