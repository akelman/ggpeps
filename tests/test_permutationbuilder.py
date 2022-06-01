import unittest
import numpy as np
from ggpeps import lattice, utils

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

    def test_2copy_3x2(self):
        permbuilder_3x2 = lattice.PermutationBuilderGMS2D2C(
            lattice.Lattice2D(3, 2), 1)
        permutation = permbuilder_3x2.perm()
        self.assertTrue(utils.is_permutation(permutation))


    def test_2copies_2x3(self):
        permbuilder_2x3 = lattice.PermutationBuilderGMS2D2C(
            lattice.Lattice2D(2, 3), 1)
        permutation = permbuilder_2x3.perm()
        self.assertTrue(utils.is_permutation(permutation))

    def test_2copies_2x2(self):
        permbuilder_2x2 = lattice.PermutationBuilderGMS2D2C(
            lattice.Lattice2D(2, 2), 1)
        permutation = permbuilder_2x2.perm()
        self.assertTrue(utils.is_permutation(permutation))
