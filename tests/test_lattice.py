import unittest

import numpy as np

from ggpeps import lattice
from ggpeps.lattice import Direction


class TestLattice(unittest.TestCase):

    def setUp(self):
        self.lat2d = lattice.Lattice2D(8, 8)
        self.lat3d = lattice.Lattice3D(8, 8, 8)

    def test_ind2coord_2d(self):
        ref = (3, 4)
        ind = self.lat2d.coord2ind(ref)
        coord = self.lat2d.ind2coord(ind)
        self.assertEqual(ref, coord)

    def test_ind2coord_dir_2d(self):
        coord_ref = (2, 3)
        for dir_ref in [lattice.Direction.X, lattice.Direction.Y]:
            ind = self.lat2d.coord2ind_dir(coord_ref, dir_ref)
            coord, dir = self.lat2d.ind2coord_dir(ind)
            self.assertEqual(coord_ref, coord)
            self.assertEqual(dir_ref, dir)

    def test_coord2ind_dir(self):
        # test for negative coordinates
        self.assertTrue(
            self.lat2d.coord2ind_dir((-1, 0), Direction.X) == self.lat2d.nx - 1
        )
        self.assertTrue(
            self.lat2d.coord2ind_dir((0, -1), Direction.Y)
            == self.lat2d.nx * self.lat2d.ny + self.lat2d.ny - 1
        )

    def test_wilson_loop_1x1(self):
        ref = [
            (((0, 0), lattice.Direction.X), False),
            (((1, 0), lattice.Direction.Y), False),
            (((0, 1), lattice.Direction.X), True),
            (((0, 0), lattice.Direction.Y), True),
        ]
        path = self.lat2d.generate_wilson_loop((0, 0), (1, 1), False)
        self.assertEqual(ref, path)

    def test_wilson_loop_2x1(self):
        ref = [
            (((0, 0), lattice.Direction.X), False),
            (((1, 0), lattice.Direction.X), False),
            (((2, 0), lattice.Direction.Y), False),
            (((1, 1), lattice.Direction.X), True),
            (((0, 1), lattice.Direction.X), True),
            (((0, 0), lattice.Direction.Y), True),
        ]
        path = self.lat2d.generate_wilson_loop((0, 0), (2, 1), False)
        self.assertEqual(ref, path)

    def test_wilson_loop_1x1_periodic(self):
        ref = [
            (((7, 7), lattice.Direction.X), False),
            (((0, 7), lattice.Direction.Y), False),
            (((7, 0), lattice.Direction.X), True),
            (((7, 7), lattice.Direction.Y), True),
        ]
        path = self.lat2d.generate_wilson_loop((7, 7), (1, 1), False)
        self.assertEqual(ref, path)

    def test_wilson_loop_generation_2x2_lattice(self):
        """Test that all expected wilson loops are generated for a 2x2 lattice."""
        lat = lattice.Lattice2D(2, 2)
        refs = [
            [
                (((0, 0), lattice.Direction.X), False),
                (((1, 0), lattice.Direction.Y), False),
                (((0, 1), lattice.Direction.X), True),
                (((0, 0), lattice.Direction.Y), True),
            ]
        ]
        paths = lat.generate_all_wilson_loops((0, 0), use_indices=False)
        self.assertEqual(refs, paths)

    def test_wilson_loop_generation_4x4_lattice(self):
        """Test that all expected wilson loops are generated for a 4x4 lattice.
        Note that the order of the loops is set to match the expected output from
        Lattice2D.generate_all_wilson_loops()
        """
        lat = lattice.Lattice2D(4, 4)
        refs = [
            [
                (((0, 0), lattice.Direction.X), False),
                (((1, 0), lattice.Direction.Y), False),
                (((0, 1), lattice.Direction.X), True),
                (((0, 0), lattice.Direction.Y), True),
            ],  # 1x1 loop
            [
                (((0, 0), lattice.Direction.X), False),
                (((1, 0), lattice.Direction.X), False),
                (((2, 0), lattice.Direction.Y), False),
                (((1, 1), lattice.Direction.X), True),
                (((0, 1), lattice.Direction.X), True),
                (((0, 0), lattice.Direction.Y), True),
            ],  # 2x1 loop
            [
                (((0, 0), lattice.Direction.X), False),
                (((1, 0), lattice.Direction.X), False),
                (((2, 0), lattice.Direction.Y), False),
                (((2, 1), lattice.Direction.Y), False),
                (((1, 2), lattice.Direction.X), True),
                (((0, 2), lattice.Direction.X), True),
                (((0, 1), lattice.Direction.Y), True),
                (((0, 0), lattice.Direction.Y), True),
            ],  # 2x2 loop
        ]
        paths = lat.generate_all_wilson_loops((0, 0), use_indices=False)
        self.assertEqual(refs, paths)

    def test_wilson_loop_generation_5x5_lattice(self):
        """Test that all expected wilson loops are generated for a 5x5 lattice.
        Note that the order of the loops is set to match the expected output from
        Lattice2D.generate_all_wilson_loops()
        """
        lat = lattice.Lattice2D(5, 5)
        refs = [
            [
                (((0, 0), lattice.Direction.X), False),
                (((1, 0), lattice.Direction.Y), False),
                (((0, 1), lattice.Direction.X), True),
                (((0, 0), lattice.Direction.Y), True),
            ],  # 1x1 loop
            [
                (((0, 0), lattice.Direction.X), False),
                (((1, 0), lattice.Direction.X), False),
                (((2, 0), lattice.Direction.Y), False),
                (((1, 1), lattice.Direction.X), True),
                (((0, 1), lattice.Direction.X), True),
                (((0, 0), lattice.Direction.Y), True),
            ],  # 2x1 loop
            [
                (((0, 0), lattice.Direction.X), False),
                (((1, 0), lattice.Direction.X), False),
                (((2, 0), lattice.Direction.Y), False),
                (((2, 1), lattice.Direction.Y), False),
                (((1, 2), lattice.Direction.X), True),
                (((0, 2), lattice.Direction.X), True),
                (((0, 1), lattice.Direction.Y), True),
                (((0, 0), lattice.Direction.Y), True),
            ],  # 2x2 loop
        ]

        """ # Not included by default
            [   (((0,0),lattice.Direction.X),False),
                (((1,0),lattice.Direction.Y),False),
                (((1,1),lattice.Direction.Y),False),
                (((0,2),lattice.Direction.X),True),
                (((0,1),lattice.Direction.Y),True),
                (((0,0),lattice.Direction.Y),True)     ], # 1x2 loop
        """

        paths = lat.generate_all_wilson_loops((0, 0), use_indices=False)
        self.assertEqual(refs, paths)

    def test_polyakov_loop_hor(self):
        ref = [
            (((0, 0), lattice.Direction.X), False),
            (((1, 0), lattice.Direction.X), False),
            (((2, 0), lattice.Direction.X), False),
            (((3, 0), lattice.Direction.X), False),
            (((4, 0), lattice.Direction.X), False),
            (((5, 0), lattice.Direction.X), False),
            (((6, 0), lattice.Direction.X), False),
            (((7, 0), lattice.Direction.X), False),
        ]
        path = self.lat2d.generate_polyakov_loop(
            (0, 0), lattice.Direction.X, use_indices=False
        )
        self.assertEqual(ref, path)

    def test_polyakov_loop_vert(self):
        ref = [
            (((0, 0), lattice.Direction.Y), False),
            (((0, 1), lattice.Direction.Y), False),
            (((0, 2), lattice.Direction.Y), False),
            (((0, 3), lattice.Direction.Y), False),
            (((0, 4), lattice.Direction.Y), False),
            (((0, 5), lattice.Direction.Y), False),
            (((0, 6), lattice.Direction.Y), False),
            (((0, 7), lattice.Direction.Y), False),
        ]
        path = self.lat2d.generate_polyakov_loop(
            (0, 0), lattice.Direction.Y, use_indices=False
        )
        self.assertEqual(ref, path)

    def test_2d_covering(self):
        nx = 13
        ny = 7
        lat = lattice.Lattice2D(nx, ny)
        linkvec = np.zeros(lat.nlinks)
        sitevec = np.zeros(lat.size)
        for x in range(nx):
            for y in range(ny):
                ind_site = lat.coord2ind((x, y))
                ind_link_x = lat.coord2ind_dir((x, y), lattice.Direction.X)
                ind_link_y = lat.coord2ind_dir((x, y), lattice.Direction.Y)
                linkvec[ind_link_x] = 1
                linkvec[ind_link_y] = 1
                sitevec[ind_site] = 1
        self.assertEqual(np.sum(linkvec), lat.nlinks)
        self.assertEqual(np.sum(sitevec), lat.size)

    def test_ind2coord_3d(self):
        ref = (3, 4, 2)
        ind = self.lat3d.coord2ind(ref)
        coord = self.lat3d.ind2coord(ind)
        self.assertEqual(ref, coord)

    def test_ind2coord_dir_3d(self):
        coord_ref = (2, 3, 3)
        for dir_ref in lattice.Direction:
            ind = self.lat3d.coord2ind_dir(coord_ref, dir_ref)
            coord, dir = self.lat3d.ind2coord_dir(ind)
            self.assertEqual(coord_ref, coord)
            self.assertEqual(dir_ref, dir)

    def test_maximal_tree_generation(self):
        """Ensure that maximal trees are generated correctly"""
        lat_2x2 = lattice.Lattice2D(2, 2, -1)
        lat_4x4 = lattice.Lattice2D(4, 4, -1)
        tree2_expected = {0, 2, 4}
        tree4_expected = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18}
        self.assertEqual(tree2_expected, set(lat_2x2.fixed_tree))
        self.assertEqual(tree4_expected, set(lat_4x4.fixed_tree))

    def test_complementary_maximal_tree_generation(self):
        """Ensure that complement of the maximal tree (all the links which are not
        in the tree) is generated correctly."""
        lat_2x2 = lattice.Lattice2D(2, 2, -1)
        lat_4x4 = lattice.Lattice2D(4, 4, -1)
        comp_tree2_expected = {1, 3, 5, 6, 7}
        comp_tree4_expected = {
            3,
            7,
            11,
            15,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
        }
        self.assertEqual(comp_tree2_expected, set(lat_2x2.comp_tree))
        self.assertEqual(comp_tree4_expected, set(lat_4x4.comp_tree))

    def test_tree_againt_complement(self):
        """Check that the maximal tree and its complement are disjoint"""
        self.assertTrue(
            set(self.lat2d.fixed_tree).isdisjoint(set(self.lat2d.comp_tree))
        )
    
    def test_rows_tree(self):
        """Check that the rows tree is generated correctly for 4x4 and 2x2 lattices."""
        lat2x2 = lattice.Lattice2D(2, 2, 1)
        lat4x4 = lattice.Lattice2D(4, 4, 3)
        expected_rows_tree_4by4 = {0, 1, 2, 4, 5, 6, 8, 9,10}
        expected_rows_tree_2by2 = {0}
        self.assertEqual(expected_rows_tree_4by4, set(lat4x4.fixed_tree))
        self.assertEqual(expected_rows_tree_2by2, set(lat2x2.fixed_tree))
