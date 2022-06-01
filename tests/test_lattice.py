import unittest
from ggpeps import lattice
import numpy as np


class TestLattice(unittest.TestCase):

    def setUp(self):
        self.lat2d=lattice.Lattice2D(8,8)
        self.lat3d=lattice.Lattice3D(8,8,8)

    def test_ind2coord_2d(self):
        ref=(3,4)
        ind=self.lat2d.coord2ind(ref)
        coord=self.lat2d.ind2coord(ind)
        self.assertEqual(ref,coord)

    def test_ind2coord_dir_2d(self):
        coord_ref=(2,3)
        for dir_ref in [lattice.Direction.X, lattice.Direction.Y]:
            ind=self.lat2d.coord2ind_dir(coord_ref,dir_ref)
            coord,dir=self.lat2d.ind2coord_dir(ind)
            self.assertEqual(coord_ref,coord)
            self.assertEqual(dir_ref,dir)

    def test_wilson_loop_1x1(self):
        ref=[(((0,0),lattice.Direction.X),False),
            (((1,0),lattice.Direction.Y),False),
            (((0,1),lattice.Direction.X),True),
            (((0,0),lattice.Direction.Y),True)
            ]
        path=self.lat2d.generate_wilson_loop((0,0),(1,1),False)
        self.assertEqual(ref,path)

    def test_wilson_loop_2x1(self):
        ref=[(((0,0),lattice.Direction.X),False),
            (((1,0),lattice.Direction.X),False),
            (((2,0),lattice.Direction.Y),False),
            (((1,1),lattice.Direction.X),True),
            (((0,1),lattice.Direction.X),True),
            (((0,0),lattice.Direction.Y),True)
            ]
        path=self.lat2d.generate_wilson_loop((0,0),(2,1),False)
        self.assertEqual(ref,path)

    def test_wilson_loop_1x1_periodic(self):
        ref=[(((7,7),lattice.Direction.X),False),
            (((0,7),lattice.Direction.Y),False),
            (((7,0),lattice.Direction.X),True),
            (((7,7),lattice.Direction.Y),True)
            ]
        path=self.lat2d.generate_wilson_loop((7,7),(1,1),False)
        self.assertEqual(ref,path)

    def test_polyakov_loop_hor(self):
        ref=[(((0,0),lattice.Direction.X),False),
            (((1,0),lattice.Direction.X),False),
            (((2,0),lattice.Direction.X),False),
            (((3,0),lattice.Direction.X),False),
            (((4,0),lattice.Direction.X),False),
            (((5,0),lattice.Direction.X),False),
            (((6,0),lattice.Direction.X),False),
            (((7,0),lattice.Direction.X),False),
            ]
        path=self.lat2d.generate_polyakov_loop((0,0),lattice.Direction.X, use_indices=False)
        self.assertEqual(ref,path)

    def test_polyakov_loop_vert(self):
        ref=[(((0,0),lattice.Direction.Y),False),
            (((0,1),lattice.Direction.Y),False),
            (((0,2),lattice.Direction.Y),False),
            (((0,3),lattice.Direction.Y),False),
            (((0,4),lattice.Direction.Y),False),
            (((0,5),lattice.Direction.Y),False),
            (((0,6),lattice.Direction.Y),False),
            (((0,7),lattice.Direction.Y),False),
            ]
        path=self.lat2d.generate_polyakov_loop((0,0),lattice.Direction.Y, use_indices=False)
        self.assertEqual(ref,path)

    def test_2d_covering(self):
        nx=13
        ny=7
        lat=lattice.Lattice2D(nx,ny)
        linkvec=np.zeros(lat.nlinks)
        sitevec=np.zeros(lat.size)
        for x in range(nx):
            for y in range(ny):
                ind_site=lat.coord2ind((x,y))
                ind_link_x=lat.coord2ind_dir((x,y),lattice.Direction.X)
                ind_link_y=lat.coord2ind_dir((x,y),lattice.Direction.Y)
                linkvec[ind_link_x]=1
                linkvec[ind_link_y]=1
                sitevec[ind_site]=1
        self.assertEqual(np.sum(linkvec),lat.nlinks)
        self.assertEqual(np.sum(sitevec),lat.size)

    def test_ind2coord_3d(self):
        ref=(3,4,2)
        ind=self.lat3d.coord2ind(ref)
        coord=self.lat3d.ind2coord(ind)
        self.assertEqual(ref,coord)

    def test_ind2coord_dir_3d(self):
        coord_ref=(2,3,3)
        for dir_ref in lattice.Direction:
            ind=self.lat3d.coord2ind_dir(coord_ref,dir_ref)
            coord,dir=self.lat3d.ind2coord_dir(ind)
            self.assertEqual(coord_ref,coord)
            self.assertEqual(dir_ref,dir)
    
    def test_link_based_mode_order(self):
        # TODO: for now, this only tests the case with 1 copy 
        # (to extend the test, the tested function needs to be updated)

        lat = lattice.Lattice2D(2,3)
        modes_calc = lat.get_link_based_mode_order()

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