import unittest
import numpy as np
from ggpeps import utils
from ggpeps.modearray import ModeArray

class TestModeArray(unittest.TestCase):

    def setUp(self):
        # 1D test cases
        self.a_1d = np.arange(5).view(ModeArray)
        self.a_1d.modes = [["1","2","3","4","5"]]
        self.b_1d = (2* np.arange(5)).view(ModeArray)
        self.b_1d.modes = [["1","2","3","4","5"]]
        self.c_1d = (1+np.arange(5)).view(ModeArray)
        self.c_1d.modes = [["1","9","3","4","5"]]

        # 2D test cases
        self.a_2d = np.random.rand(3,3).view(ModeArray)
        self.a_2d.modes = [["1","2","3"],["1","2","3"]]
        self.b_2d = np.random.rand(3,3).view(ModeArray)
        self.b_2d.modes = [["1","2","3"],["1","2","3"]]
        self.c_2d = np.random.rand(3,3).view(ModeArray)
        self.c_2d.modes = [["1","9","3"],["1","2","3"]]
        self.d_2d = np.random.rand(3,3).view(ModeArray)
        self.d_2d.modes = [["1","2","3"],["1","9","3"]]

    def test_duplicate_mode(self):
        arr_1d = np.arange(2).view(ModeArray)
        arr_2d = np.random.rand(4,4).view(ModeArray)
        with self.assertRaises(ValueError):
            arr_1d.modes=[["1","1"]]
        with self.assertRaises(ValueError):
            arr_2d.modes=[["1","1","2","3"],["1","4","2","3"]]

    def test_number_modes(self):
        arr = np.arange(3).view(ModeArray)
        arr_2d = np.random.rand(4,4).view(ModeArray)
        with self.assertRaises(ValueError):
            arr.modes=[["1","2"]]
        with self.assertRaises(ValueError):
            arr.modes=[["1","2","3","4"]]
        with self.assertRaises(ValueError):
            arr_2d.modes = [["1","2"],["1","2","3","4"]]
        with self.assertRaises(ValueError):
            arr_2d.modes = [["1","2","3","4"],["1","2"]]
        with self.assertRaises(ValueError):
            arr_2d.modes = [["1","3"],["1","3","4"]]

    def test_number_modes(self):
        arr = np.arange(3).view(ModeArray)
        with self.assertRaises(ValueError):
            arr.modes=[["1","2"]]
        with self.assertRaises(ValueError):
            arr.modes=[["1","2","3","4"]]
    
    def test_addition_scalar(self):
        d_1d = self.a_1d + 4
        ref_1d = np.asarray(self.a_1d)+4
        d_2d = self.a_2d + 4
        ref_2d = np.asarray(self.a_2d)+4
        self.assertTrue(np.allclose(ref_1d, np.asarray(d_1d)))
        self.assertTrue(np.allclose(ref_2d, np.asarray(d_2d)))
        self.assertTrue(d_1d.modes == self.a_1d.modes)
        self.assertTrue(d_2d.modes == self.a_2d.modes)

    def test_addition(self):
        d_1d = self.a_1d + self.b_1d
        ref_1d = np.asarray(self.a_1d)+np.asarray(self.b_1d)
        d_2d = self.a_2d + self.b_2d
        ref_2d = np.asarray(self.a_2d)+np.asarray(self.b_2d)
        self.assertTrue(np.allclose(ref_1d, np.asarray(d_1d)))
        self.assertTrue(np.allclose(ref_2d, np.asarray(d_2d)))
        self.assertTrue(d_1d.modes == self.a_1d.modes)
        self.assertTrue(d_2d.modes == self.a_2d.modes)
        with self.assertRaises(ValueError):
            e = self.a_1d + self.c_1d
        with self.assertRaises(ValueError):
            e = self.a_2d + self.c_2d

    def test_multiplication_scalar(self):
        d_1d = self.a_1d * 2
        ref_1d = np.asarray(self.a_1d)*2

        d_2d = self.a_2d * 2
        ref_2d = np.asarray(self.a_2d)*2

        self.assertTrue(np.allclose(ref_1d,np.asarray(d_1d)))
        self.assertTrue(np.allclose(ref_2d,np.asarray(d_2d)))
        self.assertTrue(d_1d.modes == self.a_1d.modes)
        self.assertTrue(d_2d.modes == self.a_2d.modes)

    def test_multiplication(self):
        d_1d = self.a_1d * self.b_1d
        ref_1d = np.asarray(self.a_1d)*np.asarray(self.b_1d)

        d_2d = self.a_2d * self.b_2d
        ref_2d = np.asarray(self.a_2d)*np.asarray(self.b_2d)

        self.assertTrue(np.allclose(ref_1d, np.asarray(d_1d)))
        self.assertTrue(np.allclose(ref_2d, np.asarray(d_2d)))
        self.assertTrue(d_1d.modes == self.a_1d.modes)
        self.assertTrue(d_2d.modes == self.a_2d.modes)

        with self.assertRaises(ValueError):
            e = self.a_1d * self.c_1d
        with self.assertRaises(ValueError):
            e = self.a_2d * self.c_2d

    def test_matrix_multiplication(self):
        d_2d_ab = self.a_2d @ self.b_2d
        d_2d_ad = self.a_2d @ self.d_2d
        ref_2d_ab = np.asarray(self.a_2d)@np.asarray(self.b_2d)
        ref_2d_ad = np.asarray(self.a_2d)@np.asarray(self.d_2d)

        self.assertTrue(np.allclose(ref_2d_ab, np.asarray(d_2d_ab)))
        self.assertTrue(np.allclose(ref_2d_ad,np.asarray(d_2d_ad)))

        with self.assertRaises(ValueError):
            e = self.a_2d @ self.c_2d

    def test_matmul_output_modes(self):
        d_2d_ab = self.a_2d @ self.b_2d
        d_2d_ad = self.a_2d @ self.d_2d
        ref_2d_ab = [self.a_2d.modes[0], self.b_2d.modes[1]]
        ref_2d_ad = [self.a_2d.modes[0], self.d_2d.modes[1]]

        self.assertTrue(d_2d_ab.modes == ref_2d_ab)
        self.assertTrue(d_2d_ad.modes == ref_2d_ad)

    def test_transpose(self):
        dest = np.transpose(self.a_2d)
        self.assertTrue(dest.modes == [self.a_2d.modes[1],self.a_2d.modes[0]])
        self.assertTrue(np.allclose(np.transpose(self.a_2d),np.asarray(dest)))

    
    def test_gen_permutation_matrix(self):
        from ggpeps.modearray import generate_permutation_matrix
        old_modes = ["1","2","3"]
        new_modes = ["3","2","1"]
        perm = generate_permutation_matrix(old_modes,new_modes)
        self.assertTrue(utils.is_permutation(perm))
    
    def test_permute_3x3_rows(self):
        old_modes = ["1","2","3"]
        new_modes = ["3","2","1"]
        arr = ModeArray(np.array([[1,1,1],[2,2,2],[3,3,3]]),[old_modes,old_modes])
        dest = arr.permute([new_modes,old_modes])

        ref_modes = [new_modes, old_modes]
        self.assertTrue(dest.modes == ref_modes)
        self.assertTrue(np.allclose(np.asarray(dest), np.asarray( [[3, 3, 3], [2, 2, 2], [1, 1, 1]])))

    def test_permute_3x3_cols(self):
        old_modes = ["1","2","3"]
        new_modes = ["3","2","1"]
        arr = ModeArray(np.array([[1,2,3],[1,2,3],[1,2,3]]),[old_modes,old_modes])
        dest = arr.permute([old_modes,new_modes])

        ref_modes = [old_modes, new_modes]
        self.assertTrue(list(dest.modes) == ref_modes)
        self.assertTrue(np.allclose(np.asarray(dest), np.asarray( [[3, 2, 1], [3, 2, 1], [3, 2, 1]])))