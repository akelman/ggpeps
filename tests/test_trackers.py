import unittest
import numpy as np

from ggpeps import utils


# ======================= WoodburyInverter Test =========================================
class TestWoodburyInverter(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.ident = np.eye(self.n)
        self.wi = utils.WoodburyInverter(self.ident)

    def test_identity(self):
        inv_wb = self.wi.update(self.wi.inv(), self.ident, self.ident, self.ident)
        inv = np.linalg.inv(2 * self.ident)
        self.assertTrue(np.allclose(inv, inv_wb))

    def test_identity_incr(self):
        for _ in range(10):
            update = 0.1 * self.ident
            self.wi.update(self.wi.inv(), self.ident, update, self.ident)
        inv_wb = self.wi.inv()
        inv = np.linalg.inv(2 * self.ident)
        self.assertTrue(np.allclose(inv, inv_wb))

    def test_random_incr(self):
        mat = np.random.rand(self.n, self.n)
        wi = utils.WoodburyInverter(mat)
        for _ in range(100):
            incr = np.random.rand(self.n, self.n)
            wi.update(wi.inv(), self.ident, incr, self.ident)
            mat += incr
        inv_wb = wi.inv()
        inv = np.linalg.inv(mat)
        self.assertTrue(np.allclose(inv, inv_wb))

    def test_pos_update(self):
        n = 10
        n_up = 2
        mat = np.random.rand(n, n)
        update_mat = np.random.rand(n_up, n_up)
        padval = n - n_up
        update_padded = np.pad(update_mat, [(0, padval), (0, padval)], "constant", constant_values=(0, 0))
        wi = utils.WoodburyInverter(mat)
        inv_wb = wi.update_index(update_mat, 0, 0)
        inv = np.linalg.inv(mat + update_padded)
        self.assertTrue(np.allclose(inv, inv_wb))

    def test_zero_incr(self):
        mat = np.random.rand(self.n, self.n)
        wi = utils.WoodburyInverter(mat)
        zero = np.zeros((self.n, self.n))
        inv_wb = wi.update(wi.inv(), zero, 0, 0)
        inv = np.linalg.inv(mat)
        self.assertTrue(np.allclose(inv, inv_wb))


# ======================= IncDeterminant Test =========================================
class TestIncDeterminant(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.ident = np.eye(self.n)
        self.incdet = utils.IncDeterminant(self.ident)

    def test_identity(self):
        detval = self.incdet.update(self.ident, self.ident, self.ident, self.ident)
        detval_direct = np.linalg.det(2 * self.ident)
        self.assertAlmostEqual(detval, detval_direct)

    def test_identity_incr(self):
        track = np.copy(self.ident)
        for _ in range(10):
            self.incdet.update(np.linalg.inv(track), self.ident, 0.1 * self.ident, self.ident)
            track += 0.1 * self.ident
        detval = self.incdet.det()
        detval_direct = np.linalg.det(2 * self.ident)
        self.assertAlmostEqual(detval, detval_direct)

    def test_random_incr(self):
        mat = np.random.rand(self.n, self.n)
        incdet = utils.IncDeterminant(mat)
        for _ in range(100):
            incr = np.random.rand(self.n, self.n)
            incdet.update(np.linalg.inv(mat), self.ident, incr, self.ident)
            mat += incr
        detval = incdet.det()
        detval_direct = np.linalg.det(mat)
        self.assertAlmostEqual((detval - detval_direct) / detval_direct, 0)

    def test_zero_incr(self):
        mat = np.random.rand(self.n, self.n)
        zero = np.zeros((self.n, self.n))
        incdet = utils.IncDeterminant(mat)
        incdet.update(np.linalg.inv(mat), self.ident, zero, self.ident)
        detval = incdet.det()
        detval_direct = np.linalg.det(mat)
        self.assertAlmostEqual(detval, detval_direct)


# ======================= IncLogAbsDeterminant Test =======================


def generate_pos_def_matrix(n):
    rand = np.random.rand(n, n)
    rand = 0.5 * (rand + np.transpose(rand))
    rand += n * np.eye(n)
    return rand


class TestIncLogDeterminant(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.ident = np.eye(self.n)
        self.incdet = utils.IncLogAbsDeterminant(self.ident)

    def test_identity(self):
        detval = self.incdet.update(self.ident, self.ident, self.ident, self.ident)
        _, detval_direct = np.linalg.slogdet(2 * self.ident)
        self.assertAlmostEqual(detval, detval_direct)

    def test_identity_incr(self):
        track = np.copy(self.ident)
        for _ in range(10):
            self.incdet.update(np.linalg.inv(track), self.ident, 0.1 * self.ident, self.ident)
            track += 0.1 * self.ident
        detval = self.incdet.det()
        _, detval_direct = np.linalg.slogdet(2 * self.ident)
        self.assertAlmostEqual(detval, detval_direct)

    def test_random_incr(self):
        mat = generate_pos_def_matrix(self.n)
        incdet = utils.IncLogAbsDeterminant(mat)
        for _ in range(100):
            incr = generate_pos_def_matrix(self.n)
            incdet.update(np.linalg.inv(mat), self.ident, incr, self.ident)
            mat += incr
        detval = incdet.det()
        _, detval_direct = np.linalg.slogdet(mat)
        self.assertAlmostEqual(detval, detval_direct)

    def test_zero_incr(self):
        mat = generate_pos_def_matrix(self.n)
        zero = np.zeros((self.n, self.n))
        incdet = utils.IncLogAbsDeterminant(mat)
        incdet.update(np.linalg.inv(mat), self.ident, zero, self.ident)
        detval = incdet.det()
        _, detval_direct = np.linalg.slogdet(mat)
        self.assertAlmostEqual(detval, detval_direct)

    def test_revert(self):
        mat = generate_pos_def_matrix(self.n)
        incr = generate_pos_def_matrix(self.n)
        incdet = utils.IncLogAbsDeterminant(mat)
        incdet.update(np.linalg.inv(mat), self.ident, incr, self.ident)
        incdet.update(np.linalg.inv(mat + incr), self.ident, -incr, self.ident)
        detval = incdet.det()
        _, detval_direct = np.linalg.slogdet(mat)
        self.assertAlmostEqual(detval, detval_direct)
