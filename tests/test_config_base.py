import unittest
import numpy as np

import ggpeps.system.config_base as config_base
from ggpeps import gauge
from ggpeps import system
from ggpeps import lattice

from ggpeps.lattice import Direction

# ==================== Gauged Projector Terms: Test ====================


class TestGaugedProjectorTermsForZ2(unittest.TestCase):
    def _assert_complex_close(self, a, b, rtol=1e-12, atol=1e-12):
        self.assertTrue(np.isclose(a, b, rtol=rtol, atol=atol), f"Complex values differ: {a} vs {b}")

    def _to_canonical(self, terms):
        """
        Helper method to sort terms into canonical order (Majorana anti-commutation).
        This allows the test expected values to be written in any order (e.g., (2,0) instead of (0,2)).
        """
        canonical_terms = []
        for coef, idx_tuple in terms:
            idx_list = list(idx_tuple)
            swaps = 0
            # Bubble sort to count swaps
            for i in range(len(idx_list)):
                for j in range(len(idx_list) - 1 - i):
                    if idx_list[j] > idx_list[j + 1]:
                        idx_list[j], idx_list[j + 1] = idx_list[j + 1], idx_list[j]
                        swaps += 1

            # If number of swaps is odd, flip the sign of the coefficient
            final_coef = coef * (-1 if swaps % 2 else 1)
            canonical_terms.append((final_coef, tuple(idx_list)))
        return canonical_terms

    def _assert_terms_equal(self, got_indecies, got_constant, exp_indecies, exp_constant):
        # 1. Check constant
        self._assert_complex_close(got_constant, exp_constant)

        # 2. Convert both obtained and expected terms to canonical form
        # This handles cases where 'expected' in the test might be written as (2, 0)
        # while the function correctly returns -1 * (0, 2).
        exp_canonical = self._to_canonical(exp_indecies)
        got_canonical = self._to_canonical(got_indecies)

        # 3. Check number of terms
        self.assertEqual(
            len(got_canonical),
            len(exp_canonical),
            f"Different number of monomials. Got {len(got_canonical)}, expected {len(exp_canonical)}",
        )

        # 4. Sort lists for deterministic comparison
        # Sort key: length of monomial first, then the indices tuple values
        got_sorted = sorted(got_canonical, key=lambda x: (len(x[1]), x[1]))
        exp_sorted = sorted(exp_canonical, key=lambda x: (len(x[1]), x[1]))

        # 5. Element-wise comparison
        for i, ((g_coef, g_mon), (e_coef, e_mon)) in enumerate(zip(got_sorted, exp_sorted)):
            self.assertEqual(g_mon, e_mon, f"Monomial indices differ at sorted index {i}: {g_mon} vs {e_mon}")
            self._assert_complex_close(g_coef, e_coef)

    def test_generate_gauged_projector_terms_small_cases1_Z2(self):
        # Expected outputs for ncopy in {1,2}, layer in {mixed_copies, unmixed_copies},
        # orientation in {horizontal, vertical}, group_order = 2.
        # We compare ALL terms, including those with coefficients with zero real part.
        expected = {
            (1, True, Direction.X): (
                (
                    (0.25, (0, 1)),
                    (-0.25j, (2, 0)),
                    (0.25, (2, 3)),
                    (0.25j, (3, 1)),
                ),
                0.0,
            ),
            (1, True, Direction.Y): (
                (
                    (0.25, (0, 1)),
                    (0.25j, (2, 1)),
                    (0.25, (2, 3)),
                    (0.25j, (3, 0)),
                ),
                0.0,
            ),
            (1, False, Direction.X): (
                (
                    (0.25, (0, 1)),
                    (-0.25j, (2, 0)),
                    (0.25, (2, 3)),
                    (0.25j, (3, 1)),
                ),
                0.0,
            ),
            (1, False, Direction.Y): (
                (
                    (0.25, (0, 1)),
                    (0.25j, (2, 1)),
                    (0.25, (2, 3)),
                    (0.25j, (3, 0)),
                ),
                0.0,
            ),
            (2, True, Direction.X): (
                (
                    (0.0625, (2, 3, 0, 1)),
                    (-0.0625j, (2, 3, 6, 0)),
                    (0.0625, (2, 3, 6, 7)),
                    (0.0625j, (2, 3, 7, 1)),
                    (-0.0625j, (2, 4, 0, 1)),
                    (-0.0625, (2, 4, 6, 0)),
                    (-0.0625j, (2, 4, 6, 7)),
                    (0.0625, (2, 4, 7, 1)),
                    (0.0625j, (3, 5, 0, 1)),
                    (0.0625, (3, 5, 6, 0)),
                    (0.0625j, (3, 5, 6, 7)),
                    (-0.0625, (3, 5, 7, 1)),
                    (0.0625, (4, 5, 0, 1)),
                    (-0.0625j, (4, 5, 6, 0)),
                    (0.0625, (4, 5, 6, 7)),
                    (0.0625j, (4, 5, 7, 1)),
                ),
                0.0,
            ),
            (2, True, Direction.Y): (
                (
                    (0.0625, (2, 3, 0, 1)),
                    (0.0625j, (2, 3, 6, 1)),
                    (0.0625, (2, 3, 6, 7)),
                    (0.0625j, (2, 3, 7, 0)),
                    (0.0625j, (2, 5, 0, 1)),
                    (-0.0625, (2, 5, 6, 1)),
                    (0.0625j, (2, 5, 6, 7)),
                    (-0.0625, (2, 5, 7, 0)),
                    (0.0625j, (3, 4, 0, 1)),
                    (-0.0625, (3, 4, 6, 1)),
                    (0.0625j, (3, 4, 6, 7)),
                    (-0.0625, (3, 4, 7, 0)),
                    (0.0625, (4, 5, 0, 1)),
                    (0.0625j, (4, 5, 6, 1)),
                    (0.0625, (4, 5, 6, 7)),
                    (0.0625j, (4, 5, 7, 0)),
                ),
                0.0,
            ),
            (2, False, Direction.X): (
                (
                    (0.0625, (0, 1, 4, 5)),
                    (-0.0625j, (0, 1, 6, 4)),
                    (0.0625, (0, 1, 6, 7)),
                    (0.0625j, (0, 1, 7, 5)),
                    (-0.0625j, (2, 0, 4, 5)),
                    (-0.0625, (2, 0, 6, 4)),
                    (-0.0625j, (2, 0, 6, 7)),
                    (0.0625, (2, 0, 7, 5)),
                    (0.0625, (2, 3, 4, 5)),
                    (-0.0625j, (2, 3, 6, 4)),
                    (0.0625, (2, 3, 6, 7)),
                    (0.0625j, (2, 3, 7, 5)),
                    (0.0625j, (3, 1, 4, 5)),
                    (0.0625, (3, 1, 6, 4)),
                    (0.0625j, (3, 1, 6, 7)),
                    (-0.0625, (3, 1, 7, 5)),
                ),
                0.0,
            ),
            (2, False, Direction.Y): (
                (
                    (0.0625, (0, 1, 4, 5)),
                    (0.0625j, (0, 1, 6, 5)),
                    (0.0625, (0, 1, 6, 7)),
                    (0.0625j, (0, 1, 7, 4)),
                    (0.0625j, (2, 1, 4, 5)),
                    (-0.0625, (2, 1, 6, 5)),
                    (0.0625j, (2, 1, 6, 7)),
                    (-0.0625, (2, 1, 7, 4)),
                    (0.0625, (2, 3, 4, 5)),
                    (0.0625j, (2, 3, 6, 5)),
                    (0.0625, (2, 3, 6, 7)),
                    (0.0625j, (2, 3, 7, 4)),
                    (0.0625j, (3, 0, 4, 5)),
                    (-0.0625, (3, 0, 6, 5)),
                    (0.0625j, (3, 0, 6, 7)),
                    (-0.0625, (3, 0, 7, 4)),
                ),
                0.0,
            ),
        }
        gaugemgr = gauge.ZNGauge(2)
        group_element = gaugemgr.group_elements_for_el_energy[0]  # For Z2 ther's only one element here.
        for ncopy in (1, 2):
            for mix_copies in (True, False):
                for orientation in (Direction.X, Direction.Y):
                    got_ind, got_const = config_base.generate_gauged_projector_terms(
                        ncopy=ncopy,
                        ncolor=1,
                        mix_copies=mix_copies,
                        orientation=orientation,
                        group_element=group_element,
                        site=0,
                        drop_real_zero=False,
                    )
                    exp_ind, exp_const = expected[(ncopy, mix_copies, orientation)]
                    self._assert_terms_equal(got_ind, got_const, exp_ind, exp_const)

    def test_generate_gauged_projector_terms_small_cases2(self):
        # same as above but without imaginary terms
        pass

    def test_generate_gauged_projector_terms_small_cases1_D6(self):
        # Expected outputs for ncopy in {1}, layer in {unmixed_copies},
        # orientation in {horizontal, vertical}, group_order = 2.
        # We compare ALL terms, including those with coefficients with zero real part.
        expected = {
            (1, False, Direction.X): (
                (
                    (0.25 / 4, (4, 5)),
                    (0.25 / 4, (6, 7)),
                    (1j * 0.25 / 4, (4, 6)),
                    (-1j * 0.25 / 4, (5, 7)),
                    (0.25 / 4, (0, 1, 2, 3, 4, 5)),
                    (0.25 / 4, (0, 1, 2, 3, 6, 7)),
                    (1j * 0.25 / 4, (0, 1, 2, 3, 4, 6)),
                    (-1j * 0.25 / 4, (0, 1, 2, 3, 5, 7)),
                    (0.25 / 4, (0, 3, 4, 5)),
                    (0.25 / 4, (0, 3, 6, 7)),
                    (1j * 0.25 / 4, (0, 3, 4, 6)),
                    (-1j * 0.25 / 4, (0, 3, 5, 7)),
                    (0.25 / 4, (1, 2, 4, 5)),
                    (0.25 / 4, (1, 2, 6, 7)),
                    (1j * 0.25 / 4, (1, 2, 4, 6)),
                    (-1j * 0.25 / 4, (1, 2, 5, 7)),
                ),
                0.0,
            ),
            (1, True, Direction.X): (
                (
                    (0.25 / 4, (4, 5)),
                    (0.25 / 4, (6, 7)),
                    (0.25 / 4, (0, 1, 2, 3, 4, 5)),
                    (0.25 / 4, (0, 1, 2, 3, 6, 7)),
                    (0.25 / 4, (0, 3, 4, 5)),
                    (0.25 / 4, (0, 3, 6, 7)),
                    (0.25 / 4, (1, 2, 4, 5)),
                    (0.25 / 4, (1, 2, 6, 7)),
                ),
                0.0,
            ),
        }
        gaugemgr = gauge.D2nGauge(3)
        group_element = gaugemgr.get_representation(0, 1)
        ncopy = 1
        mix_copies = False
        for drop_real_zero in [False, True]:
            orientation = Direction.X
            got_ind, got_const = config_base.generate_gauged_projector_terms(
                ncopy=ncopy,
                ncolor=2,
                mix_copies=mix_copies,
                orientation=orientation,
                group_element=group_element,
                site=0,
                drop_real_zero=drop_real_zero,
            )
            exp_ind, exp_const = expected[(ncopy, drop_real_zero, orientation)]
            self._assert_terms_equal(got_ind, got_const, exp_ind, exp_const)

    def assertPolyEqual(self, result, expected):
        """Helper to compare polynomial dictionaries."""
        # Convert dict {indices: factor} -> sorted list [(factor, indices)]
        # This ensures deterministic comparison regardless of hash order
        res_list = sorted([(v, k) for k, v in result.items()], key=lambda x: x[1])
        exp_list = sorted([(v, k) for k, v in expected.items()], key=lambda x: x[1])

        self.assertEqual(len(res_list), len(exp_list), f"Length mismatch: {res_list} vs {exp_list}")

        for (r_val, r_idx), (e_val, e_idx) in zip(res_list, exp_list):
            self.assertEqual(r_idx, e_idx, f"Indices mismatch: {r_idx} != {e_idx}")
            self.assertAlmostEqual(r_val, e_val, places=12, msg=f"Value mismatch for {r_idx}")

    def test_anti_commutation(self):
        """Test 1: Reordering checks sign flipping (c_2 c_1 -> -c_1 c_2)."""
        # Input is now a dictionary: {indices: factor}
        poly = {(2, 1): 1.0}
        expected = {(1, 2): -1.0}
        self.assertPolyEqual(config_base.simplify_majorana_acc(poly), expected)

    def test_nested_contraction(self):
        """Test 2: Recursive contraction (c_1 c_2 c_2 c_1 -> c_1 c_1 -> 1)."""
        poly = {(1, 2, 2, 1): 1.0}
        expected = {(): 1.0}  # Empty tuple is Identity
        self.assertPolyEqual(config_base.simplify_majorana_acc(poly), expected)

    def test_cancellation(self):
        """Test 3: Aggregation and cancellation of terms."""
        # c_1 c_2 + c_2 c_1  -->  c_1 c_2 - c_1 c_2  -->  0
        poly = {(1, 2): 1.0, (2, 1): 1.0}
        expected = {}  # Should be empty after cancellation
        self.assertPolyEqual(config_base.simplify_majorana_acc(poly), expected)

    def test_mixed_case(self):
        """Test 4: Sort then contract combined."""
        # c_1 c_3 c_1 --> swap c_3/c_1 (sign flip) --> -c_1 c_1 c_3 --> contract --> -c_3
        poly = {(1, 3, 1): 1.0}
        expected = {(3,): -1.0}
        self.assertPolyEqual(config_base.simplify_majorana_acc(poly), expected)


class TestElectricContstants(unittest.TestCase):
    """The electric energy is a fairly complicated operator.
    This class is for testing its implementation, as well as its helper functions."""

    def setUp(self):

        lat = lattice.Lattice2D(2, 2)
        num_pg_layer = 1
        num_fermionic_layer = 1
        nlayer = num_pg_layer + num_fermionic_layer
        unitcell_size = 1
        paramvec = np.random.rand(nlayer, unitcell_size, 20)
        cfg = system.Z2System2D_G2C_F2C_Config(lat, 1, 1, 1, 1, None, num_pg_layer=1, num_fermionic_layer=1)
        cfg.paramvec = paramvec
        self.system_z2 = system.Z2System2D(cfg)
        self.system_z2.cfg.enforce_parameter_conditions(self.system_z2.cfg.paramvec)

    def test_make_sigma(self):
        # Test for Z2, Zn, Dn
        pass

    def test_bracket_term(self):
        pass

    def test_pfaffian_wick_phase(self):
        pass

    def test_get_cov_matrix_idx(self):
        pass

    def test_simplify_majorana_acc(self):
        pass

    def test_generate_gauged_projector_terms(self):
        pass

    # this test should be moved to a more suitable location
    def test_el_energy_imaginary(self):
        """Test that the electric energy calculation returns the same result whether or not
        the imaginary parts are included in the calculation."""
        pass
