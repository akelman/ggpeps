import unittest
from collections import defaultdict

import numpy as np

import ggpeps.system.config_base as config_base
from ggpeps import gauge, system, lattice, utils
from ggpeps.system.backend import backend

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
                        drop_imag=False,
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
                drop_imag=drop_real_zero,
            )
            exp_ind, exp_const = expected[(ncopy, drop_real_zero, orientation)]
            self._assert_terms_equal(got_ind, got_const, exp_ind, exp_const)

    def assertPolyEqual(self, result, expected):
        """Helper to compare polynomial dictionaries."""
        # TODO: Is this working?

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
        self.assertPolyEqual(config_base.simplify_polynomial(poly), expected)

    def test_nested_contraction(self):
        """Test 2: Recursive contraction (c_1 c_2 c_2 c_1 -> c_1 c_1 -> 1)."""
        poly = {(1, 2, 2, 1): 1.0}
        expected = {(): 1.0}  # Empty tuple is Identity
        self.assertPolyEqual(config_base.simplify_polynomial(poly), expected)

    def test_cancellation(self):
        """Test 3: Aggregation and cancellation of terms."""
        # c_1 c_2 + c_2 c_1  -->  c_1 c_2 - c_1 c_2  -->  0
        poly = {(1, 2): 1.0, (2, 1): 1.0}
        expected = {}  # Should be empty after cancellation
        self.assertPolyEqual(config_base.simplify_polynomial(poly), expected)

    def test_mixed_case(self):
        """Test 4: Sort then contract combined."""
        # c_1 c_3 c_1 --> swap c_3/c_1 (sign flip) --> -c_1 c_1 c_3 --> contract --> -c_3
        poly = {(1, 3, 1): 1.0}
        expected = {(3,): -1.0}
        self.assertPolyEqual(config_base.simplify_polynomial(poly), expected)

    def test_longer_anti_commutation(self):
        """Test 5: Reordering checks sign flipping (c_2 c_1 -> -c_1 c_2)."""
        # Input is now a dictionary: {indices: factor}
        # TODO: this should fail, but it doesn't, because assertPolyEqual is broken
        poly = {(2, 1, 7, 12, 15, 9): 1.0}
        expected = {(1, 2, 7, 9, 12, 15): -1.0}
        self.assertPolyEqual(config_base.simplify_polynomial(poly), expected)


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
        cfg = utils.make_z2_2copy_config(lat, 1, 1, 1, 1, None, num_pg_layer=1, num_fermionic_layer=1)
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


# ============================================================================
# These helpers re-derive, FROM SCRATCH (creation/annihilation operators), the
# polynomials that `_vacuum_terms`, `_w_gauged_terms`, `_w_dag_terms` and the
# grouped assembly in `generate_gauged_projector_terms` are supposed to return.
#
#   c^dag = 1/2 (g1 + i g2),   c = 1/2 (g1 - i g2)
#   xi_k  = 1 (Direction.X) or i (Direction.Y)
#   V_a      = (l l^dag)(r r^dag)
#   W_a(h)   = 1 + xi  l^dag_(color,sc) sum_m M[m,color] r^dag_(m,copy)
#   w_a^dag  = 1 + conj(xi)  r_(color,copy) l_(color,sc)
#   O_h      = pref * (prod_a W_a)(prod_a V_a)(prod_a w_a^dag),  pref = 2^-(ncolor*ncopy)
# ============================================================================


def _ref_canon(poly):
    """Canonicalize a Majorana polynomial: sort indices (sign from transpositions),
    contract c_i^2 = 1, aggregate, drop ~0. Independent of the production code."""
    out = defaultdict(complex)
    for inds, coef in poly.items():
        lst = list(inds)
        sign = 1
        n = len(lst)
        for i in range(n):  # bubble sort, track parity of swaps
            for j in range(n - 1 - i):
                if lst[j] > lst[j + 1]:
                    lst[j], lst[j + 1] = lst[j + 1], lst[j]
                    sign = -sign
        stack = []  # contract adjacent equal pairs (c^2 = 1)
        for x in lst:
            if stack and stack[-1] == x:
                stack.pop()
            else:
                stack.append(x)
        out[tuple(stack)] += coef * sign
    return {k: v for k, v in out.items() if abs(v) > 1e-12}


def _ref_mul(p, q):
    """Multiply two polynomials; monomial = p_indices ++ q_indices (order matters)."""
    out = defaultdict(complex)
    for pi, pc in p.items():
        for qi, qc in q.items():
            out[pi + qi] += pc * qc
    return _ref_canon(out)


def _ref_scale(poly, s):
    return {k: v * s for k, v in poly.items()}


def _ref_add(*polys):
    acc = defaultdict(complex)
    for p in polys:
        for k, v in p.items():
            acc[k] += v
    return _ref_canon(acc)


_REF_ID = {(): 1.0 + 0j}


def _ref_idx(color, copy, direction, majorana, ncolors, ncopies, ndir=2):
    """Flat Majorana index built by ENUMERATING the documented mode ordering
    (grouped color -> copy -> direction -> majorana), rather than re-deriving the
    mixed-radix arithmetic the production code uses. This keeps this reference
    implementation an independent encoding of the *convention*: a wrong radix / off-by-one in
    get_cov_matrix_idx would diverge from this enumeration."""
    order = [
        (c, cp, d, m)
        for c in range(1, ncolors + 1)
        for cp in range(1, ncopies + 1)
        for d in range(1, ndir + 1)
        for m in range(1, 3)
    ]
    return order.index((color, copy, direction, majorana))


def _ref_cdag(g1, g2):
    return {(g1,): 0.5 + 0j, (g2,): 0.5j}


def _ref_cann(g1, g2):
    return {(g1,): 0.5 + 0j, (g2,): -0.5j}


def _ref_sigma(ncopy, mix_copies):
    """Independent reimplementation of make_sigma's pairing permutation (1-based)."""
    if not mix_copies or ncopy == 1:
        return tuple(range(1, ncopy + 1))
    s = [0] * ncopy
    for a in range(1, ncopy // 2 + 1):
        i, j = 2 * a - 1, 2 * a
        s[i - 1] = j
        s[j - 1] = i
    return tuple(s)


def _ref_vacuum(copy, sc, color, ncolors, ncopies):
    l1 = _ref_idx(color, sc, 1, 1, ncolors, ncopies)
    l2 = _ref_idx(color, sc, 1, 2, ncolors, ncopies)
    r1 = _ref_idx(color, copy, 2, 1, ncolors, ncopies)
    r2 = _ref_idx(color, copy, 2, 2, ncolors, ncopies)
    ll = _ref_mul(_ref_cann(l1, l2), _ref_cdag(l1, l2))
    rr = _ref_mul(_ref_cann(r1, r2), _ref_cdag(r1, r2))
    return _ref_mul(ll, rr)


def _ref_w_gauged(copy, sc, eta2, color, ncolors, ncopies, M):
    l1 = _ref_idx(color, sc, 1, 1, ncolors, ncopies)
    l2 = _ref_idx(color, sc, 1, 2, ncolors, ncopies)
    pieces = [_REF_ID]
    for m in range(1, ncolors + 1):
        r1 = _ref_idx(m, copy, 2, 1, ncolors, ncopies)
        r2 = _ref_idx(m, copy, 2, 2, ncolors, ncopies)
        ldag_rdag = _ref_mul(_ref_cdag(l1, l2), _ref_cdag(r1, r2))
        pieces.append(_ref_scale(ldag_rdag, eta2 * M[m - 1][color - 1]))
    return _ref_add(*pieces)


def _ref_w_dag(copy, sc, eta2, color, ncolors, ncopies):
    l1 = _ref_idx(color, sc, 1, 1, ncolors, ncopies)
    l2 = _ref_idx(color, sc, 1, 2, ncolors, ncopies)
    r1 = _ref_idx(color, copy, 2, 1, ncolors, ncopies)
    r2 = _ref_idx(color, copy, 2, 2, ncolors, ncopies)
    rl = _ref_mul(_ref_cann(r1, r2), _ref_cann(l1, l2))
    return _ref_add(_REF_ID, _ref_scale(rl, np.conj(eta2)))


def _ref_projector(ncopy, ncolor, mix_copies, orientation, group_element, site=0):
    """Full grouped assembly: (prod W)(prod V)(prod w^dag), pref, and Wick phase i^-(n/2).
    Returns (items_dict, constant) comparable to generate_gauged_projector_terms."""
    eta2 = 1.0 if orientation == Direction.X else 1j
    sigma = _ref_sigma(ncopy, mix_copies)
    M = group_element if site % 2 == 0 else np.conjugate(group_element)

    blockV = dict(_REF_ID)
    blockW = dict(_REF_ID)
    blockWd = dict(_REF_ID)
    for color in range(1, ncolor + 1):
        for copy in range(1, ncopy + 1):
            sc = sigma[copy - 1]
            blockV = _ref_mul(blockV, _ref_vacuum(copy, sc, color, ncolor, ncopy))
            blockW = _ref_mul(blockW, _ref_w_gauged(copy, sc, eta2, color, ncolor, ncopy, M))
            blockWd = _ref_mul(blockWd, _ref_w_dag(copy, sc, eta2, color, ncolor, ncopy))

    full = _ref_mul(_ref_mul(blockW, blockV), blockWd)
    full = _ref_scale(full, 2.0 ** (-ncopy * ncolor))

    constant = full.get((), 0.0)
    items = {}
    for mon, coef in full.items():
        if mon == ():
            continue
        phase = (1j) ** (-(len(mon) // 2))  # i^-(n/2), == pfaffian_wick_phase
        c = coef * phase
        if abs(c) > 1e-6:
            items[mon] = c
    return items, constant


# ============================================================================
# Tests
# ============================================================================


class _PolyAssertMixin:
    def assert_poly_close(self, got, exp, tol=1e-9, msg=""):
        """Compare two {monomial: coeff} dicts, ignoring near-zero terms."""
        got = {k: v for k, v in got.items() if abs(v) > tol}
        exp = {k: v for k, v in exp.items() if abs(v) > tol}
        self.assertEqual(
            set(got),
            set(exp),
            f"{msg}\n  monomials only in code: {sorted(set(got) - set(exp))}"
            f"\n  monomials only in reference: {sorted(set(exp) - set(got))}",
        )
        for k in exp:
            self.assertTrue(
                abs(got[k] - exp[k]) < 1e-7,
                f"{msg}\n  coeff mismatch at {k}: code={got[k]} reference={exp[k]}",
            )


class TestMajoranaAlgebraHelpers(_PolyAssertMixin, unittest.TestCase):
    """Low-level helpers used to assemble the gauged projector."""

    def test_poly_mul_concatenates_on_the_right(self):
        # c_0 * c_1 -> (0, 1)
        p = config_base.simplify_polynomial({(0,): 1.0})
        out = config_base.poly_mul(p, {(1,): 1.0})
        self.assert_poly_close(out, {(0, 1): 1.0}, msg="_poly_mul order")

    def test_poly_mul_contracts_squares(self):
        # (0, 1) * (1, 2) = (0, 1, 1, 2) -> (0, 2)
        p = config_base.simplify_polynomial({(0, 1): 1.0})
        out = config_base.poly_mul(p, {(1, 2): 1.0})
        self.assert_poly_close(out, {(0, 2): 1.0}, msg="_poly_mul contraction")

    def test_poly_mul_distributes_over_terms_and_coeffs(self):
        p = config_base.simplify_polynomial({(0,): 2.0})
        out = config_base.poly_mul(p, {(1,): 3.0, (2,): 1j})
        self.assert_poly_close(out, {(0, 1): 6.0, (0, 2): 2j}, msg="_poly_mul distribute")

    def test_simplify_majorana_acc_matches_independent_canon(self):
        # Random monomials checked against the independent reference canonicalizer.
        rng = np.random.default_rng(0)
        for _ in range(50):
            n = rng.integers(0, 7)
            inds = tuple(int(x) for x in rng.integers(0, 5, size=n))
            coef = complex(rng.standard_normal(), rng.standard_normal())
            got = config_base.simplify_polynomial({inds: coef})
            exp = _ref_canon({inds: coef})
            self.assert_poly_close(got, exp, msg=f"simplify vs canon for {inds}")

    def test_make_sigma(self):
        self.assertEqual(config_base.make_sigma(1, True), (1,))
        self.assertEqual(config_base.make_sigma(1, False), (1,))
        self.assertEqual(config_base.make_sigma(2, False), (1, 2))
        self.assertEqual(config_base.make_sigma(2, True), (2, 1))
        self.assertEqual(config_base.make_sigma(4, False), (1, 2, 3, 4))
        self.assertEqual(config_base.make_sigma(4, True), (2, 1, 4, 3))
        with self.assertRaises(ValueError):
            config_base.make_sigma(3, False)

    def test_get_cov_matrix_idx_matches_documented_ordering(self):
        for ncolors, ncopies in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            for color in range(1, ncolors + 1):
                for copy in range(1, ncopies + 1):
                    for direction in (1, 2):
                        for majorana in (1, 2):
                            got = config_base.get_cov_matrix_idx(color, copy, direction, majorana, ncolors, ncopies)
                            exp = _ref_idx(color, copy, direction, majorana, ncolors, ncopies)
                            self.assertEqual(got, exp, f"idx({color},{copy},{direction},{majorana})")


class TestGaugedProjectorPrimitives(_PolyAssertMixin, unittest.TestCase):
    """`_vacuum_terms`, `_w_gauged_terms`, `_w_dag_terms` vs the independent reference implementation."""

    def _group_elements(self):
        g = gauge.D2nGauge(3)
        return {
            "identity": g.get_representation(0, 0),
            "rotation": g.get_representation(1, 0),  # color-mixing (off-diagonal)
            "diag_reflection": g.get_representation(0, 1),  # diag(1, -1)
            "mixing_reflection": g.get_representation(1, 1),  # color-mixing reflection
        }

    def test_vacuum_terms(self):
        for ncolor, ncopy in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            for color in range(1, ncolor + 1):
                for copy in range(1, ncopy + 1):
                    for sc in range(1, ncopy + 1):
                        got = _ref_canon(config_base.vacuum_terms(copy, sc, color, ncolor, ncopy))
                        exp = _ref_vacuum(copy, sc, color, ncolor, ncopy)
                        self.assert_poly_close(got, exp, msg=f"vacuum_terms c{color} cp{copy} sc{sc}")

    def test_w_dag_terms(self):
        for eta2 in (1.0, 1j):
            for ncolor, ncopy in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                for color in range(1, ncolor + 1):
                    for copy in range(1, ncopy + 1):
                        for sc in range(1, ncopy + 1):
                            got = _ref_canon(config_base._w_dag_terms(copy, sc, eta2, color, ncolor, ncopy))
                            exp = _ref_w_dag(copy, sc, eta2, color, ncolor, ncopy)
                            self.assert_poly_close(got, exp, msg=f"_w_dag_terms eta2={eta2} c{color}")

    def test_w_gauged_terms(self):
        elems = self._group_elements()
        for name, M in elems.items():
            for eta2 in (1.0, 1j):
                ncolor, ncopy = 2, 2
                for color in range(1, ncolor + 1):
                    for copy in range(1, ncopy + 1):
                        for sc in range(1, ncopy + 1):
                            got = _ref_canon(config_base._w_gauged_terms(copy, sc, eta2, color, ncolor, ncopy, M))
                            exp = _ref_w_gauged(copy, sc, eta2, color, ncolor, ncopy, M)
                            self.assert_poly_close(
                                got, exp, msg=f"_w_gauged_terms {name} eta2={eta2} c{color} cp{copy}"
                            )

    def test_w_gauged_identity_decouples_colors(self):
        # For the identity rep M = I the gauged factor must reference only the same-color r mode
        # (no color mixing). This is the property that distinguishes Abelian from non-Abelian.
        M = gauge.D2nGauge(3).get_representation(0, 0)
        ncolor, ncopy = 2, 2
        terms = config_base._w_gauged_terms(
            copy=1, sigma_copy=1, eta2=1.0, color=1, ncolors=ncolor, ncopies=ncopy, gauging_matrix=M
        )
        other_color_modes = {
            config_base.get_cov_matrix_idx(2, 1, 2, 1, ncolor, ncopy),
            config_base.get_cov_matrix_idx(2, 1, 2, 2, ncolor, ncopy),
        }
        used = {idx for inds, _ in terms.items() for idx in inds}
        self.assertTrue(used.isdisjoint(other_color_modes), "identity rep must not mix colors")


class TestGaugedProjectorAssembly(_PolyAssertMixin, unittest.TestCase):
    """`generate_gauged_projector_terms` (the grouped (prod W)(prod V)(prod w_dag) assembly,
    pref, and Wick phase) vs the independent reference implementation. This is the regression net for the
    operator-ordering bug, which was invisible for diagonal reps but real for color-mixing ones."""

    def _cases(self):
        d6 = gauge.D2nGauge(3)
        cases = []
        # D6: ncolor = 2, mix_copies = False, identity / rotation / reflections.
        for ncopy in (1, 2):
            for orientation in (Direction.X, Direction.Y):
                for q in (0, 1):
                    for p in (0, 1):
                        cases.append(
                            dict(
                                ncopy=ncopy,
                                ncolor=2,
                                mix_copies=False,
                                orientation=orientation,
                                group_element=d6.get_representation(p, q),
                                tag=f"D6 ncopy={ncopy} {orientation} rep({p},{q})",
                            )
                        )
        # Z2-like: ncolor = 1, both mix_copies settings, both 1x1 reps.
        for ncopy in (1, 2):
            for mix_copies in (True, False):
                for orientation in (Direction.X, Direction.Y):
                    for sign in (1, -1):
                        cases.append(
                            dict(
                                ncopy=ncopy,
                                ncolor=1,
                                mix_copies=mix_copies,
                                orientation=orientation,
                                group_element=np.array([[float(sign)]]),
                                tag=f"Z2 ncopy={ncopy} mix={mix_copies} {orientation} M={sign}",
                            )
                        )
        return cases

    def test_assembly_matches_reference(self):
        for case in self._cases():
            tag = case.pop("tag")
            got_ind, got_const = config_base.generate_gauged_projector_terms(
                ncopy=case["ncopy"],
                ncolor=case["ncolor"],
                mix_copies=case["mix_copies"],
                orientation=case["orientation"],
                group_element=case["group_element"],
                site=0,
                drop_imag=False,
            )
            exp_items, exp_const = _ref_projector(
                case["ncopy"],
                case["ncolor"],
                case["mix_copies"],
                case["orientation"],
                case["group_element"],
                site=0,
            )
            self.assertTrue(
                abs(got_const - exp_const) < 1e-9,
                f"{tag}: constant {got_const} vs {exp_const}",
            )
            got_poly = _ref_canon({mon: coef for coef, mon in got_ind})
            self.assert_poly_close(got_poly, exp_items, msg=tag)


# ==================== Unique electric-energy terms (dedup across group elements) ====================


def _with_captured_raw_terms(cfg_cls):
    """Subclass of a config class that records the raw per-group-element structures on their way
    into set_el_energy_terms (the config itself keeps only the derived unique-basis form), so
    tests can use them as an independent ground truth."""

    class _Capturing(cfg_cls):  # type: ignore[valid-type, misc]
        def set_el_energy_terms(self, idx_vec, coeffs_vec, constants_vec):
            self.raw_idx_vec = tuple(idx_vec)
            self.raw_coeffs_vec = tuple(coeffs_vec)
            super().set_el_energy_terms(idx_vec, coeffs_vec, constants_vec)

    return _Capturing


class TestBuildUniqueElTerms(unittest.TestCase):
    """build_unique_el_terms re-expresses the raw idx/coeffs structures in a unique-index basis
    shared across group elements (the system then computes each Pfaffian once). The defining
    invariant: for every (group element, layer, link), the map {index tuple -> total coefficient}
    must be unchanged by the re-expression."""

    @classmethod
    def setUpClass(cls):
        lat = lattice.Lattice2D(2, 2)
        cls.cfg_d6 = _with_captured_raw_terms(system.D6System2D_Config)(
            lat, 1, 1, 0, 0, None, num_pg_layer=1, num_fermionic_layer=0
        )
        cls.cfg_z2 = _with_captured_raw_terms(system.Z2System2D_Config)(
            lattice.Lattice2D(2, 2), 1, 1, 1, 1, None, ncopy=2
        )

    @staticmethod
    def _term_dict(idx_link, coeffs_link, tol=1e-15):
        """Collapse one link's (buckets-of-tuples, buckets-of-coeffs) into {tuple: summed coeff}."""
        acc = defaultdict(complex)
        for bucket, coeff_bucket in zip(idx_link, coeffs_link):
            for tup, coeff in zip(bucket, coeff_bucket):
                acc[tup] += complex(coeff)
        return {k: v for k, v in acc.items() if abs(v) > tol}

    def _assert_dicts_close(self, got, exp, msg, tol=1e-12):
        self.assertEqual(set(got), set(exp), f"{msg}: index tuples differ")
        for tup in exp:
            self.assertTrue(abs(got[tup] - exp[tup]) < tol, f"{msg}: coeff at {tup}: {got[tup]} vs {exp[tup]}")

    def test_handcrafted_union_and_coefficients(self):
        """Exact expected output on a hand-built structure covering: union across group elements
        with first-seen ordering, zero-padding where an element lacks a tuple, summing of
        duplicate tuples within one element, size classes sorted ascending, and empty-bucket
        skipping. (0,1) vs (1,0) kept distinct pins the keying contract: real monomials are
        always canonically sorted by simplify_polynomial, so dedup must key on the exact tuple
        rather than the index set -- merging orderings would flip the Pfaffian sign if a
        non-canonical tuple ever appeared."""
        # 2 group elements, 1 layer, 2 links
        idx_vec = (
            (  # ge0
                (
                    (((0, 1), (2, 3), (0, 1)),),  # link 0: one size-2 bucket, (0,1) listed twice
                    ((), ((4, 5),)),  # link 1: empty bucket must be skipped
                ),
            ),
            (  # ge1
                (
                    ((((2, 3)), (1, 0)), ((0, 1, 2, 3),)),  # link 0: size-2 and size-4 buckets
                    (((4, 5),),),  # link 1
                ),
            ),
        )
        coeffs_vec = (
            ((((1.0, 2.0, 0.5),), ((), (1j,))),),  # ge0
            ((((3.0, 4j), (5.0,)), ((2.0,),)),),  # ge1
        )
        uniq_idx, uniq_coeffs = config_base.build_unique_el_terms(idx_vec, coeffs_vec)

        expected_idx = (  # [layer][link][size_class]
            (
                (((0, 1), (2, 3), (1, 0)), ((0, 1, 2, 3),)),
                (((4, 5),),),
            ),
        )
        expected_coeffs = (  # [ge][layer][link][size_class]
            ((((1.5, 2.0, 0.0), (0.0,)), ((1j,),)),),
            ((((0.0, 3.0, 4j), (5.0,)), ((2.0,),)),),
        )
        self.assertEqual(uniq_idx, expected_idx)
        self.assertEqual(uniq_coeffs, expected_coeffs)

    def test_z2_single_group_element_is_noop_reindexing(self):
        """For a single stored group element (Z2) the unique basis must be the original structure
        verbatim: same index tuples in the same order, same coefficients, nothing dropped or merged."""
        cfg = self.cfg_z2
        self.assertEqual(len(cfg.raw_idx_vec), 1)
        self.assertEqual(len(cfg.uniq_coeffs_vec), 1)
        self.assertEqual(cfg.uniq_idx_vec, cfg.raw_idx_vec[0])
        for lay in range(len(cfg.raw_idx_vec[0])):
            for link in range(len(cfg.raw_idx_vec[0][lay])):
                for size_ind, coeff_bucket in enumerate(cfg.raw_coeffs_vec[0][lay][link]):
                    got = cfg.uniq_coeffs_vec[0][lay][link][size_ind]
                    self.assertTrue(
                        np.allclose(np.asarray(got), np.asarray(coeff_bucket, dtype=complex)),
                        f"Z2 coeffs changed at layer {lay}, link {link}, size class {size_ind}",
                    )

    def test_d6_unique_basis_preserves_terms_per_group_element(self):
        """D6 (3 group elements with genuinely different sparsity): re-expressing each element's
        coefficients in the shared unique basis must preserve its {index tuple -> coefficient} map,
        and the unique basis itself must be sane (aligned shapes, no duplicate tuples, ascending
        size classes, no invented tuples)."""
        cfg = self.cfg_d6
        num_ge = len(cfg.raw_idx_vec)
        self.assertEqual(num_ge, 3)
        for lay in range(len(cfg.uniq_idx_vec)):
            for link in range(len(cfg.uniq_idx_vec[lay])):
                uniq_link = cfg.uniq_idx_vec[lay][link]
                sizes = [len(bucket[0]) for bucket in uniq_link]
                self.assertEqual(sizes, sorted(sizes), "size classes not ascending")
                all_original = set()
                for ge in range(num_ge):
                    for bucket in cfg.raw_idx_vec[ge][lay][link]:
                        all_original.update(bucket)
                for size_ind, bucket in enumerate(uniq_link):
                    self.assertEqual(len(bucket), len(set(bucket)), "duplicate tuples in unique basis")
                    self.assertTrue(set(bucket) <= all_original, "unique basis invented a tuple")
                    self.assertTrue(all(len(t) == sizes[size_ind] for t in bucket), "mixed sizes in a bucket")
                    for ge in range(num_ge):
                        self.assertEqual(
                            len(cfg.uniq_coeffs_vec[ge][lay][link][size_ind]),
                            len(bucket),
                            "coeffs not aligned with unique index basis",
                        )
                for ge in range(num_ge):
                    got = self._term_dict(
                        [uniq_link[i] for i in range(len(uniq_link))],
                        cfg.uniq_coeffs_vec[ge][lay][link],
                    )
                    exp = self._term_dict(cfg.raw_idx_vec[ge][lay][link], cfg.raw_coeffs_vec[ge][lay][link])
                    self._assert_dicts_close(got, exp, f"ge {ge}, layer {lay}, link {link}")

    def test_d6_dedup_actually_reduces_pfaffian_count(self):
        """The point of the unique basis: D6's group elements share index sets (the two
        color-mixing reflections have identical sets, the color-diagonal one is a subset), so the
        number of Pfaffians per link must drop by at least 2x vs computing per group element."""
        cfg = self.cfg_d6
        num_ge = len(cfg.raw_idx_vec)
        for lay in range(len(cfg.uniq_idx_vec)):
            for link in range(len(cfg.uniq_idx_vec[lay])):
                uniq_count = sum(len(bucket) for bucket in cfg.uniq_idx_vec[lay][link])
                per_ge_counts = [
                    sum(len(bucket) for bucket in cfg.raw_idx_vec[ge][lay][link]) for ge in range(num_ge)
                ]
                self.assertEqual(
                    uniq_count,
                    max(per_ge_counts),
                    "expected the largest group element's index sets to contain all others (D6 structure)",
                )
                self.assertLessEqual(2 * uniq_count, sum(per_ge_counts), "dedup saves less than 2x on D6")


class TestUniqueElTermsSystemConsumption(unittest.TestCase):
    """Integration: the system evaluates Pfaffians once in the unique basis (_compute_el_pfaffians
    with cfg.uniq_idx_vec) and applies per-group-element coefficient dots (cfg.uniq_coeffs_vec).
    This must reproduce the pre-dedup semantics: Pfaffians evaluated directly from each group
    element's own raw idx/coeffs structures (captured on their way into set_el_energy_terms)."""

    def test_d6_dedup_pf_tot_matches_per_group_element_evaluation(self):
        rng = np.random.RandomState(20260706)
        lat = lattice.Lattice2D(2, 2)
        cfg = _with_captured_raw_terms(system.D6System2D_Config)(
            lat, 1, 1, 0, 0, None, num_pg_layer=1, num_fermionic_layer=0
        )
        cfg.paramvec = rng.rand(1, 1, 20)
        cfg.enforce_parameter_conditions(cfg.paramvec)
        sys_d6 = system.D2nSystem2D(cfg)

        # Random (non-identity) gauge field so the covariance matrices are generic
        vals = cfg.gaugemgr.get_possible_gauge_values()
        gauge_config = [vals[rng.randint(len(vals))].copy() for _ in range(lat.nlinks)]
        sys_d6.update_gauge_full_system(gauge_config)

        el_pfaffians = np.asarray(sys_d6.el_pfaffians)  # unique basis, no group-element axis
        covmats = np.asarray(sys_d6.covmat_out_mod_vec)
        num_ge = len(cfg.raw_idx_vec)
        for ge in range(num_ge):
            for lay in range(cfg.nlayer):
                for link_pos in range(len(cfg.mod_link_inds)):
                    # Production assembly (as in _compute_el_energy_op_vec): constant + coefficient
                    # dots against the shared unique-basis Pfaffians
                    pf_dedup = complex(cfg.constants_vec[ge][lay][link_pos])
                    for size_ind, coeffs in enumerate(cfg.uniq_coeffs_vec[ge][lay][link_pos]):
                        pf_dedup += np.dot(np.asarray(coeffs), el_pfaffians[lay, link_pos, size_ind, : len(coeffs)])

                    # Reference: this group element's own terms, one Pfaffian per stored tuple
                    pf_ref = complex(cfg.constants_vec[ge][lay][link_pos])
                    for bucket, coeff_bucket in zip(
                        cfg.raw_idx_vec[ge][lay][link_pos], cfg.raw_coeffs_vec[ge][lay][link_pos]
                    ):
                        for tup, coeff in zip(bucket, coeff_bucket):
                            inds = np.asarray(tup)
                            sub = covmats[lay][link_pos][np.ix_(inds, inds)]
                            pf_ref += complex(coeff) * backend.pfaffian(sub)

                    np.testing.assert_allclose(
                        pf_dedup,
                        pf_ref,
                        rtol=1e-9,
                        atol=1e-12,
                        err_msg=f"pf_tot mismatch at ge {ge}, layer {lay}, link {link_pos}",
                    )


class TestPaddedElTerms(unittest.TestCase):
    """The padded evaluation layout (cfg.el_eval_mode == "padded"): all monomials padded to one
    matrix dimension K via auxiliary indices into a Pf = 1 aux block, so the electric pipeline is
    a single batched Pfaffian call. Padding must not change any value."""

    def test_padding_preserves_pfaffian_values(self):
        """Direct check of the J-block trick on random antisymmetric matrices: gathering a padded
        index tuple from the aux-extended matrix gives the same Pfaffian as the bare tuple."""
        rng = np.random.RandomState(3)
        nmodes, K = 10, 8
        m = rng.standard_normal((nmodes, nmodes))
        cov = m - m.T
        aux = np.kron(np.eye(K // 2), np.array([[0.0, 1.0], [-1.0, 0.0]]))
        padded = np.block(
            [[cov, np.zeros((nmodes, K))], [np.zeros((K, nmodes)), aux]]
        )
        for k in (2, 4, 6, 8):
            tup = list(rng.choice(nmodes, size=k, replace=False))
            padded_tup = np.array(tup + [nmodes + i for i in range(K - k)])
            bare = backend.pfaffian(cov[np.ix_(tup, tup)])
            via_pad = backend.pfaffian(padded[np.ix_(padded_tup, padded_tup)])
            self.assertAlmostEqual(bare, via_pad, places=10, msg=f"padding changed Pf for size {k}")

    def test_build_padded_el_terms_structure(self):
        """Exact expected padded arrays for a small hand-built unique basis: aux indices appended
        after nmodes, dummy rows all-aux with zero coefficient, coefficients aligned."""
        uniq_idx = (  # 1 layer, 2 links; link 0 has sizes 2 and 4, link 1 has one size-2 term
            (
                (((0, 1), (2, 3)), ((0, 1, 2, 3),)),
                (((4, 5),),),
            ),
        )
        uniq_coeffs = (  # 2 group elements
            ((((1.0, 2.0), (3.0,)), ((4.0,),)),),
            ((((0.0, 5j), (6.0,)), ((7.0,),)),),
        )
        nmodes = 6
        idx_arr, coeffs_arr, aux_block = config_base.build_padded_el_terms(uniq_idx, uniq_coeffs, nmodes)

        self.assertEqual(idx_arr.shape, (1, 2, 3, 4))  # T = 3 terms, K = 4
        self.assertEqual(coeffs_arr.shape, (2, 1, 2, 3))
        np.testing.assert_array_equal(
            idx_arr[0, 0], [[0, 1, 6, 7], [2, 3, 6, 7], [0, 1, 2, 3]]
        )
        np.testing.assert_array_equal(
            idx_arr[0, 1], [[4, 5, 6, 7], [6, 7, 8, 9], [6, 7, 8, 9]]  # 2 dummy rows
        )
        np.testing.assert_array_equal(coeffs_arr[0, 0, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(coeffs_arr[1, 0, 0], [0.0, 5j, 6.0])
        np.testing.assert_array_equal(coeffs_arr[0, 0, 1], [4.0, 0.0, 0.0])
        np.testing.assert_array_equal(coeffs_arr[1, 0, 1], [7.0, 0.0, 0.0])
        expected_aux = np.array(
            [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=float
        )
        np.testing.assert_array_equal(aux_block, expected_aux)

    def _assert_modes_agree(self, make_system):
        sizes = make_system("sizes")
        padded = make_system("padded")
        ev_s, ev_p = np.asarray(sizes.el_energy_op_vec), np.asarray(padded.el_energy_op_vec)
        np.testing.assert_allclose(ev_p, ev_s, rtol=1e-10, atol=1e-12, err_msg="el_energy_op_vec differs")
        g_s, g_p = np.asarray(sizes.el_energy_op_grad_vec), np.asarray(padded.el_energy_op_grad_vec)
        scale = max(np.max(np.abs(g_s)), 1.0)
        self.assertLess(np.max(np.abs(g_p - g_s)) / scale, 1e-10, "el gradients differ between modes")

    def test_d6_padded_matches_sizes(self):
        def make_system(mode):
            rng = np.random.RandomState(11)
            lat = lattice.Lattice2D(2, 2)
            cfg = system.D6System2D_Config(lat, 1, 1, 0, 0, None, num_pg_layer=1, num_fermionic_layer=0)
            cfg.paramvec = rng.rand(1, 1, 20)
            cfg.enforce_parameter_conditions(cfg.paramvec)
            cfg.el_eval_mode = mode
            s = system.D2nSystem2D(cfg)
            vals = cfg.gaugemgr.get_possible_gauge_values()
            s.update_gauge_full_system([vals[rng.randint(len(vals))].copy() for _ in range(lat.nlinks)])
            return s

        self._assert_modes_agree(make_system)

    def test_z2_padded_matches_sizes(self):
        def make_system(mode):
            rng = np.random.RandomState(12)
            lat = lattice.Lattice2D(2, 2)
            cfg = system.Z2System2D_Config(lat, 1, 1, 0, 0, None, ncopy=2)
            cfg.paramvec = rng.rand(cfg.nlayer, 1, 20)
            cfg.enforce_parameter_conditions(cfg.paramvec)
            cfg.el_eval_mode = mode
            s = system.Z2System2D(cfg)
            vals = cfg.gaugemgr.get_possible_gauge_values()
            s.update_gauge_full_system([vals[rng.randint(len(vals))].copy() for _ in range(lat.nlinks)])
            return s

        self._assert_modes_agree(make_system)
