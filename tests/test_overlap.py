"""Tests for the independent Gaussian-overlap electric-energy backend (``el_method="overlap"``).

The "overlap" path computes the per-configuration electric-energy observable F(G) directly from
the Bravyi & Gosset three-state Gaussian-overlap identity (Commun. Math. Phys. 356, 451 (2017),
eqs. 24-25), bypassing the bracket/Pfaffian open-link Schur machinery used by the default
"pfaffian" path. It is an independent oracle: any disagreement between the two paths on Z2 (the
gold-standard, ED-validated theory) is a bug in the test, not the framework.

Test layers:
  * TestOverlapMath          - the pure-math identity (vacuum reduction).
  * TestOverlapNormCrosscheck- |<phi_1|phi_2>|^2 matches the existing per-layer norm (Z2).
  * TestOverlapSystemMethod  - shape + identity-element ratio == 1.
  * TestOverlapPerConfigZ2   - F(G)==pfaffian for several explicit Z2 configs (per-config). PASSES.
  * TestOverlapReproducesZ2  - full exact-eval <el_energy> matches pfaffian (Z2, 1c/2c, gauge-fixed).
  * TestOverlapPerConfigD6   - F(G) overlap-vs-pfaffian for explicit D6 configs, 1 and 2 layers
                               (gauge-fixed), for both hand-picked sensitive params and random params.
  * TestOverlapD6FullEval    - full exact-eval <el_energy> for D6, 1 and 2 layers (skipped: too slow;
                               run explicitly):
"""

import unittest
from functools import partial

import numpy as np
import scipy.linalg as sla

from ggpeps.system import overlap as ov
from ggpeps.system.backend import backend
from ggpeps import lattice
from ggpeps import system
from ggpeps import exacteval
from ggpeps.exacteval import ExactEvaluatorConfig


def _random_even_pure_covmat(n_modes, rng, complex_state=False):
    """Return a 2n x 2n covariance M with M^2 = -I and Pf(M) = +1 (even parity),
    built as M = R M0 R^T with R a (complex) special-orthogonal rotation (det R = 1)."""
    dim = 2 * n_modes
    M0 = ov.vacuum_covmat(dim)
    A = rng.standard_normal((dim, dim))
    if complex_state:
        A = A + 1j * rng.standard_normal((dim, dim))
    A = A - A.T  # antisymmetric generator -> expm is (complex) orthogonal, det=1
    R = sla.expm(A)
    M = R @ np.asarray(M0) @ R.T
    M = 0.5 * (M - M.T)  # clean complex-expm round-off -> exact covariance (antisym)
    return M, np.asarray(M0)


class TestOverlapMath(unittest.TestCase):
    def test_chi_reduces_to_norm_when_states_equal(self):
        """For phi1 = phi2 = phi (even parity), the triple-overlap product reduces to
        |<Omega|phi>|^2, so 4^-n * chi(M, M, M0) == 2^-n * Pf(M0 + M)  (Bravyi eq 22)."""
        rng = np.random.default_rng(0)
        for complex_state in (False, True):
            for n_modes in (2, 3, 4):
                M, M0 = _random_even_pure_covmat(n_modes, rng, complex_state)
                n = M.shape[0] // 2
                chi = ov.gaussian_overlap_chi(M, M, M0)
                lhs = (4.0 ** (-n)) * chi
                rhs = (2.0 ** (-n)) * backend.pfaffian(M0 + M)
                self.assertTrue(
                    np.allclose(lhs, rhs, atol=1e-9),
                    msg=f"complex={complex_state} n={n_modes}: {lhs} != {rhs}",
                )


def _build_z2_2c_system(seed=1, gf=0, num_pg_layer=2):
    lat = lattice.Lattice2D(2, 2, gf)
    cfg = system.Z2System2D_Config(
        lat, 1.0, 1.0, 0.0, 0.0, None, ncopy=2, num_pg_layer=num_pg_layer, num_fermionic_layer=0, mod_link_inds=(0,)
    )
    rng = np.random.default_rng(seed)
    cfg.paramvec = rng.standard_normal(cfg.param_shape())
    cfg.enforce_parameter_conditions(cfg.paramvec)
    sysobj = system.Z2System2D(cfg)
    return sysobj


def _el_op_per_config(build_fn, configvec, method):
    """Per-config electric-energy observable F(G) = el_energy_op for the given backend.

    Builds a FRESH system fixed to `method` and measures one configuration. Building fresh (rather
    than switching el_method on a reused system) is deliberate: the pfaffian norm path is
    incremental (calculate_lognormvec_inc), so reusing one system across configs/methods can leak
    tracker state. A fresh, single-method system matches exactly how ExactEvaluator measures."""
    sysobj = build_fn()
    sysobj.cfg.el_method = method
    sysobj.update_gauge_full_system(configvec)
    return float(sysobj.el_energy_op)


class TestOverlapNormCrosscheck(unittest.TestCase):
    def test_overlap_magnitude_matches_existing_norm_per_layer(self):
        """overlap_magnitude_sq(-gamma_in, -mat_d) reproduces the existing per-layer squared norm
        exp(lognorm) for a non-trivial Z2 config — cross-checks the overlap math against the norm."""
        sysobj = _build_z2_2c_system()
        # pick a non-trivial (non-neutral) config on all links
        configvec = [sysobj.cfg.gaugemgr.get_possible_gauge_values()[0]] * sysobj.cfg.lattice.nlinks
        sysobj.update_gauge_full_system(configvec)

        gamma_in = np.asarray(sysobj.gamma_in_sys_vec)  # (nlayer, dim, dim)
        mat_d = np.asarray(sysobj.mat_d_vec)  # (nlayer, dim, dim)
        lognormvec = np.asarray(sysobj.calculate_lognormvec(all_factors=True))  # per layer

        for lay in range(sysobj.cfg.nlayer):
            m1 = -gamma_in[lay]
            m2 = -mat_d[lay]
            mag = ov.overlap_magnitude_sq(m1, m2)  # == |psi_lay(G)|^2
            expected = np.exp(lognormvec[lay])  # exp(lognorm), NOT 2*lognorm
            self.assertTrue(
                np.allclose(np.abs(mag), expected, rtol=1e-7, atol=1e-12),
                msg=f"layer {lay}: |{mag}| != {expected}",
            )


class TestOverlapSystemMethod(unittest.TestCase):
    def test_overlap_el_op_vec_shape_and_identity_ratio(self):
        """The overlap el-op vector has shape (n_h, nlayer, n_el_links), and feeding the identity
        element (G' == G) gives a per-layer ratio of exactly 1, independent of params/config."""
        sysobj = _build_z2_2c_system()
        configvec = [sysobj.cfg.gaugemgr.get_possible_gauge_values()[0]] * sysobj.cfg.lattice.nlinks
        sysobj.update_gauge_full_system(configvec)

        n_h = len(sysobj.cfg.gaugemgr.group_elements_for_el_energy)
        nlayer = sysobj.cfg.nlayer
        n_el = len(sysobj.cfg.mod_link_inds)

        vec = np.asarray(sysobj._compute_el_energy_op_vec_overlap())
        self.assertEqual(vec.shape, (n_h, nlayer, n_el))

        # Feeding the identity element as the only "reflection" must give ratio == 1 exactly
        # (G' == G), independent of params/config.
        ident = sysobj.cfg.gaugemgr.get_neutral_gauge_value()
        vec_id = np.asarray(sysobj._overlap_el_op_vec_for_elements((ident,)))
        self.assertTrue(np.allclose(vec_id, 1.0, atol=1e-10))


class TestOverlapPerConfigZ2(unittest.TestCase):
    """Compare the per-configuration observable F(G) = el_energy_op between the overlap and
    pfaffian backends for several explicit Z2 gauge configurations. This is the per-config
    analogue of the full-eval test below, and isolates F(G) before the gauge-weighted sum."""

    def _configs(self, sysobj):
        gv = sysobj.cfg.gaugemgr.get_possible_gauge_values()  # [neutral, flipped] for Z2
        nlk = sysobj.cfg.lattice.nlinks
        neutral, flipped = gv[0], gv[1]
        return [
            [neutral] * nlk,
            [flipped] * nlk,
            [flipped if i % 2 == 0 else neutral for i in range(nlk)],
            [flipped if i in (0, 3, 5) else neutral for i in range(nlk)],
        ]

    def test_z2_2c_per_config_matches_pfaffian(self):
        """Per-config F(G) from the overlap backend equals the pfaffian backend for several explicit
        2-copy Z2 configs (no gauge fixing)."""
        build = partial(_build_z2_2c_system, seed=7, gf=0)
        for ci, cv in enumerate(self._configs(build())):
            vp = _el_op_per_config(build, cv, "pfaffian")
            vo = _el_op_per_config(build, cv, "overlap")
            self.assertTrue(
                np.allclose(vp, vo, rtol=1e-7, atol=1e-9),
                msg=f"config {ci}: overlap F(G)={vo} != pfaffian F(G)={vp}",
            )

    def test_z2_1c_per_config_matches_pfaffian(self):
        """Per-config F(G) overlap == pfaffian for several explicit 1-copy (pure-gauge) Z2 configs —
        exercises the (-1)^nlayer Wick-phase handling of the odd ncolor*ncopy case."""

        def build():
            lat = lattice.Lattice2D(2, 2, 0)
            cfg = system.Z2System2D_Config(
                lat, 1.0, 1.0, 0.0, 0.0, None, ncopy=1, num_pg_layer=2, num_fermionic_layer=0, mod_link_inds=(0,)
            )
            rng = np.random.default_rng(13)
            cfg.paramvec = rng.standard_normal(cfg.param_shape())
            cfg.make_pure_gauge()  # 1-copy Z2 must be forced pure gauge (manager.py)
            cfg.enforce_parameter_conditions(cfg.paramvec)
            return system.Z2System2D(cfg)

        for ci, cv in enumerate(self._configs(build())):
            vp = _el_op_per_config(build, cv, "pfaffian")
            vo = _el_op_per_config(build, cv, "overlap")
            self.assertTrue(
                np.allclose(vp, vo, rtol=1e-6, atol=1e-8),
                msg=f"1c config {ci}: overlap F(G)={vo} != pfaffian F(G)={vp}",
            )


_Z2_GENERIC_1C = partial(system.Z2System2D_Config, ncopy=1)
_Z2_GENERIC_2C = partial(system.Z2System2D_Config, ncopy=2)


def _exact_el_energy(
    cfg_class, lat, paramvec, el_method, system_type, num_pg_layer=2, g_el=1.0, g_mag=1.0, mod_links=(0,)
):
    """Run exact eval and return aggregated <el_energy>. (Aggregated el_energy_op is NOT in the
    evaluator obsdict, so we compare el_energy — the physical observable.)"""
    cfg = cfg_class(
        lat, g_el, g_mag, 0.0, 0.0, None, num_pg_layer=num_pg_layer, num_fermionic_layer=0, mod_link_inds=mod_links
    )
    cfg.paramvec = np.reshape(paramvec, cfg.param_shape())
    if cfg.ncopy == 1:
        cfg.make_pure_gauge()  # 1-copy Z2 must be forced pure gauge (manager.py:383)
    cfg.enforce_parameter_conditions(cfg.paramvec)
    cfg.el_method = el_method
    sysobj = system_type(cfg)
    ec_cfg = ExactEvaluatorConfig()
    ec_cfg.compute_grads = False
    ev = exacteval.ExactEvaluator(ec_cfg, sysobj)
    ev.evaluate()
    return ev.get_obs_mean("el_energy")


class TestOverlapReproducesZ2(unittest.TestCase):
    def test_z2_2c_exact_matches_pfaffian(self):
        """Full exact-eval <el_energy> from the overlap backend equals the pfaffian backend for
        2-copy Z2 (no gauge fixing) — the gauge-weighted physical observable."""
        lat = lattice.Lattice2D(2, 2, 0)
        rng = np.random.default_rng(7)
        paramvec = rng.standard_normal((2, 1, 20))
        el_p = _exact_el_energy(_Z2_GENERIC_2C, lat, paramvec, "pfaffian", system.Z2System2D)
        el_b = _exact_el_energy(_Z2_GENERIC_2C, lat, paramvec, "overlap", system.Z2System2D)
        self.assertTrue(
            np.allclose(el_b, el_p, rtol=1e-7, atol=1e-8), msg=f"el_energy: overlap {el_b} != pfaffian {el_p}"
        )

    def test_z2_1c_exact_matches_pfaffian_all_layers(self):
        """Full exact-eval <el_energy> overlap == pfaffian for 1-copy Z2 at 1, 2 and 3 PG layers —
        odd and even layer counts lock in the (-1)^nlayer Wick-phase fix."""
        lat = lattice.Lattice2D(2, 2, 0)
        for num_pg_layer in (1, 2, 3):  # odd AND even: locks the (-1)^nlayer Wick-phase fix
            rng = np.random.default_rng(100 + num_pg_layer)
            sh = system.Z2System2D_Config(
                lat, 1.0, 1.0, 0.0, 0.0, None, ncopy=1, num_pg_layer=num_pg_layer, num_fermionic_layer=0
            ).param_shape()
            paramvec = rng.standard_normal(sh)
            el_p = _exact_el_energy(
                _Z2_GENERIC_1C, lat, paramvec, "pfaffian", system.Z2System2D, num_pg_layer=num_pg_layer
            )
            el_b = _exact_el_energy(
                _Z2_GENERIC_1C, lat, paramvec, "overlap", system.Z2System2D, num_pg_layer=num_pg_layer
            )
            self.assertTrue(
                np.allclose(el_b, el_p, rtol=1e-6, atol=1e-7),
                msg=f"1c num_pg_layer={num_pg_layer}: overlap {el_b} != pfaffian {el_p}",
            )

    def test_z2_gauge_fixed_matches_pfaffian(self):
        """Full exact-eval <el_energy> overlap == pfaffian for 2-copy Z2 WITH gauge fixing — checks
        the overlap backend is unaffected by the gauge-fixed lattice."""
        lat = lattice.Lattice2D(2, 2, -1)
        rng = np.random.default_rng(11)
        paramvec = rng.standard_normal((2, 1, 20))
        el_p = _exact_el_energy(_Z2_GENERIC_2C, lat, paramvec, "pfaffian", system.Z2System2D)
        el_b = _exact_el_energy(_Z2_GENERIC_2C, lat, paramvec, "overlap", system.Z2System2D)
        self.assertTrue(np.allclose(el_b, el_p, rtol=1e-6, atol=1e-7))


# Specific, hand-picked D6 parameter vectors (NOT random) that exhibit the open electric-energy
# discrepancy. These are SENSITIVE: the 2-layer set has large entries (|.| up to ~68) and is the
# configuration where the gauge-weighted electric energy goes unphysical for the pfaffian backend
# (full-eval pfaffian -26.74 vs overlap +22.66). They are written out verbatim so the regression is
# pinned to exactly these values. Shape is (num_pg_layer, unitcell_size=1, nparams_per_site=20).
D6_1LAYER_PARAMVEC = np.array(
    [
        0.0,
        -0.9939527757343662,
        1.4260579366967028,
        0.0,
        -0.28000887715354855,
        0.8089407935296115,
        -1.3065504646046338,
        1.865642743080219,
        -1.6649918678103066,
        2.1196735319722277,
        0.0,
        1.8397448679904251,
        -0.4227470147907068,
        0.0,
        1.3538850781838103,
        -0.5407833330276589,
        -0.7780308822579828,
        0.9742888873324884,
        -0.5166044214592881,
        0.32950471997126796,
    ]
).reshape(1, 1, 20)

D6_2LAYER_PARAMVEC = np.array(
    [
        0.0,
        -52.8917263263219,
        19.78345618719102,
        0.0,
        -42.27171900640149,
        17.746641326666904,
        43.61807316876326,
        -16.946194917588613,
        -27.8886671204573,
        25.88035087784146,
        0.0,
        -18.003447995609516,
        -3.148152878314228,
        0.0,
        -52.077659929094075,
        29.386499281816484,
        -58.689942035772205,
        16.797500524676142,
        32.23274988398398,
        8.263898579638377,
        0.0,
        36.326909166788774,
        -3.4463705034947996,
        0.0,
        34.292370450080206,
        67.64738719605107,
        -19.4054466188742,
        48.84705248075478,
        -39.814331170415855,
        -15.637498538375208,
        0.0,
        -12.417914185140402,
        56.57154429687569,
        0.0,
        11.449625035471854,
        1.8054305840486773,
        -21.545921775346713,
        -58.06042095755204,
        -1.941659389661235,
        -54.451686759216436,
    ]
).reshape(2, 1, 20)


def _build_d6_gauge_fixed_system(paramvec, num_pg_layer, el_method):
    """Fresh gauge-fixed (L=2) D6 system from an explicit paramvec, fixed to one backend.
    Building one system per backend (not switching el_method on a reused system) is required: see
    the module docstring. comp_tree links are free under gauge fixing; the rest are held neutral."""
    lat = lattice.Lattice2D(2, 2, -1)
    cfg = system.D6System2D_Config(
        lat, 1.0, 1.0, 0.0, 0.0, None, num_pg_layer=num_pg_layer, num_fermionic_layer=0, mod_link_inds=(0,)
    )
    cfg.paramvec = np.reshape(paramvec, cfg.param_shape())
    cfg.enforce_parameter_conditions(cfg.paramvec)
    cfg.el_method = el_method
    return system.D2nSystem2D(cfg)


def _d6_configvec(sysobj, free_gauge_indices):
    """Valid gauge-fixed configvec: comp_tree (free) links take the given gauge-value indices into
    get_possible_gauge_values() (0 == neutral); all other (tree-fixed) links stay neutral."""
    gv = sysobj.cfg.gaugemgr.get_possible_gauge_values()
    neutral = sysobj.cfg.gaugemgr.get_neutral_gauge_value()
    tree = sysobj.cfg.lattice.comp_tree
    assert len(free_gauge_indices) == len(tree), "one gauge index per free (comp_tree) link"
    cv = [neutral] * sysobj.cfg.lattice.nlinks
    for pos, gi in zip(tree, free_gauge_indices):
        cv[pos] = gv[gi]
    return cv


class TestOverlapPerConfigD6(unittest.TestCase):
    """Per-config F(G) = el_energy_op for gauge-fixed D6, overlap vs pfaffian, for 1 and 2 layers.

    The two backends compute the same per-config observable F(G) for the SAME gauge configuration,
    so they must be equal — exactly as TestOverlapPerConfigZ2 verifies for Z2. The combos
    include the historically flagged (3, 2, 1, 2, 2) [full-eval index 4370 on these params], where
    the buggy pfaffian gave +14700 vs overlap -18.95."""

    # free-link (comp_tree) gauge-value indices; comp_tree has 5 free links for L=2.
    _COMBOS = [
        (0, 0, 0, 0, 0),  # all-neutral
        (3, 0, 4, 0, 3),  # reflection-heavy
        (3, 2, 1, 2, 2),  # historic divergent config (full-eval index 4370)
    ]

    def _assert_match(self, paramvec, num_pg_layer):
        # one fresh system per backend; iterate configs on each (matches ExactEvaluator)
        sp = _build_d6_gauge_fixed_system(paramvec, num_pg_layer, "pfaffian")
        so = _build_d6_gauge_fixed_system(paramvec, num_pg_layer, "overlap")
        for combo in self._COMBOS:
            cv = _d6_configvec(sp, combo)
            sp.update_gauge_full_system(cv)
            so.update_gauge_full_system(cv)
            vp, vo = float(sp.el_energy_op), float(so.el_energy_op)
            self.assertTrue(
                np.allclose(vp, vo, rtol=1e-5, atol=1e-6),
                msg=f"npg={num_pg_layer} combo={combo}: overlap F(G)={vo} != pfaffian F(G)={vp}",
            )

    def test_d6_1layer_per_config_matches_pfaffian(self):
        """Per-config F(G) overlap == pfaffian for 1-layer gauge-fixed D6 on the sensitive
        hand-picked params."""
        self._assert_match(D6_1LAYER_PARAMVEC, 1)

    def test_d6_2layer_per_config_matches_pfaffian(self):
        """Per-config F(G) overlap == pfaffian for 2-layer gauge-fixed D6 on the sensitive
        hand-picked params.."""
        self._assert_match(D6_2LAYER_PARAMVEC, 2)

    def test_d6_1layer_per_config_random(self):
        """Per-config F(G) overlap == pfaffian for 1-layer gauge-fixed D6 on random params
        (fixed seed for reproducibility)."""
        paramvec = np.random.default_rng(2026).standard_normal((1, 1, 20))
        self._assert_match(paramvec, 1)

    def test_d6_2layer_per_config_random(self):
        """Per-config F(G) overlap == pfaffian for 2-layer gauge-fixed D6 on random params
        (fixed seed for reproducibility)."""
        paramvec = np.random.default_rng(4052).standard_normal((2, 1, 20))
        self._assert_match(paramvec, 2)


@unittest.skip("Takes too long")
class TestOverlapD6FullEval(unittest.TestCase):
    """Full exact-eval <el_energy> for gauge-fixed D6 (the physical, gauge-weighted observable),
    overlap vs pfaffian, for 1 and 2 layers. Skipped by default because it is SLOW (~5-10 min/eval
    on JAX, much longer on numpy) — remove the @skip to run it explicitly.
    """

    def _eval(self, paramvec, num_pg_layer):
        lat = lattice.Lattice2D(2, 2, -1)
        el_p = _exact_el_energy(
            system.D6System2D_Config, lat, paramvec, "pfaffian", system.D2nSystem2D, num_pg_layer=num_pg_layer
        )
        el_o = _exact_el_energy(
            system.D6System2D_Config, lat, paramvec, "overlap", system.D2nSystem2D, num_pg_layer=num_pg_layer
        )
        return float(el_p), float(el_o)

    def _check(self, paramvec, num_pg_layer):
        el_p, el_o = self._eval(paramvec, num_pg_layer)
        print(f"\nD6 {num_pg_layer}-layer: pfaffian={el_p:.6f}  overlap={el_o:.6f}")
        self.assertTrue(
            np.allclose(el_p, el_o, rtol=1e-5, atol=1e-6),
            msg=f"{num_pg_layer}-layer: overlap {el_o} != pfaffian {el_p}",
        )

    def test_d6_1layer_full_eval(self):
        """Full exact-eval <el_energy> overlap == pfaffian for 1-layer gauge-fixed D6"""
        self._check(D6_1LAYER_PARAMVEC, 1)

    def test_d6_2layer_full_eval(self):
        """Full exact-eval <el_energy> overlap == pfaffian for 2-layer gauge-fixed D6"""
        self._check(D6_2LAYER_PARAMVEC, 2)


if __name__ == "__main__":
    unittest.main()
