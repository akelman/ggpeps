import unittest
import numpy as np
import scipy.linalg as sla

from ggpeps.system import bravyi_overlap as bo
from ggpeps.system.backend import backend
from ggpeps import lattice
from ggpeps import system
from ggpeps import exacteval
from ggpeps.exacteval import ExactEvaluatorConfig


def _random_even_pure_covmat(n_modes, rng, complex_state=False):
    """Return a 2n x 2n covariance M with M^2 = -I and Pf(M) = +1 (even parity),
    built as M = R M0 R^T with R a (complex) special-orthogonal rotation (det R = 1)."""
    dim = 2 * n_modes
    M0 = bo.vacuum_covmat(dim)
    A = rng.standard_normal((dim, dim))
    if complex_state:
        A = A + 1j * rng.standard_normal((dim, dim))
    A = A - A.T                      # antisymmetric generator -> expm is (complex) orthogonal, det=1
    R = sla.expm(A)
    M = R @ np.asarray(M0) @ R.T
    M = 0.5 * (M - M.T)              # clean complex-expm round-off -> exact covariance (antisym)
    return M, np.asarray(M0)


class TestBravyiOverlapMath(unittest.TestCase):
    def test_chi_reduces_to_norm_when_states_equal(self):
        """For phi1 = phi2 = phi (even parity), the triple-overlap product reduces to
        |<Omega|phi>|^2, so 4^-n * chi(M, M, M0) == 2^-n * Pf(M0 + M)  (Bravyi eq 22)."""
        rng = np.random.default_rng(0)
        for complex_state in (False, True):
            for n_modes in (2, 3, 4):
                M, M0 = _random_even_pure_covmat(n_modes, rng, complex_state)
                n = M.shape[0] // 2
                chi = bo.gaussian_overlap_chi(M, M, M0)
                lhs = (4.0 ** (-n)) * chi
                rhs = (2.0 ** (-n)) * backend.pfaffian(M0 + M)
                self.assertTrue(
                    np.allclose(lhs, rhs, atol=1e-9),
                    msg=f"complex={complex_state} n={n_modes}: {lhs} != {rhs}",
                )


def _build_z2_2c_system(seed=1):
    lat = lattice.Lattice2D(2, 2, 0)  # no gauge fixing
    cfg = system.Z2System2D_G2C_F2C_Config(
        lat, 1.0, 1.0, 0.0, 0.0, None, num_pg_layer=2, num_fermionic_layer=0
    )
    rng = np.random.default_rng(seed)
    cfg.paramvec = rng.standard_normal(cfg.param_shape())
    cfg.enforce_parameter_conditions(cfg.paramvec)
    sysobj = system.Z2System2D(cfg)
    return sysobj


class TestBravyiNormCrosscheck(unittest.TestCase):
    def test_overlap_magnitude_matches_existing_norm_per_layer(self):
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
            mag = bo.overlap_magnitude_sq(m1, m2)  # == |psi_lay(G)|^2
            expected = np.exp(lognormvec[lay])  # exp(lognorm), NOT 2*lognorm
            self.assertTrue(
                np.allclose(np.abs(mag), expected, rtol=1e-7, atol=1e-12),
                msg=f"layer {lay}: |{mag}| != {expected}",
            )


class TestBravyiSystemMethod(unittest.TestCase):
    def test_bravyi_el_op_vec_shape_and_identity_ratio(self):
        sysobj = _build_z2_2c_system()
        configvec = [sysobj.cfg.gaugemgr.get_possible_gauge_values()[0]] * sysobj.cfg.lattice.nlinks
        sysobj.update_gauge_full_system(configvec)

        n_h = len(sysobj.cfg.gaugemgr.group_elements_for_el_energy)
        nlayer = sysobj.cfg.nlayer
        n_el = len(sysobj.cfg.mod_link_inds)

        vec = np.asarray(sysobj._compute_el_energy_op_vec_bravyi())
        self.assertEqual(vec.shape, (n_h, nlayer, n_el))

        # Feeding the identity element as the only "reflection" must give ratio == 1 exactly
        # (G' == G), independent of params/config.
        ident = sysobj.cfg.gaugemgr.get_neutral_gauge_value()
        vec_id = np.asarray(sysobj._bravyi_el_op_vec_for_elements((ident,)))
        self.assertTrue(np.allclose(vec_id, 1.0, atol=1e-10))


def _exact_el_energy(cfg_class, lat, paramvec, el_method, system_type, num_pg_layer=2,
                     g_el=1.0, g_mag=1.0, mod_links=(0,)):
    """Run exact eval and return aggregated <el_energy>. (Aggregated el_energy_op is NOT in the
    evaluator obsdict, so we compare el_energy — the physical observable.)"""
    cfg = cfg_class(lat, g_el, g_mag, 0.0, 0.0, None, num_pg_layer=num_pg_layer,
                    num_fermionic_layer=0, mod_link_inds=mod_links)
    cfg.paramvec = np.reshape(paramvec, cfg.param_shape())
    if isinstance(cfg, system.Z2System2DConfig):
        cfg.make_pure_gauge()                 # 1-copy Z2 must be forced pure gauge (manager.py:383)
    cfg.enforce_parameter_conditions(cfg.paramvec)
    cfg.el_method = el_method
    sysobj = system_type(cfg)
    ec_cfg = ExactEvaluatorConfig()
    ec_cfg.compute_grads = False
    ev = exacteval.ExactEvaluator(ec_cfg, sysobj)
    ev.evaluate()
    return ev.get_obs_mean("el_energy")


class TestBravyiReproducesZ2(unittest.TestCase):
    def test_z2_2c_exact_matches_pfaffian(self):
        lat = lattice.Lattice2D(2, 2, 0)
        rng = np.random.default_rng(7)
        paramvec = rng.standard_normal((2, 1, 20))
        el_p = _exact_el_energy(system.Z2System2D_G2C_F2C_Config, lat, paramvec,
                                "pfaffian", system.Z2System2D)
        el_b = _exact_el_energy(system.Z2System2D_G2C_F2C_Config, lat, paramvec,
                                "bravyi", system.Z2System2D)
        self.assertTrue(np.allclose(el_b, el_p, rtol=1e-7, atol=1e-8),
                        msg=f"el_energy: bravyi {el_b} != pfaffian {el_p}")

    def test_z2_1c_exact_matches_pfaffian_all_layers(self):
        lat = lattice.Lattice2D(2, 2, 0)
        for num_pg_layer in (1, 2, 3):     # odd AND even: locks the (-1)^nlayer Wick-phase fix
            rng = np.random.default_rng(100 + num_pg_layer)
            sh = system.Z2System2DConfig(lat, 1.0, 1.0, 0.0, 0.0, None, num_pg_layer=num_pg_layer,
                                         num_fermionic_layer=0).param_shape()
            paramvec = rng.standard_normal(sh)
            el_p = _exact_el_energy(system.Z2System2DConfig, lat, paramvec, "pfaffian",
                                    system.Z2System2D, num_pg_layer=num_pg_layer)
            el_b = _exact_el_energy(system.Z2System2DConfig, lat, paramvec, "bravyi",
                                    system.Z2System2D, num_pg_layer=num_pg_layer)
            self.assertTrue(np.allclose(el_b, el_p, rtol=1e-6, atol=1e-7),
                            msg=f"1c num_pg_layer={num_pg_layer}: bravyi {el_b} != pfaffian {el_p}")

    def test_z2_gauge_fixed_matches_pfaffian(self):
        lat = lattice.Lattice2D(2, 2, -1)
        rng = np.random.default_rng(11)
        paramvec = rng.standard_normal((2, 1, 20))
        el_p = _exact_el_energy(system.Z2System2D_G2C_F2C_Config, lat, paramvec,
                                "pfaffian", system.Z2System2D)
        el_b = _exact_el_energy(system.Z2System2D_G2C_F2C_Config, lat, paramvec,
                                "bravyi", system.Z2System2D)
        self.assertTrue(np.allclose(el_b, el_p, rtol=1e-6, atol=1e-7))


import os

PARAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
RUN_SLOW_D6 = os.environ.get("GGPEPS_RUN_SLOW_D6") == "1"


@unittest.skipUnless(RUN_SLOW_D6, "set GGPEPS_RUN_SLOW_D6=1 (and GGPEPS_BACKEND=jax) to run the slow D6 evals")
class TestBravyiD6(unittest.TestCase):
    """D6 application of the Bravyi oracle (gauge-fixed, L=2). SLOW: ~3-9 min/eval on JAX.
    Verified results (JAX, 2026-06-15):
      large_paramvec2.npy   (2 layer): pfaffian el=-26.744 (UNPHYSICAL), bravyi el=+22.663 (physical)
      1layer_with_largeparamvec.npy (1 layer): pfaffian=12.332168, bravyi=12.332166 (agree ~1e-6)
    The 2-layer case localizes the residual D6 bug to the pfaffian electric-energy path; the 1-layer
    case agreement validates the Bravyi oracle on D6."""

    def _run(self, fname, num_pg_layer):
        path = os.path.join(PARAM_DIR, fname)
        if not os.path.isfile(path):
            self.skipTest(f"{fname} not present")
        lat = lattice.Lattice2D(2, 2, -1)        # gauge fixing (7776 configs); JAX recommended
        paramvec = np.load(path)
        el_p = _exact_el_energy(system.D6System2D_Config, lat, paramvec, "pfaffian",
                                system.D2nSystem2D, num_pg_layer=num_pg_layer)
        el_b = _exact_el_energy(system.D6System2D_Config, lat, paramvec, "bravyi",
                                system.D2nSystem2D, num_pg_layer=num_pg_layer)
        return el_p, el_b

    def test_d6_1layer_bravyi_nonnegative(self):
        # pfaffian is non-divergent here (Woodbury-drift fix) but not independently known correct.
        # Assert only that Bravyi is physical; report the comparison.
        el_p, el_b = self._run("1layer_with_largeparamvec.npy", 1)
        print(f"\nD6 1-layer: pfaffian el={el_p:.6f}  bravyi el={el_b:.6f}  "
              f"(agree? {np.allclose(el_b, el_p, rtol=1e-5, atol=1e-6)})")
        self.assertGreaterEqual(el_b, -1e-6, msg=f"bravyi el_energy {el_b} < 0")

    def test_d6_2layer_bravyi_nonnegative(self):
        # large_paramvec2: residual bug -> pfaffian unphysical (~-26.7); bravyi must be physical.
        el_p, el_b = self._run("large_paramvec2.npy", 2)
        print(f"\nD6 2-layer: pfaffian el={el_p:.6f}  bravyi el={el_b:.6f}")
        self.assertGreaterEqual(el_b, -1e-6, msg=f"bravyi el_energy {el_b} < 0")
