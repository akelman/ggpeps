import unittest
import numpy as np
import scipy.linalg as sla

from ggpeps.system import bravyi_overlap as bo
from ggpeps.system.backend import backend
from ggpeps import lattice
from ggpeps import system


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
