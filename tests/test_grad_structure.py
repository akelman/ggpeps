"""Structural guards for the gradient hot path.

The optimized gradient code replaces the dense triple products

    prod          = mat_d_inv     @ wi_gamma_in     @ gamma_in_sys        (norm gradient)
    prod_mod_norm = mat_d_mod_inv @ wi_gamma_in_mod @ gamma_in_sys_mod    (electric gradient)

with ``-wi_gamma_out(_mod)`` via the pure-state identity
``(1 - Gamma D)^-1 Gamma = -(D + Gamma)^-1``, valid iff ``Gamma^2 = -1``.
These tests pin both the purity of the (gauged) link covariance and the identity itself,
against explicitly computed products, after genuine single-link gauge updates.
"""
import unittest

import numpy as np

from ggpeps import lattice, system
from ggpeps.system.system_D2n import D2nSystem2D
from ggpeps.system.system_Z2 import Z2System2D

TOL = 1e-10


def _build_d6(seed=3, num_pg_layer=2):
    lat = lattice.Lattice2D(2, 2)
    cfg = system.D6System2D_Config(
        lat, 1, 1, 0, 0, None, ncopy=2, num_pg_layer=num_pg_layer, num_fermionic_layer=0, mod_link_inds=(0,)
    )
    rng = np.random.RandomState(seed)
    cfg.paramvec = rng.rand(*cfg.param_shape())
    cfg.enforce_parameter_conditions(cfg.paramvec)
    return cfg, D2nSystem2D(cfg), rng


def _build_z2(seed=7, num_pg_layer=1):
    lat = lattice.Lattice2D(2, 2)
    cfg = system.Z2System2D_G2C_F2C_Config(
        lat, 1.0, 1.0, 0.0, 0.0, np.zeros(0), num_pg_layer=num_pg_layer, num_fermionic_layer=0, mod_link_inds=(0,)
    )
    rng = np.random.RandomState(seed)
    cfg.paramvec = rng.rand(*cfg.param_shape())
    cfg.enforce_parameter_conditions(cfg.paramvec)
    return cfg, Z2System2D(cfg), rng


def _randomize_gauge(cfg, sys_, rng, nsteps=25):
    gvals = cfg.gaugemgr.get_possible_gauge_values()
    for _ in range(nsteps):
        link = int(rng.randint(0, cfg.lattice.nlinks))
        sys_.update_gauge_ind(link, gvals[int(rng.randint(0, len(gvals)))])


def _dense_grad_over_norm_reference(sys_):
    """Straightforward reference for the norm gradient: grad_a = -0.5 Tr(dD_a @ prod) with the
    full dense system derivative and prod = -wi_gamma_out (the identity checked above). The
    block-structured compute_grad_over_norm_vec must reproduce this."""
    cfg = sys_.cfg
    prod_vec = -np.asarray(sys_.wi_gamma_out_vec)
    dense = np.asarray(sys_.gamma_maj_sys_deriv_layvec_ucvec_symbvec)
    offset = 2 * cfg.lattice.size * cfg.nphysmodes_site  # skip the physical modes
    grads = np.zeros((cfg.nlayer, cfg.unitcell_size, len(cfg.symbolvec)))
    for lay in range(cfg.nlayer):
        for uc in range(cfg.unitcell_size):
            for a in range(len(cfg.symbolvec)):
                deriv_virt = dense[lay, uc, a][offset:, offset:]
                grads[lay, uc, a] = np.real(-0.5 * np.trace(deriv_virt @ prod_vec[lay]))
    for lay, uc_ind, symbol_ind in cfg.zeroed_params:
        grads[lay, uc_ind, symbol_ind] = 0.0
    return grads


class GradStructureChecks:
    """Mixin with the actual assertions; subclasses provide the system."""

    def _check(self, cfg, sys_, rng):
        sys_.initialize()
        _randomize_gauge(cfg, sys_, rng)

        gamma = np.asarray(sys_.gamma_in_sys_vec)
        gamma_mod = np.asarray(sys_.gamma_in_sys_mod_vec)
        eye = np.eye(gamma.shape[-1])
        eye_mod = np.eye(gamma_mod.shape[-1])

        # Purity of the gauged link covariance: Gamma^2 = -1 (closed and open-link/mod).
        self.assertLess(np.abs(np.einsum("lij,ljk->lik", gamma, gamma) + eye).max(), TOL)
        self.assertLess(np.abs(np.einsum("lmij,lmjk->lmik", gamma_mod, gamma_mod) + eye_mod).max(), TOL)

        # Closed identity: mat_d_inv @ wi_gamma_in @ gamma_in_sys == -wi_gamma_out.
        prod = np.asarray(sys_.mat_d_inv_vec) @ np.asarray(sys_.wi_gamma_in_vec) @ gamma
        self.assertLess(np.abs(prod + np.asarray(sys_.wi_gamma_out_vec)).max(), TOL)

        # Mod identity: mat_d_mod_inv @ wi_gamma_in_mod @ gamma_in_sys_mod == -wi_gamma_out_mod.
        prod_mod = np.asarray(sys_.mat_d_mod_inv_vec) @ np.asarray(sys_.wi_gamma_in_mod_vec) @ gamma_mod
        self.assertLess(np.abs(prod_mod + np.asarray(sys_.wi_gamma_out_mod_vec)).max(), TOL)

    def _check_grad_over_norm(self, cfg, sys_, rng):
        sys_.initialize()
        _randomize_gauge(cfg, sys_, rng)
        block = np.asarray(sys_.grad_over_norm_vec)
        dense = _dense_grad_over_norm_reference(sys_)
        scale = max(np.abs(dense).max(), 1.0)
        self.assertLess(np.abs(block - dense).max() / scale, TOL)


class TestGradStructureD6(unittest.TestCase, GradStructureChecks):
    def test_identity_two_layers(self):
        self._check(*_build_d6(seed=3, num_pg_layer=2))

    def test_identity_one_layer(self):
        self._check(*_build_d6(seed=11, num_pg_layer=1))

    def test_grad_over_norm_matches_dense_reference(self):
        self._check_grad_over_norm(*_build_d6(seed=5, num_pg_layer=2))


class TestGradStructureZ2(unittest.TestCase, GradStructureChecks):
    def test_identity(self):
        self._check(*_build_z2(seed=7, num_pg_layer=1))

    def test_grad_over_norm_matches_dense_reference(self):
        self._check_grad_over_norm(*_build_z2(seed=13, num_pg_layer=1))


if __name__ == "__main__":
    unittest.main()
