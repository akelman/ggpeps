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


class TestGradStructureD6(unittest.TestCase, GradStructureChecks):
    def test_identity_two_layers(self):
        self._check(*_build_d6(seed=3, num_pg_layer=2))

    def test_identity_one_layer(self):
        self._check(*_build_d6(seed=11, num_pg_layer=1))


class TestGradStructureZ2(unittest.TestCase, GradStructureChecks):
    def test_identity(self):
        self._check(*_build_z2(seed=7, num_pg_layer=1))


if __name__ == "__main__":
    unittest.main()
