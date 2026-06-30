"""Bravyi & Gosset three-state Gaussian-overlap identity (Commun. Math. Phys. 356, 451 (2017),
eqs. 24-25 with |\phi_0> being the vacuum),
can be used to compute the pure gauge amplitudes psi(G) = <B(G)|A>.
This can be used to compute the electric energies for pure guge states
independently of the pfaffian expansion method.

Convention: unlike our convrntion,
covariance matrices follow Bravyi's M_{pq} = (-i/2)<[c_p, c_q]>
with c_1 = c+c^\dagger c_2 = -i(c-c^\dagger)
(both c_2 and the covariance matrix have opposite signs from our convention).

The vacuum covariance
is M0 = (+) [[0,1],[-1,0]] per mode.
"""

from ggpeps import xnp
from ggpeps.system.backend import backend  # the backend INSTANCE (exposes .pfaffian)


def _antisymmetrize(x: xnp.ndarray) -> xnp.ndarray:
    """Project onto the antisymmetric part. The matrices fed to the Pfaffian (M1+M2 and
    Delta+M0) are antisymmetric in exact arithmetic, but Delta = (...)(M1+M2)^-1 accumulates
    round-off from inv/expm; pfapack asserts antisymmetry to 1e-14, so we clean the round-off.
    This does not change the true Pfaffian (only removes the symmetric round-off component)."""
    return 0.5 * (x - xnp.transpose(x))


def vacuum_covmat(dim: int) -> xnp.ndarray:
    """Covariance matrix of the fermionic vacuum |Omega>, size dim x dim (dim even).

    M0 = blockdiag([[0, 1], [-1, 0]]) over dim/2 modes.
    """
    assert dim % 2 == 0, "covariance matrix dimension must be even"
    m0 = xnp.zeros((dim, dim), dtype=xnp.complex128)
    idx = xnp.arange(0, dim, 2)
    m0 = backend.array_assign(m0, (idx, idx + 1), 1.0)
    m0 = backend.array_assign(m0, (idx + 1, idx), -1.0)
    return m0


def to_bravyi_covmat(gamma: xnp.ndarray) -> xnp.ndarray:
    """Convert a covariance matrix from OUR convention to Bravyi's.

    Our convention: Gamma_{ab} = (+i/2)<[gamma_a, gamma_b]>
    with gamma^(1) = a + a^dagger and gamma^(2) = i(a - a^dagger).
    Bravyi (Commun. Math. Phys. 356, 451 (2017)): M_{pq} = (-i/2)<[c_p, c_q]> (eq. 17) with
    c_{2j-1} = a_j + a_j^dagger and c_{2j} = -i(a_j - a_j^dagger) (eq. 1).

    The two share the first Majorana (gamma^(1) = c_{2j-1}) but our second is MINUS Bravyi's
    (gamma^(2)_j = -c_{2j}), i.e. c = S gamma with S = diag(1, -1, 1, -1, ...). Combining the
    (-i/2 vs +i/2) prefactor (a global minus) with S on each index gives the exact map

        M = -S Gamma S,   i.e.   M_{pq} = -s_p s_q Gamma_{pq}

    - a global sign, plus a sign flip on every entry coupling a first-type to a second-type
    Majorana.
    """
    dim = gamma.shape[-1]
    sign = xnp.where(xnp.arange(dim) % 2 == 0, 1.0, -1.0).astype(gamma.dtype)
    return -(sign[:, None] * gamma * sign[None, :])


def gaussian_overlap_chi(m1: xnp.ndarray, m2: xnp.ndarray, m0: xnp.ndarray) -> xnp.ndarray:
    """The G-dependent Pfaffian product chi = Pf(M1 + M2) * Pf(Delta + M0), where
    Delta = (-2 I + i M1 - i M2) (M1 + M2)^{-1}  (Bravyi eqs. 24-25).

    Returns a complex scalar. The full triple-overlap product is 4^{-n} * chi with n = dim/2.
    """
    dim = m1.shape[0]
    eye = xnp.eye(dim, dtype=xnp.complex128)
    m1m2 = m1 + m2
    delta = (-2.0 * eye + 1j * m1 - 1j * m2) @ xnp.linalg.inv(m1m2)
    return backend.pfaffian(_antisymmetrize(m1m2)) * backend.pfaffian(_antisymmetrize(delta + m0))


def overlap_magnitude_sq(m1: xnp.ndarray, m2: xnp.ndarray) -> xnp.ndarray:
    """|<phi_1|phi_2>|^2 = 2^{-n} Pf(M1 + M2)  (Bravyi eq 22, parity sigma = +1)."""
    n = m1.shape[0] // 2
    return (2.0 ** (-n)) * backend.pfaffian(_antisymmetrize(m1 + m2))
