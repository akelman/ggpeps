############## JAX VERSIONS ##############
from typing import Optional
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap

# The following line ensures that JAX is configured to 64-bit precision.
# Without this line, some of the precision tests do not pass.
jax.config.update("jax_enable_x64", True)

import py_pfaffian.jax

from ggpeps.system.backend_base import BackendBase


def derivative_pfaffian_jax(mat: jnp.ndarray, d_mat: jnp.ndarray, pfaval: Optional[float] = None):
    """Compute the derivative of a Pfaffian of a matrix A.
    The explicit derivative dA/dx is given as a second argument

    The given formula is only valid if A is not singular.

    Args:
        mat (jnp.ndarray): Input Matrix A
        d_mat (jnp.ndarray): Derivative dA/dx
        pfaval (Optional[float]): Pfaffian value of mat, if already known. If None, it will be computed.

    Returns:
        float: d(Pf(A))/dx
    """
    if pfaval is None:
        pfaval = py_pfaffian.jax.pfaffian(mat)

    return 0.5 * pfaval * jnp.trace(jnp.linalg.inv(mat) @ d_mat)


@jax.jit
def jax_pfaffian(mat):
    # The jax pfaffian implementation does not properly guard against singular matrices, in which case
    # it returns nan's. We replace these with 0's.
    val = py_pfaffian.jax.pfaffian(mat)
    return jnp.where(jnp.isnan(val), 0.0, val)


pfaffian_batched = jit(vmap(jax_pfaffian, in_axes=0, out_axes=0))


class BackendJax(BackendBase):
    """Backend for jax."""

    backend_type = "jax"

    def __init__(self) -> None:
        pass

    @staticmethod
    def array_assign(mat, inds, val):
        mat = mat.at[inds].set(val)
        return mat

    @staticmethod
    def take_block(mat, start, size, axis):
        # Gather with a traced index vector: `start` may be a tracer -> jitted callers compile once.
        return jnp.take(mat, start + jnp.arange(size), axis=axis)

    @staticmethod
    def dynamic_update_slice(mat, inds, vals):
        """Set mat[inds] = vals in a jax.jit compatible way"""
        return jax.lax.dynamic_update_slice(mat, vals, inds)

    @staticmethod
    def array_add(mat, inds, val):
        mat = mat.at[inds].add(val)
        return mat

    @staticmethod
    def array_mult(mat, inds, val):
        mat = mat.at[inds].multiply(val)
        return mat

    @staticmethod
    def pfaffian(mat):
        return jax_pfaffian(mat)

    @staticmethod
    def pfaffian_vectorized(mat):
        return pfaffian_batched(mat)
