from abc import ABC, abstractmethod


class BackendBase(ABC):
    """Abstract base class for the backend.

    This class defines the interface for the backend, which can be implemented
        1) using different libraries (e.g., numpy, jax)
        2) for different groups (e.g., Z2, Dn).
    At present, backends do not differ across groups.
    """

    def __init__(self):
        """Initialize the backend."""
        pass

    @staticmethod
    def slice_matrix(mat, a, b, c, d):
        return mat[a:b, c:d]

    @staticmethod
    def take_block(mat, start, size, axis):
        """Contiguous block of ``size`` indices starting at ``start`` along ``axis``.
        ``size`` and ``axis`` must be static; ``start`` may be a jax tracer (jax backend)."""
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    def dynamic_update_slice(mat, inds, vals):
        """Set mat[inds] = vals in a jax.jit compatible way."""
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    def array_assign(mat, inds, val):
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    def array_add(mat, inds, val):
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    def array_mult(mat, inds, val):
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    def pfaffian(mat):
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    def pfaffian_vectorized(mat):
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    @abstractmethod
    def calculate_lognormvec(gamma_in_sys_vec, mat_d_vec, all_factors=False):
        raise NotImplementedError("This is an abstract method. Implement in child class please.")
