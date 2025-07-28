from abc import ABC, abstractmethod

from ggpeps import xnp as xnp


class BackendBase(ABC):
    """Abstract base class for the backend.

    This class defines the interface for the backend, which can be implemented
        1) using different libraries (e.g., numpy, jax)
        2) for different groups (e.g., Z2, Dn).
    """

    def __init__(self):
        """Initialize the backend."""
        pass

    @staticmethod
    def slice_matrix(mat, a, b, c, d):
        return mat[a:b, c:d]

    @staticmethod
    def array_assign(mat, inds, val):
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    @abstractmethod
    def calculate_lognormvec(gamma_in_sys_vec, mat_d_vec, all_factors=False):
        raise NotImplementedError("This is an abstract method. Implement in child class please.")
