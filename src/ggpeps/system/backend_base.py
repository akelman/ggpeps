from abc import ABC, abstractmethod


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
    @abstractmethod
    def slice_matrix(mat, a, b, c, d):
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    @abstractmethod
    def extract_partial_covmats(mat, corner):
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    @abstractmethod
    def calculate_lognormvec(gamma_in_sys_vec, mat_d_vec, all_factors=False):
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    @abstractmethod
    def compute_grad_over_norm(gamma_in_sys, diff, deriv_d, mat_d_inv):
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    @abstractmethod
    def compute_el_grad_vec(system):
        raise NotImplementedError("This is an abstract method. Implement in child class please.")

    @staticmethod
    @abstractmethod
    def gamma_in_sys_mod(gamma_in_sys, single_link_offset):
        raise NotImplementedError("This is an abstract method. Implement in child class please.")
