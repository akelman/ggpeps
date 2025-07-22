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

    @classmethod
    def calculate_lognorm(
        cls, gamma_in_sys_vec: list[xnp.ndarray], mat_d_vec: list[xnp.ndarray], all_factors: bool = False
    ) -> float:
        # This is still the plain formula, without any update mechanism
        normvec = cls.calculate_lognormvec(gamma_in_sys_vec, mat_d_vec, all_factors=all_factors)
        return xnp.sum(normvec)

    @staticmethod
    def calculate_lognormvec_inc(incdet_vec, det_mat_d_vec, n, all_factors: bool = False):
        dest = []
        for ind in range(len(incdet_vec)):
            detval = incdet_vec[ind].det()
            if all_factors:
                detval -= n * xnp.log(2)
                detval += det_mat_d_vec[ind]
            # The factor 0.5 is the sqrt of the formula. We are storing the logarithm of the norm.
            # The addition of the cumval is the multiplication of the indpendent PEPS
            dest.append(0.5 * detval)
        return xnp.array(dest)

    @classmethod
    def calculate_lognorm_inc(cls, incdet_vec, det_mat_d_vec, n, all_factors: bool = False):
        lognormvec = cls.calculate_lognormvec_inc(incdet_vec, det_mat_d_vec, n, all_factors=all_factors)
        return xnp.sum(lognormvec)
