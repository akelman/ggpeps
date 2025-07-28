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
    def compute_el_grad_vec(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        unitcell_size: int,
        nvirtmodes_link: int,
        nphysmodes_site: int,
        symbolvec: tuple,
        overall_factors,
        idxarr_vec,
        el_energy_vec,
        mat_b_mod_vec,
        gamma_in_sys_mod_vec,
        covmat_out_virt_vec,
        norm_mod_vec,
        lognorm_default_vec,
        wi_gamma_in_mod_inv_vec,
        wi_gamma_out_mod_inv_vec,
        mat_d_mod_inv_vec,
        gamma_maj_sys_deriv_layvec_ucvec_symbvec,
        grad_over_norm_vec,
        zeroed_params,
    ):
        raise NotImplementedError("This is an abstract method. Implement in child class please.")
