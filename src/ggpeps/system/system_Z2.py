import logging

from ggpeps import xnp as xnp

import ggpeps
from ggpeps.system.backend import backend

from .system_base import System2DBase
from .system_base import maybe_jit

logger = logging.getLogger(ggpeps.LOGGER_NAME)


###################### Z2System2D ##########################


class Z2System2D(System2DBase):
    """2D Z2 system GGPEPS ansatz with physical fermions.

    Some general notes about conventions:

    Order of the paramvec: see the functions that create the symbolvec in the configs.
    We split the real and the imaginary part of the parameters into independent variables.
    Mode order of tmat:
        {p,l1,r1,d1,u1,l2,r2,d2,u2,l3,r3... and so on}
    Mode order of gamma_dirac:
        {p,l1,r1,d1,u1,l2,r2,d2,u2,p_dag,l1_dag,r1_dag,u1_dag,d1_dag,l2_dat,r2_dag,u2_dag,d2_dag,l3,r3...and so on}
    Mode order of gamma_maj:
        {p_1,p_2,l1_1,l1_2,r1_1,r1_2,d1_1,d1_2,u1_1,u1_2,l2_1,l2_2,r2_1,r2_2,d2_1,d2_2,u2_1,u2_2,l3_1,l3_2...and so on}
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    ################## Observables ##################
    @staticmethod
    @maybe_jit(static_argnames=["use_trans_inv"])
    def _compute_mass_energy_op_vec(
        occupations_after_ph: xnp.ndarray,
        use_trans_inv: bool = False,
    ) -> xnp.ndarray:

        if use_trans_inv:
            logger.warning(
                "Translation invariance need not be assumed for the mass energy operator."
                "Doing so gives negligible speedup, so we proceed without it."
            )

        mass_energy_op = xnp.sum(occupations_after_ph, axis=1)  # sum over sites

        return mass_energy_op

    @staticmethod
    @maybe_jit(
        static_argnames=[
            "lattice_size",
            "symbolvec",
            "unitcell_size",
            "use_trans_inv",
            "num_pg_layer",
            "num_fermionic_layer",
            "zeroed_params",
        ],
    )
    def _compute_mass_energy_grad(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        unitcell_size: int,
        symbolvec: tuple,
        d_gamma_out_symbolvec: xnp.ndarray,
        zeroed_params: tuple,
        use_trans_inv: bool = True,
    ) -> xnp.ndarray:

        if not use_trans_inv:
            raise NotImplementedError("Translation invariance must be set to True.")

        nlayer = num_pg_layer + num_fermionic_layer
        param_shape = (nlayer, unitcell_size, len(symbolvec))
        gradients = xnp.zeros(param_shape)

        for layer_ind in range(num_pg_layer, nlayer):
            # only the fermionic layers directly contribute to the mass

            for site_ind in range(0, 2 * lattice_size, 2):

                for uc_ind in range(unitcell_size):
                    for symbol_ind, symbol in enumerate(symbolvec):
                        # the derivative calculation is relatively compuationally expensive
                        # (though less than for electric energy)
                        # we can skip it for parameters that are forced by the ansatz to be zero
                        if (layer_ind, uc_ind, symbol_ind) not in zeroed_params:

                            d_gamma_out = d_gamma_out_symbolvec[layer_ind, uc_ind, symbol_ind]
                            grad = 0.5 * d_gamma_out[site_ind + 1, site_ind]
                            gradients = backend.array_add(gradients, (layer_ind, uc_ind, symbol_ind), grad)

                    # further terms of the derivative are included higher up in the computation stack
                    # because computing them requires knowing various expectation values, which are not available here

        return gradients

    @staticmethod
    @maybe_jit(
        static_argnames=[
            "lattice_size",
            "num_pg_layer",
            "num_fermionic_layer",
            "horizontal_neighbor_data",
            "vertical_neighbor_data",
        ],
    )
    def _compute_int_energy_op_vec(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        gaugefieldvec: xnp.ndarray,
        ferm_covmat_vec: xnp.ndarray,
        horizontal_neighbor_data: tuple,
        vertical_neighbor_data: tuple,
    ) -> xnp.ndarray:
        """
        Note: this function assumes that U = U^dagger, which is only valid for Z2.
        For other groups, the calculation will not be as simple.
        """

        nlayer = num_pg_layer + num_fermionic_layer
        int_energy_op = xnp.zeros(nlayer)

        for layer_ind in range(num_pg_layer, nlayer):
            layer_int_energy = 0.0
            covmat = ferm_covmat_vec[layer_ind]

            for site_ind in range(lattice_size):
                site_ind_cov = 2 * site_ind  # index into covariance matrix, factor of 2 for Majorana modes per site

                # Horizontal link
                hor_link_ind = horizontal_neighbor_data[site_ind][0]
                neighborX_ind = 2 * horizontal_neighbor_data[site_ind][1]  # 2 * index of neighboring site

                gaugefield_hor = gaugefieldvec[hor_link_ind]  # a matrix representation of the group element
                cos_factor_hor = xnp.real(gaugefield_hor[0][0]).astype(
                    float
                )  # get U from gauge representation, this handles cosine
                hor_energy = 0.5 * (covmat[site_ind_cov, neighborX_ind] - covmat[site_ind_cov + 1, neighborX_ind + 1])
                layer_int_energy += hor_energy * cos_factor_hor

                # Vertical link
                vert_link_ind = vertical_neighbor_data[site_ind][0]
                neighborY_ind = 2 * vertical_neighbor_data[site_ind][1]

                gaugefield_vert = gaugefieldvec[vert_link_ind]
                cos_factor_vert = xnp.real(gaugefield_vert[0][0]).astype(float)
                vert_energy = 0.5 * (covmat[site_ind_cov, neighborY_ind + 1] + covmat[site_ind_cov + 1, neighborY_ind])
                layer_int_energy -= vert_energy * cos_factor_vert

            int_energy_op = backend.array_assign(int_energy_op, layer_ind, layer_int_energy)

        return int_energy_op

    @staticmethod
    @maybe_jit(
        static_argnames=[
            "lattice_size",
            "num_pg_layer",
            "num_fermionic_layer",
            "unitcell_size",
            "nparams",
            "horizontal_neighbor_data",
            "vertical_neighbor_data",
            "zeroed_params",
        ],
    )
    def _compute_int_energy_grad(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        unitcell_size: int,
        nparams: int,
        gaugefieldvec: xnp.ndarray,
        d_gamma_out_symbolvec: xnp.ndarray,
        horizontal_neighbor_data: tuple,
        vertical_neighbor_data: tuple,
        zeroed_params: tuple,
    ) -> xnp.ndarray:
        """
        Note: this function assumes that U = U^dagger, which is only valid for Z2.
        For other groups, the calculation will not be as simple.
        """

        nlayer = num_pg_layer + num_fermionic_layer
        param_shape = (nlayer, unitcell_size, nparams)
        gradients = xnp.zeros(param_shape)

        for layer_ind in range(num_pg_layer, nlayer):

            for site_ind in range(lattice_size):
                site_ind_cov = 2 * site_ind  # index into covariance matrix, factor of 2 for Majorana modes per site

                # Horizontal link
                hor_link_ind = horizontal_neighbor_data[site_ind][0]
                neighborX_ind = 2 * horizontal_neighbor_data[site_ind][1]  # 2 * index of neighboring site

                gaugefield_hor = gaugefieldvec[hor_link_ind]  # a matrix representation of the group element
                cos_factor_hor = xnp.real(gaugefield_hor[0][0]).astype(
                    float
                )  # get U from gauge representation, this handles cosine

                # Vertical link
                vert_link_ind = vertical_neighbor_data[site_ind][0]
                neighborY_ind = 2 * vertical_neighbor_data[site_ind][1]

                gaugefield_vert = gaugefieldvec[vert_link_ind]
                cos_factor_vert = xnp.real(gaugefield_vert[0][0]).astype(float)

                # Calculate derivatives
                for uc_ind in range(unitcell_size):
                    for symbol_ind in range(nparams):
                        # the derivative calculation is relatively compuationally expensive
                        # (though less than for electric energy)
                        # we can skip it for parameters that are forced by the ansatz to be zero
                        if (layer_ind, uc_ind, symbol_ind) not in zeroed_params:

                            d_gamma_out = d_gamma_out_symbolvec[layer_ind, uc_ind, symbol_ind]
                            grad = (
                                0.5
                                * cos_factor_hor
                                * (
                                    d_gamma_out[site_ind_cov, neighborX_ind]
                                    - d_gamma_out[site_ind_cov + 1, neighborX_ind + 1]
                                )
                            )
                            grad += (
                                -0.5
                                * cos_factor_vert
                                * (
                                    d_gamma_out[site_ind_cov, neighborY_ind + 1]
                                    + d_gamma_out[site_ind_cov + 1, neighborY_ind]
                                )
                            )
                            gradients = backend.array_add(gradients, (layer_ind, uc_ind, symbol_ind), grad)

        return gradients

    @staticmethod
    @maybe_jit(static_argnames=[])
    def _compute_chem_energy_op_vec(
        occupations_before_ph: xnp.ndarray,
    ) -> xnp.ndarray:

        chem_energy_op = xnp.sum(occupations_before_ph, axis=1)

        return chem_energy_op

    @staticmethod
    @maybe_jit(
        static_argnames=[
            "lattice_size",
            "num_pg_layer",
            "num_fermionic_layer",
            "unitcell_size",
            "symbolvec",
            "sublattice_factors",
            "zeroed_params",
        ],
    )
    def _compute_chem_energy_grad(
        lattice_size: int,
        num_pg_layer: int,
        num_fermionic_layer: int,
        unitcell_size: int,
        symbolvec: tuple,
        sublattice_factors: tuple,
        zeroed_params: tuple,
        d_gamma_out_vec: xnp.ndarray,
    ) -> xnp.ndarray:

        nlayer = num_pg_layer + num_fermionic_layer
        param_shape = (nlayer, unitcell_size, len(symbolvec))
        gradients = xnp.zeros(param_shape)

        for layer_ind in range(num_pg_layer, nlayer):
            # only the fermionic layers directly contribute to the chemical potential

            # Calculate chem term
            # Since we set the system to have different parameters on the even and odd sites when using a non-zero
            # chemical potential (i.e. the system is translationally invariant by two sites),
            # we could just calculate it for one even and one odd site and multiply by the size of the system
            for site in range(lattice_size):
                site_ind = 2 * site  # index into covariance matrix
                site_factor = sublattice_factors[site]  # even or odd sublattice

                for uc_ind in range(unitcell_size):
                    for symbol_ind, symbol in enumerate(symbolvec):
                        # the derivative calculation is relatively compuationally expensive
                        # (though less than for electric energy)
                        # we can skip it for parameters that are forced by the ansatz to be zero
                        if (layer_ind, uc_ind, symbol_ind) not in zeroed_params:

                            d_gamma_out = d_gamma_out_vec[layer_ind, uc_ind, symbol_ind]
                            grad = 0.5 * site_factor * d_gamma_out[site_ind + 1, site_ind]
                            gradients = backend.array_add(gradients, (layer_ind, uc_ind, symbol_ind), grad)

                    # further terms of the derivative are included higher up in the computation stack
                    # because computing them requires knowing various expectation values, which are not available here

        return gradients

    def _meson_string_vec(self, path: tuple[tuple[int, bool], ...]) -> xnp.ndarray:

        meson_op_vec = xnp.zeros(self.cfg.nlayer)

        # value of the fields
        path_factor = self.compute_path(path)

        # indices into the covariance matrices at the start and end of the path
        # TODO: it is a waste to calculate this for every gauge config - instead, this function should accept
        #       as input the start and end site indices
        start_site_ind, end_site_ind = self.cfg.lattice.get_path_endpoints(path)
        site_ind_cov_in = 2 * start_site_ind
        site_ind_cov_fin = 2 * end_site_ind

        for layer_ind in range(self.cfg.num_pg_layer, self.cfg.nlayer):
            covmat = self.ferm_covmat_vec[layer_ind]

            # Since for the L-shaped strings considered here the endpoints are always on the same sublattice,
            # we still have \psi^\dagger \psi after the PH transformation
            layer_val = (
                0.25
                * path_factor
                * (
                    -1j * covmat[site_ind_cov_in, site_ind_cov_fin]
                    - 1j * covmat[site_ind_cov_in + 1, site_ind_cov_fin + 1]
                    + covmat[site_ind_cov_in + 1, site_ind_cov_fin]
                    - covmat[site_ind_cov_in, site_ind_cov_fin + 1]
                )
            )

            # TODO: is the absolute value necessary? why?
            meson_op_vec = backend.array_assign(meson_op_vec, layer_ind, xnp.abs(layer_val))
        return meson_op_vec

    @staticmethod
    @maybe_jit(static_argnames=["after_ph"])
    def occupation(covmat: xnp.ndarray, site: int, site_coord: tuple[int, int], after_ph: bool = False) -> float:

        site_ind = 2 * site  # index into covariance matrix

        x, y = site_coord
        if after_ph:
            site_factor = 1
        else:
            site_factor = (-1) ** (x + y)  # even or odd sublattice

        mass_site = 0.5 * (1 + site_factor * covmat[site_ind + 1, site_ind])

        return mass_site
