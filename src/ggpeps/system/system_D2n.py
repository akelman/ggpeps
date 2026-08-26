import logging

from ggpeps import xnp as xnp

import ggpeps
from ggpeps.system.backend import backend

from .system_base import System2DBase
from .config_D6_2d import D6System2D_Config


from .system_base import maybe_jit

logger = logging.getLogger(ggpeps.LOGGER_NAME)


###################### D2nSystem2D ##########################


class D2nSystem2D(System2DBase):
    """2D Z2 system GGPEPS ansatz with physical fermions.

    Some general notes about conventions:

    Order of the paramvec: see the functions that create the symbolvec in the configs.
        We split the real and the imaginary part of the parameters into independent variables.
    Mode order of tmat: {p,l1,r1,d1,u1,l2,r2,d2,u2,l3,r3...}.
    Mode order of gamma_dirac:
        {p,l1,r1,d1,u1,l2,r2,d2,u2,p_dag,l1_dag,r1_dag,u1_dag,d1_dag,l2_dat,r2_dag,u2_dag,d2_dag,l3,r3...}.
    Mode order of gamma_maj:
        {p_1,p_2,l1_1,l1_2,r1_1,r1_2,d1_1,d1_2,u1_1,u1_2,l2_1,l2_2,r2_1,r2_2,d2_1,d2_2,u2_1,u2_2,l3_1,l3_2...}.
    """

    def __init__(self, cfg: D6System2D_Config):
        self.cfg: D6System2D_Config
        super().__init__(cfg)

    # Observables
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

        mass_energy_op = xnp.sum(occupations_after_ph, axis=(1, 2))  # sum over sites and colors
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

        ncolors = 2  # the faithful D2n representation is 2-dimensional
        nlayer = num_pg_layer + num_fermionic_layer
        param_shape = (nlayer, unitcell_size, len(symbolvec))
        gradients = xnp.zeros(param_shape)

        for layer_ind in range(num_pg_layer, nlayer):
            # only the fermionic layers directly contribute to the mass

            for site_ind in range(lattice_size):
                for uc_ind in range(unitcell_size):
                    for symbol_ind, symbol in enumerate(symbolvec):
                        # we can skip parameters that are forced by the ansatz to be zero
                        if (layer_ind, uc_ind, symbol_ind) not in zeroed_params:

                            d_gamma_out = d_gamma_out_symbolvec[layer_ind, uc_ind, symbol_ind]
                            grad = 0.0
                            for color in range(ncolors):
                                color_ind = 2 * ncolors * site_ind + 2 * color
                                grad += 0.5 * d_gamma_out[color_ind + 1, color_ind]
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
        Hopping energy: each color pair (m, n) contributes the single-color bilinear weighted
        by the representation matrix element D_mn of the link.
        Note: assumes a real faithful representation (true for D2n); a complex representation
        would contribute additional imaginary-part terms.
        """
        ncolors = 2  # the faithful D2n representation is 2-dimensional

        nlayer = num_pg_layer + num_fermionic_layer
        int_energy_op = xnp.zeros(nlayer)

        for layer_ind in range(num_pg_layer, nlayer):
            layer_int_energy = 0.0
            covmat = ferm_covmat_vec[layer_ind]

            for site_ind in range(lattice_size):
                site_ind_cov = 2 * ncolors * site_ind  # index into covariance matrix

                # Horizontal link
                hor_link_ind = horizontal_neighbor_data[site_ind][0]
                neighborX_ind = 2 * ncolors * horizontal_neighbor_data[site_ind][1]
                gaugefield_hor = xnp.real(gaugefieldvec[hor_link_ind])  # D(g) of the link, real 2x2 for D2n

                # Vertical link
                vert_link_ind = vertical_neighbor_data[site_ind][0]
                neighborY_ind = 2 * ncolors * vertical_neighbor_data[site_ind][1]
                gaugefield_vert = xnp.real(gaugefieldvec[vert_link_ind])

                for m in range(ncolors):
                    site_color_ind = site_ind_cov + 2 * m
                    for n in range(ncolors):
                        neighborX_color_ind = neighborX_ind + 2 * n
                        neighborY_color_ind = neighborY_ind + 2 * n

                        hor_energy = 0.5 * (
                            covmat[site_color_ind, neighborX_color_ind]
                            - covmat[site_color_ind + 1, neighborX_color_ind + 1]
                        )
                        layer_int_energy += hor_energy * gaugefield_hor[m][n]

                        vert_energy = 0.5 * (
                            covmat[site_color_ind, neighborY_color_ind + 1]
                            + covmat[site_color_ind + 1, neighborY_color_ind]
                        )
                        layer_int_energy -= vert_energy * gaugefield_vert[m][n]

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
        """Same color-contracted hopping formula as _compute_int_energy_op_vec, with the
        covariance matrix replaced by its parameter derivative (also assumes a real
        faithful representation)."""
        ncolors = 2  # the faithful D2n representation is 2-dimensional
        nlayer = num_pg_layer + num_fermionic_layer
        param_shape = (nlayer, unitcell_size, nparams)
        gradients = xnp.zeros(param_shape)

        for layer_ind in range(num_pg_layer, nlayer):

            for site_ind in range(lattice_size):
                site_ind_cov = 2 * ncolors * site_ind  # index into covariance matrix

                # Horizontal link
                hor_link_ind = horizontal_neighbor_data[site_ind][0]
                neighborX_ind = 2 * ncolors * horizontal_neighbor_data[site_ind][1]
                gaugefield_hor = xnp.real(gaugefieldvec[hor_link_ind])  # D(g) of the link, real 2x2

                # Vertical link
                vert_link_ind = vertical_neighbor_data[site_ind][0]
                neighborY_ind = 2 * ncolors * vertical_neighbor_data[site_ind][1]
                gaugefield_vert = xnp.real(gaugefieldvec[vert_link_ind])

                # Calculate derivatives
                for uc_ind in range(unitcell_size):
                    for symbol_ind in range(nparams):
                        # we can skip parameters that are forced by the ansatz to be zero
                        if (layer_ind, uc_ind, symbol_ind) not in zeroed_params:

                            d_gamma_out = d_gamma_out_symbolvec[layer_ind, uc_ind, symbol_ind]
                            grad = 0.0
                            for m in range(ncolors):
                                site_color_ind = site_ind_cov + 2 * m
                                for n in range(ncolors):
                                    neighborX_color_ind = neighborX_ind + 2 * n
                                    neighborY_color_ind = neighborY_ind + 2 * n

                                    grad += (
                                        0.5
                                        * gaugefield_hor[m][n]
                                        * (
                                            d_gamma_out[site_color_ind, neighborX_color_ind]
                                            - d_gamma_out[site_color_ind + 1, neighborX_color_ind + 1]
                                        )
                                    )
                                    grad += (
                                        -0.5
                                        * gaugefield_vert[m][n]
                                        * (
                                            d_gamma_out[site_color_ind, neighborY_color_ind + 1]
                                            + d_gamma_out[site_color_ind + 1, neighborY_color_ind]
                                        )
                                    )
                            gradients = backend.array_add(gradients, (layer_ind, uc_ind, symbol_ind), grad)

        return gradients

    @staticmethod
    @maybe_jit(static_argnames=[])
    def _compute_chem_energy_op_vec(
        occupations_before_ph: xnp.ndarray,
    ) -> xnp.ndarray:
        chem_energy_op = xnp.sum(occupations_before_ph, axis=(1, 2))  # sum over sites and colors
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
        ncolors = 2  # the faithful D2n representation is 2-dimensional
        nlayer = num_pg_layer + num_fermionic_layer
        param_shape = (nlayer, unitcell_size, len(symbolvec))
        gradients = xnp.zeros(param_shape)

        for layer_ind in range(num_pg_layer, nlayer):
            # only the fermionic layers directly contribute to the chemical potential

            for site_ind in range(lattice_size):
                site_factor = sublattice_factors[site_ind]  # even or odd sublattice

                for uc_ind in range(unitcell_size):
                    for symbol_ind, symbol in enumerate(symbolvec):
                        # we can skip parameters that are forced by the ansatz to be zero
                        if (layer_ind, uc_ind, symbol_ind) not in zeroed_params:

                            d_gamma_out = d_gamma_out_vec[layer_ind, uc_ind, symbol_ind]
                            grad = 0.0
                            for color in range(ncolors):
                                color_ind = 2 * ncolors * site_ind + 2 * color
                                grad += 0.5 * site_factor * d_gamma_out[color_ind + 1, color_ind]
                            gradients = backend.array_add(gradients, (layer_ind, uc_ind, symbol_ind), grad)

                    # further terms of the derivative are included higher up in the computation stack
                    # because computing them requires knowing various expectation values, which are not available here

        return gradients

    def _meson_string_vec(self, path: tuple[tuple[int, bool], ...]) -> xnp.ndarray:
        meson_op_vec = xnp.zeros(self.cfg.nlayer)
        return xnp.array(meson_op_vec)

    @staticmethod
    @maybe_jit(static_argnames=["after_ph"])
    def occupation(
        covmat: xnp.ndarray, site: int, site_coord: tuple[int, int], after_ph: bool = False, color: int = 0
    ) -> float:
        """Occupation of a single color at the given site (in [0, 1])."""
        ncolors = 2  # the faithful D2n representation is 2-dimensional
        site_ind = 2 * ncolors * site + 2 * color  # index into covariance matrix, 2 majorana modes per Dirac mode

        x, y = site_coord
        if after_ph:
            site_factor = 1
        else:
            site_factor = (-1) ** (x + y)  # even or odd sublattice

        return 0.5 * (1 + site_factor * covmat[site_ind + 1, site_ind])
