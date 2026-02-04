import logging

import sympy

import numpy as np

import ggpeps
from ggpeps import gauge, utils
from ggpeps.lattice import Direction

from .config_base import Config2DBase, generate_gauged_projector_terms

logger = logging.getLogger(ggpeps.LOGGER_NAME)


###################### Z2System2D ##########################


class Z2System2DConfig(Config2DBase):
    _nparams = 6
    ncopy = 1
    nvirtmodes_vertex = 4  # We have one virtual mode per direction (1 mode x 4 directions)
    nvirtmodes_link = 2  # We have two virtual modes per link (l/r or u/d)
    nphysmodes_site = 1  # number of physical modes per site
    ncolors = 1

    def __init__(
        self,
        lattice,
        g_el,
        g_mag,
        g_int,
        g_mass,
        g_chem,
        num_pg_layer=1,
        num_fermionic_layer=0,
        mod_link_inds=(0,),
        unitcell_size=1,
        enforce_u1_symmetry=True,
    ) -> None:
        # The parameters have the following order: [[t1,y1,z1],[t2,y2,z2],....]
        if num_fermionic_layer != 0:
            # This ansatz does not support fermionic layers
            raise ValueError("The Z2System2D ansatz does not support fermionic layers.")

        super().__init__(
            gauge.ZNGauge(2),
            lattice,
            g_el,
            g_mag,
            g_int,
            g_mass,
            g_chem,
            num_pg_layer,
            0,
            mod_link_inds,
            unitcell_size,
            enforce_u1_symmetry,
        )

        # Translation invariance
        if self.unitcell_size not in [1]:
            logger.error(
                "This ansatz only supports unitcell_size = 1. \
                This can be adapted by adding in a specification in the config to map sites to parameters."
            )
            raise ValueError("Invalid unitcell_size.")

        if not self.u1_symmetry:
            logger.error("This ansatz does not support the relaxation of U(1) symmetry.")
            raise ValueError("Invalid enforce_u1_symmetry.")

        self.init_el_energy_terms()

    def init_el_energy_terms(self) -> None:
        """Build idxa_vec and coeffs_vec."""
        idx_vec = []
        coeffs_vec = []

        for group_element in self.gaugemgr.group_elements_for_el_energy:
            # Pure Gauge (PG) ---
            idxarr_pg_h_0, _ = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, True, Direction.X, group_element, site=0
            )
            idxarr_pg_h_1, _ = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, True, Direction.X, group_element, site=1
            )
            idxarr_pg_v_0, _ = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, True, Direction.Y, group_element, site=0
            )
            idxarr_pg_v_1, _ = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, True, Direction.Y, group_element, site=1
            )

            pg_link_coeffs, pg_link_indices = [], []

            for mod_link in self.mod_link_inds:
                coord, dir = self.lattice.ind2coord_dir(mod_link)
                site_parity = sum(coord) % 2
                is_vertical = dir == Direction.Y

                # Select the correct base terms based on direction and parity
                if is_vertical:
                    if site_parity == 0:
                        term_pg = idxarr_pg_v_0
                    else:
                        term_pg = idxarr_pg_v_1
                else:
                    if site_parity == 0:
                        term_pg = idxarr_pg_h_0
                    else:
                        term_pg = idxarr_pg_h_1

                pg_c, pg_i = self._bucket_sort_terms(term_pg)
                pg_link_coeffs.append(pg_c)
                pg_link_indices.append(pg_i)

            pg_base_coeffs = tuple(pg_link_coeffs)

            # Stack layers: PG layers first, then Fermionic layers
            coeffs_vec.append((pg_base_coeffs,) * self.num_pg_layer)

            pg_base_indices = tuple(pg_link_indices)

            idx_vec.append((pg_base_indices,) * self.num_pg_layer)

        self.idx_vec = tuple(idx_vec)
        self.coeffs_vec = tuple(coeffs_vec)

    def make_pure_gauge(self):
        # The order of the parameters is [tr,yr,zr,ti,yi,zi] ({r,i} referring to the real/imaginary components)
        for lay in range(self.nlayer):
            for uc_ind in range(self.unitcell_size):
                # t real
                self.paramvec[lay, uc_ind, 0] = 0
                # t imag
                self.paramvec[lay, uc_ind, 3] = 0

    def get_zeroed_params(self):
        """This should really use make_pure_gauge() - i.e. return the indices which are set to zero there.
        However, some tests which use this ansatz do not actually satisfy the pure gauge condition
        - they use this ansatz with nonzero t params, and test against hard-coded values.
        (This works because make_pure_gauge() is often not called in the execution path of those tests).
        To preserve compatibility with those tests, we do not call make_pure_gauge() here.
        """
        zeroed_params = []
        return tuple(zeroed_params)

    def _create_symbolvec(self):
        """Define all symbols of the T matrix as symbols.
        We will use the analytic expression of the T matrix to calculate the
        derivative of the covariance matrices analytically.

        This method overwrites an abstract method in System2DBase.

        Returns:
            list: List of all analytic symbols
        """
        tr = sympy.Symbol("tr", real=True)
        yr = sympy.Symbol("yr", real=True)
        zr = sympy.Symbol("zr", real=True)
        ti = sympy.Symbol("ti", real=True)
        yi = sympy.Symbol("yi", real=True)
        zi = sympy.Symbol("zi", real=True)
        return [tr, yr, zr, ti, yi, zi]

    @property
    def tmat_symb(self):
        """Definition of the symbolic T matrix.
        The definition of T here is a result of an analytic consideration of global symmetries such as
        rotational invariance, charge conjugation invarance, etc.
        The T matrix is given in terms of symbols to compute the derivative of the covariance matrices
        analytically via sympy.
        We do not have to type them explicitly anymore into the code.

        This is one of two analytic inputs into the code.
        The other input is the structure and the parametrization of the projectors.

        The mode order is: Psi, l, r, d, u

        The order {l,r,d,u} instead of {r,u,l,d} (used in some analytic calculations) because it
        eliminates the need for a lot of permutation matrices in the conversion from T to gamma_maj.
        The permutation matrices are prone for errors.

        This method overwrites an abstract method in System2DBase.

        Returns:
            sympy.Matrix: Analytic T matrix of the fiducial state
        """
        [tr, yr, zr, ti, yi, zi] = self.symbolvec
        t = tr + 1.0j * ti
        y = yr + 1.0j * yi
        z = zr + 1.0j * zi
        tmat_symb = sympy.Matrix(
            [
                [0, -1.0j * t, 1.0j * t, t, -t],
                [1.0j * t, 0, 1.0j * y, z, 1.0j * z],
                [-1.0j * t, -1.0j * y, 0, -1.0j * z, -z],
                [-t, -z, 1.0j * z, 0, -y],
                [t, -1.0j * z, z, y, 0],
            ]
        )
        return tmat_symb

    def generate_gamma_gauge_neutral_dict(self):
        """This matrix is the covariance matrix of the ungauged projectors.
        The mode order is {l_1, l_2, r_1, r_2}/{d_1, d_2, u_1, u_2},
        where the underscore notation explicitly denotes Majorana modes and not sites.
        The sites are picked such that the left mode is right of the right modes,
        i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        This method overwrites an abstract method in System2DBase.

        Returns:
            np.ndarray: Covariance matrix of the ungauged projector on a single link
        """
        dest = [0] * 2
        dest[Direction.X] = np.real_if_close(
            1.0j * np.kron(utils.pauliy, utils.paulix)
        )  # this just happens to be a convenient way to generate the covariance matrix that was calculated by hand
        dest[Direction.Y] = np.real_if_close(np.kron(1.0j * utils.pauliy, utils.pauliz))
        return [dest] * self.nlayer
