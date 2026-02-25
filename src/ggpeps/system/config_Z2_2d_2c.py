import sympy
import logging

import numpy as np

import ggpeps
from ggpeps import utils, gauge
from ggpeps.lattice import Direction

from .config_base import Config2DBase, generate_gauged_projector_terms

logger = logging.getLogger(ggpeps.LOGGER_NAME)


###################### Z2System2D ##########################


class Z2System2D2CConfig(Config2DBase):
    """Configuration of the Z2 system in 2D with 2 copies of virtual fermions on the links.
    More details about the mode order and the parameters can be found in the documentation of `Z2System2D2C`.
    """

    _nparams = 20
    ncopy = 2
    nvirtmodes_vertex = 8  # We have two virtual modes per direction (4 directions x 2 modes)
    nvirtmodes_link = 4  # Number of virtual modes per link (2 copies and l/r or u/d)
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
        # The parameters have the following order: [[t1r,y1r,z1r,t2r,y2r,z2r,ar,br,cr,dr,t1i...],[..next layer..],....]
        if num_fermionic_layer != 0:
            # This ansatz does not support fermionic layers
            raise ValueError("The Z2System2D2C ansatz does not support fermionic layers.")
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
        """Build idx_vec, coeffs_vec and constants_vec.
        constants_vec is expected to contain only zeros for Z2"""
        idx_vec = []
        coeffs_vec = []
        constants_vec = []

        for group_element in self.gaugemgr.group_elements_for_el_energy:
            # Pure Gauge (PG) ---
            idxarr_pg_h_0, const_pg_h_0 = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, True, Direction.X, group_element, site=0
            )
            idxarr_pg_h_1, const_pg_h_1 = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, True, Direction.X, group_element, site=1
            )
            idxarr_pg_v_0, const_pg_v_0 = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, True, Direction.Y, group_element, site=0
            )
            idxarr_pg_v_1, const_pg_v_1 = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, True, Direction.Y, group_element, site=1
            )

            pg_link_coeffs, pg_link_indices, pg_link_constants = [], [], []

            for mod_link in self.mod_link_inds:
                coord, dir = self.lattice.ind2coord_dir(mod_link)
                site_parity = sum(coord) % 2
                is_vertical = dir == Direction.Y

                # Select the correct base terms based on direction and parity
                if is_vertical:
                    if site_parity == 0:
                        term_pg = idxarr_pg_v_0
                        contant_pg = const_pg_v_0
                    else:
                        term_pg = idxarr_pg_v_1
                        contant_pg = const_pg_v_1
                else:
                    if site_parity == 0:
                        term_pg = idxarr_pg_h_0
                        contant_pg = const_pg_h_0
                    else:
                        term_pg = idxarr_pg_h_1
                        contant_pg = const_pg_h_1

                pg_c, pg_i = self._bucket_sort_terms(term_pg)
                pg_link_coeffs.append(pg_c)
                pg_link_indices.append(pg_i)
                pg_link_constants.append(contant_pg)

            pg_base_constants = tuple(pg_link_constants)
            constants_vec.append((pg_base_constants,) * self.num_pg_layer)

            pg_base_coeffs = tuple(pg_link_coeffs)

            # Stack layers: PG layers first, then Fermionic layers
            coeffs_vec.append((pg_base_coeffs,) * self.num_pg_layer)

            pg_base_indices = tuple(pg_link_indices)

            idx_vec.append((pg_base_indices,) * self.num_pg_layer)

        self.idx_vec = tuple(idx_vec)
        self.coeffs_vec = tuple(coeffs_vec)
        self.constants_vec = tuple(constants_vec)

    def make_pure_gauge(self):
        """Ensure the system stays as pure_gauge.
        Setting the t parameters to zero automatically ensures they remain zero,
        since the derivative includes a factor of t."""

        # The order of the parameters is [t1r,y1r,z1r,t2r,y2r,z2r,ar,br,cr,dr,t1i,y1i,z1i,t2i,y2i,z2i,ai,bi,ci,di]
        # Here we set the t parameters to zero for the pure gauge layers (which is all the layers)
        assert self.nlayer == self.num_pg_layer
        for lay in range(self.nlayer):
            for uc_ind in range(self.unitcell_size):
                self.paramvec[lay, uc_ind, 0] = 0  # Set t1r to 0
                self.paramvec[lay, uc_ind, 10] = 0  # Set t1i to 0
                self.paramvec[lay, uc_ind, 3] = 0  # Set t2r to 0
                self.paramvec[lay, uc_ind, 13] = 0  # Set t2i to 0

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
        We will use the analytic expression of the T matrix to calculate
        the derivative of the covariance matrices analytically.

        Returns:
            list: List of all analytic symbols
        """
        t1r = sympy.Symbol("t1r", real=True)
        y1r = sympy.Symbol("y1r", real=True)
        z1r = sympy.Symbol("z1r", real=True)
        t2r = sympy.Symbol("t2r", real=True)
        y2r = sympy.Symbol("y2r", real=True)
        z2r = sympy.Symbol("z2r", real=True)
        ar = sympy.Symbol("ar", real=True)
        br = sympy.Symbol("br", real=True)
        cr = sympy.Symbol("cr", real=True)
        dr = sympy.Symbol("dr", real=True)

        t1i = sympy.Symbol("t1i", real=True)
        y1i = sympy.Symbol("y1i", real=True)
        z1i = sympy.Symbol("z1i", real=True)
        t2i = sympy.Symbol("t2i", real=True)
        y2i = sympy.Symbol("y2i", real=True)
        z2i = sympy.Symbol("z2i", real=True)
        ai = sympy.Symbol("ai", real=True)
        bi = sympy.Symbol("bi", real=True)
        ci = sympy.Symbol("ci", real=True)
        di = sympy.Symbol("di", real=True)
        return [
            t1r,
            y1r,
            z1r,
            t2r,
            y2r,
            z2r,
            ar,
            br,
            cr,
            dr,
            t1i,
            y1i,
            z1i,
            t2i,
            y2i,
            z2i,
            ai,
            bi,
            ci,
            di,
        ]

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

        The mode order is: Psi, l_1, r_1, d_1, u_1, l_2, r_2, d_2, u_2

        The order {l,r,d,u} instead of {r,u,l,d} (used in some analytic calculations) because it eliminates
        the need for a lot of permutation matrices in the conversion from T to gamma_maj.
        The permutation matrices are prone for errors.

        Returns:
            sympy.Matrix: Analytic T matrix of the fiducial state
        """
        [
            t1r,
            y1r,
            z1r,
            t2r,
            y2r,
            z2r,
            ar,
            br,
            cr,
            dr,
            t1i,
            y1i,
            z1i,
            t2i,
            y2i,
            z2i,
            ai,
            bi,
            ci,
            di,
        ] = self.symbolvec
        t1 = t1r + 1.0j * t1i
        y1 = y1r + 1.0j * y1i
        z1 = z1r + 1.0j * z1i
        t2 = t2r + 1.0j * t2i
        y2 = y2r + 1.0j * y2i
        z2 = z2r + 1.0j * z2i
        a = ar + 1.0j * ai
        b = br + 1.0j * bi
        c = cr + 1.0j * ci
        d = dr + 1.0j * di
        tmat_symb = sympy.Matrix(
            [
                [0, -1.0j * t1, 1.0j * t1, t1, -t1, -1.0j * t2, 1.0j * t2, t2, -t2],
                [
                    1.0j * t1,
                    0,
                    1.0j * y1,
                    z1,
                    1.0j * z1,
                    -1.0j * a,
                    -1.0j * c,
                    -1.0j * b,
                    -1.0j * d,
                ],
                [
                    -1.0j * t1,
                    -1.0j * y1,
                    0,
                    -1.0j * z1,
                    -z1,
                    1.0j * c,
                    1.0j * a,
                    1.0j * d,
                    1.0j * b,
                ],
                [-t1, -z1, 1.0j * z1, 0, -y1, d, b, a, c],
                [t1, -1.0j * z1, z1, y1, 0, -b, -d, -c, -a],
                [1.0j * t2, 1.0j * a, -1.0j * c, -d, b, 0, 1.0j * y2, z2, 1.0j * z2],
                [
                    -1.0j * t2,
                    1.0j * c,
                    -1.0j * a,
                    -b,
                    d,
                    -1.0j * y2,
                    0,
                    -1.0j * z2,
                    -z2,
                ],
                [-t2, 1.0j * b, -1.0j * d, -a, c, -z2, 1.0j * z2, 0, -y2],
                [t2, 1.0j * d, -1.0j * b, -c, a, -1.0j * z2, z2, y2, 0],
            ]
        )
        return tmat_symb

    def generate_gamma_gauge_neutral_dict(self):
        """Generate the the covariance matrix of the ungauged projectors.
        The morde order is:
            {l1_1, l1_2, r1_1, r1_2, l2_1, l2_2, r2_1, r2_2}/{d1_1, d1_2, u1_1, u1_2,d2_1, d2_2, u2_1, u2_2}.
        The naming convention here is <mode letter><number of copy>_<majorana mode>.
        We order first by link and then by copy.
        Modes of copy one are coupled to modes of copy 2. The projectors mix copies.
        The sites are picked such that the left mode is right of the right modes,
        i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        This method overwrites an abstract method in System2DBase.

        Returns:
            list[np.ndarray]: Covariance matrix of the ungauged projector on a single link
        """
        dest = [0] * 2
        dest[Direction.X] = np.real_if_close(1.0j * np.kron(utils.paulix, np.kron(utils.pauliy, utils.paulix)))
        dest[Direction.Y] = np.real_if_close(1.0j * np.kron(utils.paulix, np.kron(utils.pauliy, utils.pauliz)))
        return [dest] * self.nlayer
