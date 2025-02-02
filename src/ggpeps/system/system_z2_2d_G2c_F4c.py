import sympy
import logging
import numpy as np
from scipy.linalg import block_diag

import ggpeps
from ggpeps import utils
from ggpeps.lattice import Direction

from .system_base import Config2DBase, System2DBase
from .system_base import get_pfaffian_arrays

logger = logging.getLogger(ggpeps.LOGGER_NAME)


###################### Z2System2D ##########################


class Z2System2D_G2C_F4C_Config(Config2DBase):
    """Configuration of the Z2 system in 2D with 4 copies of virtual fermions on the links.
    More details about the mode order and the parameters can be found in the documentation of `Z2System2D2C`.

    Some general notes about conventions:

    Order of the paramvec: see the functions that create the symbolvec.
    Mode order of tmat: {p,l1,r1,d1,u1,l2,r2,d2,u2,l3,r3,d3,u3,l4,r4,d4,u4}.
    Mode order of gamma_dirac: {p,l1,r1,d1,u1,l2,r2,d2,u2,p_dag,l1_dag,r1_dag,u1_dag,d1_dag,l2_dat,r2_dag,u2_dag,d2_dag,l3,r3... and so on}.
    Mode order of gamma_maj: {p_1,p_2,l1_1,l1_2,r1_1,r1_2,d1_1,d1_2,u1_1,u1_2,l2_1,l2_2,r2_1,r2_2,d2_1,d2_2,u2_1,u2_2,l3_1,l3_2... and so on}.
    """

    _nparams_per_layer = 52  # 36
    ncopy = 4
    nvirtmodes_vertex = 16
    nvirtmodes_link = 8

    def __init__(
        self,
        lattice,
        g_el,
        g_mag,
        g_int,
        g_mass,
        g_chem,
        num_pg_layer=1,
        num_fermionic_layer=1,
        unitcell_size=1,
    ):
        super().__init__(
            lattice,
            g_el,
            g_mag,
            g_int,
            g_mass,
            g_chem,
            num_pg_layer,
            num_fermionic_layer,
        )

        # Translation invariance
        if unitcell_size not in [1]:
            logger.error(
                "This ansatz only supports unitcell_size = 1 or 2. \
                This can be adapted by adding in a specification in the config to map sites to parameters."
            )
            raise ValueError("Invalid unitcell_size.")
        self.site_params_dict = {
            site: 0 for site in range(self.lattice.size)
        }  # map from site to index of independent parameters
        self.unitcell_size = 1

        # Constants used in the calculation of the electric energy
        prefactors = [
            [1, -1, 1.0j, 1.0j],
            [1, -1, 1.0j, 1.0j],
            [1, -1, 1.0j, 1.0j],
            [1, -1, 1.0j, 1.0j],
        ]
        indices_layer_pg = [
            [(2, 4), (3, 5), (4, 5), (2, 3)],
            [(6, 0), (7, 1), (0, 1), (6, 7)],
            [(10, 12), (11, 13), (12, 13), (10, 11)],
            [(14, 8), (15, 9), (8, 9), (14, 15)],
        ]
        indices_layer_fermionic = [
            [(2, 0), (3, 1), (0, 1), (2, 3)],
            [(6, 4), (7, 5), (4, 5), (6, 7)],
            [(10, 8), (11, 9), (8, 9), (10, 11)],
            [(14, 12), (15, 13), (12, 13), (14, 15)],
        ]

        idxarr_lay_pg = get_pfaffian_arrays(indices_layer_pg, prefactors)
        idxarr_lay_fermionic = get_pfaffian_arrays(indices_layer_fermionic, prefactors)
        self.idxarr_vec = [idxarr_lay_pg] * self.num_pg_layer + [
            idxarr_lay_fermionic
        ] * self.num_fermionic_layer

        self.el_overall_factors = [1 / 256] * (
            self.nlayer
        )  # this arises due to normalization and the i^(# of modes/2) in the expression Tr[i^# * rho * (modes)]

    def make_pure_gauge(self):
        raise NotImplementedError(
            "Haven't implemented parameter conditions for pure gauge for this ansatz."
        )

    def enforce_parameter_conditions(self, mat):
        """Enforce conditions on parameters on each layer to get the required behaviour for the ansatz."""
        # The order of the parameters is [t1r, y1r, z1r, t2r, y2r, z2r, ar, br, cr, dr, t1i, y1i,
        #    z1i, t2i, y2i, z2i, ai, bi, ci, di,
        #    z3r, z4r, y3r, y4r, a2r, b2r, c2r, d2r,
        #    z3i, z4i, y3i, y4i, a2i, b2i, c2i, d2i,
        #    p14r, q14r, r14r, s14r, p14i, q14i, r14i, s14i,
        #    p23r, q23r, r23r, s23r, p23i, q23i, r23i, s23i]

        zeroed_params = []  # we'll save the indices of the zeroed parameters

        t_indices = [0, 3, 10, 13]  # index of t1r, t2r, t1i, t2i in symbolvec
        for layer_ind in range(self.num_pg_layer):
            for uc_ind in range(self.unitcell_size):
                for t_ind in t_indices:
                    coord = (layer_ind, uc_ind, t_ind)
                    mat[coord] = 0
                    zeroed_params.append(coord)

        zero_for_fermionic_layer = [
            1,
            2,
            4,
            5,
            11,
            12,
            14,
            15,
            20,
            21,
            22,
            23,
            28,
            29,
            30,
            31,
        ]  # indices of y's, z's in symbolvec
        for layer_ind in range(self.num_pg_layer, self.nlayer):
            for uc_ind in range(self.unitcell_size):
                for ind in zero_for_fermionic_layer:
                    coord = (layer_ind, uc_ind, ind)
                    mat[coord] = 0
                    zeroed_params.append(coord)

        # It is also possible to test the 2 copy ansatz within this one, by zeroing all the extra parameters
        # (a2, b2, c2, d2, and all the p,q,r,s params)

        # save zeroed params
        self.zeroed_params = zeroed_params
        return

    def _create_symbolvec(self):
        """Define all symbols of the T matrix as symbols.
        We will use the analytic expression of the T matrix to calculate the derivative of the covariance matrices analytically.

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

        # symbols for third and fourth copy for fermionic layer
        z3r = sympy.Symbol("z3r", real=True)
        z4r = sympy.Symbol("z4r", real=True)
        y3r = sympy.Symbol("y3r", real=True)
        y4r = sympy.Symbol("y4r", real=True)
        a2r = sympy.Symbol("a2r", real=True)
        b2r = sympy.Symbol("b2r", real=True)
        c2r = sympy.Symbol("c2r", real=True)
        d2r = sympy.Symbol("d2r", real=True)
        z3i = sympy.Symbol("z3i", real=True)
        z4i = sympy.Symbol("z4i", real=True)
        y3i = sympy.Symbol("y3i", real=True)
        y4i = sympy.Symbol("y4i", real=True)
        a2i = sympy.Symbol("a2i", real=True)
        b2i = sympy.Symbol("b2i", real=True)
        c2i = sympy.Symbol("c2i", real=True)
        d2i = sympy.Symbol("d2i", real=True)

        # symbols to couple the third and fourth copy with the first and second, for the fermions
        p14r = sympy.Symbol("p14r", real=True)  # couple copy 1 with 4, real part
        q14r = sympy.Symbol("q14r", real=True)
        r14r = sympy.Symbol("r14r", real=True)
        s14r = sympy.Symbol("s14r", real=True)
        p14i = sympy.Symbol("p14i", real=True)
        q14i = sympy.Symbol("q14i", real=True)
        r14i = sympy.Symbol("r14i", real=True)
        s14i = sympy.Symbol("s14i", real=True)
        p23r = sympy.Symbol("p23r", real=True)  # couple copy 2 with 3, real part
        q23r = sympy.Symbol("q23r", real=True)
        r23r = sympy.Symbol("r23r", real=True)
        s23r = sympy.Symbol("s23r", real=True)
        p23i = sympy.Symbol("p23i", real=True)
        q23i = sympy.Symbol("q23i", real=True)
        r23i = sympy.Symbol("r23i", real=True)
        s23i = sympy.Symbol("s23i", real=True)

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
            z3r,
            z4r,
            y3r,
            y4r,
            a2r,
            b2r,
            c2r,
            d2r,
            z3i,
            z4i,
            y3i,
            y4i,
            a2i,
            b2i,
            c2i,
            d2i,
            p14r,
            q14r,
            r14r,
            s14r,
            p14i,
            q14i,
            r14i,
            s14i,
            p23r,
            q23r,
            r23r,
            s23r,
            p23i,
            q23i,
            r23i,
            s23i,
        ]

    @property
    def tmat_symb(self):
        """Definition of the symbolic T matrix.
        The definition of T here is a result of an analytic consideration of global symmetries like rotational invariance, charge conjugation invarance, etc.
        The T matrix is given in terms of symbols to compute the derivative of the covariance matrices analytically via sympy.
        We do not have to type them explicitly anymore into the code.

        This is one of two analytic inputs into the code.
        The other input is the structure and the parametrization of the projectors.

        The mode order is: Psi, l_1, r_1, d_1, u_1, l_2, r_2, d_2, u_2, l_3, r_3...

        The order {l,r,d,u} instead of {r,u,l,d} (used in some analytic calculations) because it eliminates the need for a lot of permutation matrices in the conversion from T to gamma_maj.
        The permutation matrices are prone to errors.

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
            z3r,
            z4r,
            y3r,
            y4r,
            a2r,
            b2r,
            c2r,
            d2r,
            z3i,
            z4i,
            y3i,
            y4i,
            a2i,
            b2i,
            c2i,
            d2i,
            p14r,
            q14r,
            r14r,
            s14r,
            p14i,
            q14i,
            r14i,
            s14i,
            p23r,
            q23r,
            r23r,
            s23r,
            p23i,
            q23i,
            r23i,
            s23i,
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

        z3 = z3r + 1.0j * z3i
        z4 = z4r + 1.0j * z4i
        y3 = y3r + 1.0j * y3i
        y4 = y4r + 1.0j * y4i
        a2 = a2r + 1.0j * a2i
        b2 = b2r + 1.0j * b2i
        c2 = c2r + 1.0j * c2i
        d2 = d2r + 1.0j * d2i

        p14 = p14r + 1.0j * p14i
        q14 = q14r + 1.0j * q14i
        r14 = r14r + 1.0j * r14i
        s14 = s14r + 1.0j * s14i
        p23 = p23r + 1.0j * p23i
        q23 = q23r + 1.0j * q23i
        r23 = r23r + 1.0j * r23i
        s23 = s23r + 1.0j * s23i

        zeros_8 = sympy.zeros(8)
        Block_1 = sympy.Matrix(
            [-1.0j * t1, 1.0j * t1, t1, -t1, 0, 0, 0, 0]
        )  # this is a column matrix
        Block_2a = sympy.Matrix(
            [
                [0, 1.0j * y1, z1, 1.0j * z1],
                [-1.0j * y1, 0, -1.0j * z1, -z1],
                [-z1, 1.0j * z1, 0, -y1],
                [-1.0j * z1, z1, y1, 0],
            ]
        )
        Block_2b = sympy.Matrix(
            [
                [-1.0j * a, -1.0j * c, -1.0j * b, -1.0j * d],
                [1.0j * c, 1.0j * a, 1.0j * d, 1.0j * b],
                [d, b, a, c],
                [-b, -d, -c, -a],
            ]
        )
        Block_2 = sympy.Matrix(
            [
                [
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
                    -1.0j * y1,
                    0,
                    -1.0j * z1,
                    -z1,
                    1.0j * c,
                    1.0j * a,
                    1.0j * d,
                    1.0j * b,
                ],
                [-z1, 1.0j * z1, 0, -y1, d, b, a, c],
                [-1.0j * z1, z1, y1, 0, -b, -d, -c, -a],
                [1.0j * a, -1.0j * c, -d, b, 0, 1.0j * y2, z2, 1.0j * z2],
                [1.0j * c, -1.0j * a, -b, d, -1.0j * y2, 0, -1.0j * z2, -z2],
                [1.0j * b, -1.0j * d, -a, c, -z2, 1.0j * z2, 0, -y2],
                [1.0j * d, -1.0j * b, -c, a, -1.0j * z2, z2, y2, 0],
            ]
        )
        Block_2 = sympy.Matrix(
            sympy.BlockMatrix(
                [
                    [Block_2a, Block_2b],
                    [-Block_2b.T, -Block_2a.subs([(z1, z2), (y1, y2)]).T],
                ]
            )
        )

        substitutionsB = [
            (z1, z3),
            (z2, z4),
            (y1, y3),
            (y2, y4),
            (a, a2),
            (b, b2),
            (c, c2),
            (d, d2),
        ]
        Block_2B = Block_2.subs(substitutionsB)

        # To be used for coupling between 1-2 and 3-4 layers
        zeros_4 = sympy.zeros(4)
        Block_2C = sympy.Matrix(
            sympy.BlockMatrix(
                [
                    [zeros_4, Block_2b.subs([(a, p14), (b, q14), (c, r14), (d, s14)])],
                    [Block_2b.subs([(a, p23), (b, q23), (c, r23), (d, s23)]), zeros_4],
                ]
            )
        )

        tmat_symb = sympy.Matrix(
            sympy.BlockMatrix(
                [
                    [sympy.zeros(1), -Block_1.T, -Block_1.subs(t1, t2).T],
                    [Block_1, Block_2, Block_2C],
                    [Block_1.subs(t1, t2), -Block_2C.T, Block_2B],
                ]
            )
        )

        return tmat_symb

    def generate_gamma_gauge_neutral_dict(self):
        """Generate the covariance matrix of the ungauged projectors.
        The mode order is {l1_1, l1_2, r1_1, r1_2, l2_1, l2_2, r2_1, r2_2, l3_1...}/{d1_1, d1_2, u1_1, u1_2, d2_1, d2_2, u2_1, u2_2, d3_1...}.
        The naming convention here is <mode letter><number of copy>_<majorana mode>.
        We order first by link and then by copy.
        The sites are picked such that the left mode is right of the right modes, i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        This function returns two different covariance matrices for ungauged projectors:
        In the first, modes of copy 1 are coupled to modes of copy 2.
        In the second, the projectors don't mix copies.
        The first option is used for the pure-gauge layer, the second for the fermionic layer.

        This method overwrites an abstract method in System2DBase.

        Returns:
            List[np.ndarray]: Covariance matrices of the ungauged projector on a single link
        """

        dest_mixed = [0] * 2  # mixes copies
        dest_unmixed = [0] * 2  # does not mix copies

        zeros_8 = np.zeros((8, 8))

        # We want to give the projectors for the pure gauge part, which mix copies
        mixed_X = np.real_if_close(
            1.0j * np.kron(utils.paulix, np.kron(utils.pauliy, utils.paulix))
        )
        mixed_Y = np.real_if_close(
            1.0j * np.kron(utils.paulix, np.kron(utils.pauliy, utils.pauliz))
        )

        dest_mixed[Direction.X] = np.block([[mixed_X, zeros_8], [zeros_8, mixed_X]])
        dest_mixed[Direction.Y] = np.block([[mixed_Y, zeros_8], [zeros_8, mixed_Y]])

        # We want to give the projectors for the fermionic part which don't mix copies (so as to preserve global U(1) symmetry)
        unmixed_X = np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            ]
        )

        unmixed_Y = np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 0.0, -0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )

        dest_unmixed[Direction.X] = np.block(
            [[unmixed_X, zeros_8], [zeros_8, unmixed_X]]
        )
        dest_unmixed[Direction.Y] = np.block(
            [[unmixed_Y, zeros_8], [zeros_8, unmixed_Y]]
        )

        return [dest_mixed] * self.num_pg_layer + [
            dest_unmixed
        ] * self.num_fermionic_layer
