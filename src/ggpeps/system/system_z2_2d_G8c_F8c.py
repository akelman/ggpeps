import logging

import sympy

# import numpy as np
import ggpeps
from ggpeps import xnp as np

from ggpeps import utils, gauge
from ggpeps.lattice import Direction

from .system_base import Config2DBase
from .system_base import get_pfaffian_arrays

logger = logging.getLogger(ggpeps.LOGGER_NAME)


###################### Z2System2D ##########################


class Z2System2D_G8C_F8C_Config(Config2DBase):
    """Configuration of the Z2 system in 2D with 8 copies of virtual fermions on the links.
    More details about the mode order and the parameters can be found in the documentation of `Z2System2D2C`.
    """

    _nparams = 152
    ncopy = 8
    nvirtmodes_vertex = 32
    nvirtmodes_link = 16
    nphysmodes_site = 1  # number of physical modes per site

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
        enforce_u1_symmetry=True,
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
                "This ansatz only supports unitcell_size = 1. \
                This can be adapted by adding in a specification in the config to map sites to parameters."
            )
            raise ValueError("Invalid unitcell_size.")
        self.site_params_dict = {
            site: 0 for site in range(self.lattice.size)
        }  # map from site to index of independent parameters
        self.unitcell_size = 1

        if not enforce_u1_symmetry:
            logger.error(
                "This ansatz does not support the relaxation of U(1) symmetry."
            )
            raise ValueError("Invalid enforce_u1_symmetry.")

        # We store a list of the parameters forced to be zero by the ansatz
        # They are actually used in self.enforce_parameter_conditions(), as well as in other checks throughout
        self.zeroed_params: list[tuple[int, int, int]] = self.get_zeroed_params()

        # Constants used in the calculation of the electric energy
        prefactors = [[1, -1, 1.0j, 1.0j]] * 8
        indices_layer1 = [
            [(2, 4), (3, 5), (4, 5), (2, 3)],
            [(6, 0), (7, 1), (0, 1), (6, 7)],
            [(10, 12), (11, 13), (12, 13), (10, 11)],
            [(14, 8), (15, 9), (8, 9), (14, 15)],
            [(18, 20), (19, 21), (20, 21), (18, 19)],
            [(22, 16), (23, 17), (16, 17), (22, 23)],
            [(26, 28), (27, 29), (28, 29), (26, 27)],
            [(30, 24), (31, 25), (24, 25), (30, 31)],
        ]
        indices_layer2 = [
            [(2, 0), (3, 1), (0, 1), (2, 3)],
            [(6, 4), (7, 5), (4, 5), (6, 7)],
            [(10, 8), (11, 9), (8, 9), (10, 11)],
            [(14, 12), (15, 13), (12, 13), (14, 15)],
            [(18, 16), (19, 17), (16, 17), (18, 19)],
            [(22, 20), (23, 21), (20, 21), (22, 23)],
            [(26, 24), (27, 25), (24, 25), (26, 27)],
            [(30, 28), (31, 29), (28, 29), (30, 31)],
        ]
        idxarr_lay1 = get_pfaffian_arrays(
            indices_layer1, prefactors
        )  # pure gauge layers
        idxarr_lay2 = get_pfaffian_arrays(
            indices_layer2, prefactors
        )  # fermionic layers
        self.idxarr_vec = [idxarr_lay1] * (self.num_pg_layer) + [idxarr_lay2]
        self.el_overall_factors = [1 / 256**2] * (
            self.nlayer
        )  # this arises due to normalization and the i^(# of modes/2) in the expression Tr[i^# * rho * (modes)]
        self.gaugemgr: gauge.ZNGauge = gauge.ZNGauge(2)

    def get_zeroed_params(self):
        zeroed_params = []  # we'll save the indices of the zeroed parameters

        # pure gauge layers
        for layer in range(self.num_pg_layer):
            for uc_ind in range(self.unitcell_size):
                ind = 0
                copies = [1, 3, 5, 7]  # copies which couple to physical modes
                for cop in copies:
                    for com in ["r", "i"]:  # real or imaginary
                        ind += 1
                        zeroed_params.append((layer, uc_ind, ind))

        # fermionic layers
        for layer_ind in range(self.num_pg_layer, self.nlayer):
            for uc_ind in range(self.unitcell_size):
                ind = 0
                copies = [1, 3, 5, 7]  # copies which couple to physical modes
                for cop in copies:
                    for com in ["r", "i"]:
                        ind += 1  # don't zero out t params

                copies = [1, 2, 3, 4]  # copies which couple to themselves
                for cop in copies:
                    for l in ["z", "y"]:
                        for com in ["r", "i"]:
                            ind += 1
                            zeroed_params.append((layer, uc_ind, ind))

        return zeroed_params

    def _create_symbolvec(self):
        """Define all symbols of the T matrix as symbols.
        We will use the analytic expression of the T matrix to calculate the derivative of the covariance matrices analytically.

        Returns:
            list: List of all analytic symbols
        """

        phy_virt_symbols = []  # for coupling physical and virtual modes
        copies = [1, 3, 5, 7]  # copies which couple to physical modes
        for cop in copies:
            for com in ["r", "i"]:  # real or imaginary
                symbol = sympy.Symbol(f"t{cop}{com}", real=True)
                phy_virt_symbols.append(symbol)

        on_diag_symbols = []
        copies = [
            1,
            2,
            3,
            4,
        ]  # copies which couple to themselves (if not zeroed out in enforce_parameter_conditions)
        for cop in copies:
            for l in ["z", "y"]:
                for com in ["r", "i"]:
                    symbol = sympy.Symbol(f"{l}{cop}{com}", real=True)
                    on_diag_symbols.append(symbol)

        off_diag_symbols = []  # off-diagonal blocks
        copies_odd = [1, 3, 5, 7]
        copies_even = [2, 4, 6, 8]
        for r in copies_odd:
            for c in copies_even:
                for l in ["p", "q", "r", "s"]:
                    for com in ["r", "i"]:
                        symbol = sympy.Symbol(f"{l}{r}{c}{com}", real=True)
                        off_diag_symbols.append(symbol)

        all_symbols = phy_virt_symbols + on_diag_symbols + off_diag_symbols
        return all_symbols

    @property
    def tmat_symb(self):
        """Definition of the symbolic T matrix.
        The definition of T here is a result of an analytic consideration of global
        symmetries like rotational invariance, charge conjugation invarance, etc.
        The T matrix is given in terms of symbols to compute the derivative of the
        covariance matrices analytically via sympy.
        We do not have to type them explicitly anymore into the code.

        This is one of two analytic inputs into the code.
        The other input is the structure and the parametrization of the projectors.

        The mode order is: Psi, l_1, r_1, d_1, u_1, l_2, r_2, d_2, u_2, l_3, r_3...

        The order {l,r,d,u} instead of {r,u,l,d} (used in some analytic calculations)
        because it eliminates the need for a lot of permutation matrices in the conversion from T to gamma_maj.
        The permutation matrices are prone to errors.

        Returns:
            sympy.Matrix: Analytic T matrix of the fiducial state
        """

        # Build dictionary of parameters
        all_params = {}
        ind = 0

        copies = [1, 3, 5, 7]  # copies which couple to physical modes
        for cop in copies:
            all_params[f"t{cop}"] = self.symbolvec[ind] + 1.0j * self.symbolvec[ind + 1]
            ind += 2

        copies = [
            1,
            2,
            3,
            4,
        ]  # copies which couple to themselves (if not zeroed out in enforce_parameter_conditions)
        for cop in copies:
            for l in ["z", "y"]:
                all_params[f"{l}{cop}"] = (
                    self.symbolvec[ind] + 1.0j * self.symbolvec[ind + 1]
                )
                ind += 2

        copies_odd = [1, 3, 5, 7]
        copies_even = [2, 4, 6, 8]
        for r in copies_odd:
            for c in copies_even:
                for l in ["p", "q", "r", "s"]:
                    all_params[f"{l}{r}{c}"] = (
                        self.symbolvec[ind] + 1.0j * self.symbolvec[ind + 1]
                    )
                    ind += 2

        # Extract params as variables for convenience
        z1 = all_params["z1"]
        z2 = all_params["z2"]
        y1 = all_params["y1"]
        y2 = all_params["y2"]
        z3 = all_params["z3"]
        z4 = all_params["z4"]
        y3 = all_params["y3"]
        y4 = all_params["y4"]

        p12, q12, r12, s12 = (
            all_params["p12"],
            all_params["q12"],
            all_params["r12"],
            all_params["s12"],
        )
        p14, q14, r14, s14 = (
            all_params["p14"],
            all_params["q14"],
            all_params["r14"],
            all_params["s14"],
        )
        p16, q16, r16, s16 = (
            all_params["p16"],
            all_params["q16"],
            all_params["r16"],
            all_params["s16"],
        )
        p18, q18, r18, s18 = (
            all_params["p18"],
            all_params["q18"],
            all_params["r18"],
            all_params["s18"],
        )

        p32, q32, r32, s32 = (
            all_params["p32"],
            all_params["q32"],
            all_params["r32"],
            all_params["s32"],
        )
        p34, q34, r34, s34 = (
            all_params["p34"],
            all_params["q34"],
            all_params["r34"],
            all_params["s34"],
        )
        p36, q36, r36, s36 = (
            all_params["p36"],
            all_params["q36"],
            all_params["r36"],
            all_params["s36"],
        )
        p38, q38, r38, s38 = (
            all_params["p38"],
            all_params["q38"],
            all_params["r38"],
            all_params["s38"],
        )

        p52, q52, r52, s52 = (
            all_params["p52"],
            all_params["q52"],
            all_params["r52"],
            all_params["s52"],
        )
        p54, q54, r54, s54 = (
            all_params["p54"],
            all_params["q54"],
            all_params["r54"],
            all_params["s54"],
        )
        p56, q56, r56, s56 = (
            all_params["p56"],
            all_params["q56"],
            all_params["r56"],
            all_params["s56"],
        )
        p58, q58, r58, s58 = (
            all_params["p58"],
            all_params["q58"],
            all_params["r58"],
            all_params["s58"],
        )

        p72, q72, r72, s72 = (
            all_params["p72"],
            all_params["q72"],
            all_params["r72"],
            all_params["s72"],
        )
        p74, q74, r74, s74 = (
            all_params["p74"],
            all_params["q74"],
            all_params["r74"],
            all_params["s74"],
        )
        p76, q76, r76, s76 = (
            all_params["p76"],
            all_params["q76"],
            all_params["r76"],
            all_params["s76"],
        )
        p78, q78, r78, s78 = (
            all_params["p78"],
            all_params["q78"],
            all_params["r78"],
            all_params["s78"],
        )

        # Block matrices that appear many times in the T matrix
        Block_1 = sympy.Matrix(
            [
                -1.0j * all_params["t1"],
                1.0j * all_params["t1"],
                all_params["t1"],
                -all_params["t1"],
                0,
                0,
                0,
                0,
            ]
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
                [-1.0j * p12, -1.0j * r12, -1.0j * q12, -1.0j * s12],
                [1.0j * r12, 1.0j * p12, 1.0j * s12, 1.0j * q12],
                [s12, q12, p12, r12],
                [-q12, -s12, -r12, -p12],
            ]
        )
        zeros_4 = sympy.zeros(4)

        # first row
        M00 = sympy.zeros(1)
        M01 = -Block_1.T  # copies 1,2
        M02 = -Block_1.subs([(all_params["t1"], all_params["t3"])]).T  # copies 3,4
        M03 = -Block_1.subs([(all_params["t1"], all_params["t5"])]).T
        M04 = -Block_1.subs([(all_params["t1"], all_params["t7"])]).T

        # second row
        M10 = -M01.T
        M11 = sympy.Matrix(
            sympy.BlockMatrix(
                [
                    [Block_2a, Block_2b],
                    [-Block_2b.T, Block_2a.subs([(z1, z2), (y1, y2)])],
                ]
            )
        )
        M12 = sympy.Matrix(
            sympy.BlockMatrix(
                [
                    [
                        zeros_4,
                        Block_2b.subs([(p12, p14), (q12, q14), (r12, r14), (s12, s14)]),
                    ],
                    [
                        Block_2b.subs([(p12, p32), (q12, q32), (r12, r32), (s12, s32)]),
                        zeros_4,
                    ],
                ]
            )
        )
        M13 = sympy.Matrix(
            sympy.BlockMatrix(
                [
                    [
                        zeros_4,
                        Block_2b.subs([(p12, p16), (q12, q16), (r12, r16), (s12, s16)]),
                    ],
                    [
                        Block_2b.subs([(p12, p52), (q12, q52), (r12, r52), (s12, s52)]),
                        zeros_4,
                    ],
                ]
            )
        )
        M14 = sympy.Matrix(
            sympy.BlockMatrix(
                [
                    [
                        zeros_4,
                        Block_2b.subs([(p12, p18), (q12, q18), (r12, r18), (s12, s18)]),
                    ],
                    [
                        Block_2b.subs([(p12, p72), (q12, q72), (r12, r72), (s12, s72)]),
                        zeros_4,
                    ],
                ]
            )
        )

        # third row
        M20 = -M02.T
        M21 = -M12.T
        Block_2b_22 = Block_2b.subs([(p12, p34), (q12, q34), (r12, r34), (s12, s34)])
        M22 = sympy.Matrix(
            sympy.BlockMatrix(
                [
                    [Block_2a.subs([(z1, z3), (y1, y3)]), Block_2b_22],
                    [-Block_2b_22.T, Block_2a.subs([(z1, z4), (y1, y4)])],
                ]
            )
        )  # z3,y3 should go here
        M23 = sympy.Matrix(
            sympy.BlockMatrix(
                [
                    [
                        zeros_4,
                        Block_2b.subs([(p12, p36), (q12, q36), (r12, r36), (s12, s36)]),
                    ],
                    [
                        Block_2b.subs([(p12, p54), (q12, q54), (r12, r54), (s14, s54)]),
                        zeros_4,
                    ],
                ]
            )
        )
        M24 = sympy.Matrix(
            sympy.BlockMatrix(
                [
                    [
                        zeros_4,
                        Block_2b.subs([(p12, p38), (q12, q38), (r12, r38), (s12, s38)]),
                    ],
                    [
                        Block_2b.subs([(p12, p74), (q12, q74), (r12, r74), (s14, s74)]),
                        zeros_4,
                    ],
                ]
            )
        )

        # fourth row
        M30 = -M03.T
        M31 = -M13.T
        M32 = -M23.T
        Block_2b_33 = Block_2b.subs([(p12, p56), (q12, q56), (r12, r56), (s12, s56)])
        M33 = sympy.Matrix(
            sympy.BlockMatrix([[zeros_4, Block_2b_33], [-Block_2b_33.T, zeros_4]])
        )
        M34 = sympy.Matrix(
            sympy.BlockMatrix(
                [
                    [
                        zeros_4,
                        Block_2b.subs([(p12, p58), (q12, q58), (r12, r58), (s12, s58)]),
                    ],
                    [
                        Block_2b.subs([(p12, p76), (q12, q76), (r12, r76), (s14, s76)]),
                        zeros_4,
                    ],
                ]
            )
        )

        # fifth row
        M40 = -M04.T
        M41 = -M14.T
        M42 = -M24.T
        M43 = -M34.T
        Block_2b_44 = Block_2b.subs([(p12, p78), (q12, q78), (r12, r78), (s12, s78)])
        M44 = sympy.Matrix(
            sympy.BlockMatrix([[zeros_4, Block_2b_44], [-Block_2b_44.T, zeros_4]])
        )

        # Full T matrix
        tmat_symb = sympy.Matrix(
            sympy.BlockMatrix(
                [
                    [M00, M01, M02, M03, M04],
                    [M10, M11, M12, M13, M14],
                    [M20, M21, M22, M23, M24],
                    [M30, M31, M32, M33, M34],
                    [M40, M41, M42, M43, M44],
                ]
            )
        )

        return tmat_symb

    def generate_gamma_gauge_neutral_dict(self):
        """Generate the covariance matrix of the ungauged projectors.
        The mode order is
            {l1_1, l1_2, r1_1, r1_2, l2_1, l2_2, r2_1, r2_2, l3_1...}
            or (for vertical links)
            {d1_1, d1_2, u1_1, u1_2, d2_1, d2_2, u2_1, u2_2, d3_1...}.
        The naming convention here is <mode letter><number of copy>_<majorana mode>.
        We order first by link and then by copy.
        The sites are picked such that the left mode is right of the right modes,
        i.e. they are sitting on the same link.
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

        # We want to give the projectors for the pure gauge part, which mix copies
        mixed_X = np.real_if_close(
            1.0j * np.kron(utils.paulix, np.kron(utils.pauliy, utils.paulix))
        )
        mixed_Y = np.real_if_close(
            1.0j * np.kron(utils.paulix, np.kron(utils.pauliy, utils.pauliz))
        )

        dest_mixed[Direction.X] = np.kron(np.eye(4), mixed_X)
        dest_mixed[Direction.Y] = np.kron(np.eye(4), mixed_Y)

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

        dest_unmixed[Direction.X] = np.kron(np.eye(4), unmixed_X)
        dest_unmixed[Direction.Y] = np.kron(np.eye(4), unmixed_Y)

        return [dest_mixed] * self.num_pg_layer + [
            dest_unmixed
        ] * self.num_fermionic_layer
