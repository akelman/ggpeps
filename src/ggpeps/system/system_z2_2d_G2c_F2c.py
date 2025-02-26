import sys
import sympy
import logging
from typing import List

import numpy as np
from ggpeps import xnp as xnp
from ggpeps import xscipy as xscipy

import ggpeps
from ggpeps import utils
from ggpeps.lattice import Direction
from ggpeps.system.global_funcs import *

from .system_base import Config2DBase
from .system_base import get_pfaffian_arrays


logger = logging.getLogger(ggpeps.LOGGER_NAME)


class Z2System2D_G2C_F2C_Config(Config2DBase):
    """Configuration of the Z2 system in 2D with 2 copies of virtual fermions on the links per layer.
    Each layer can either be pure-gauge (in which case the t-params are zeroed out),
    or fermionic (in which case the y,z-params are zeroed out).

    Some general notes about conventions:

    Order of the paramvec: [t1r,y1r,z1r,t2r,y2r,z2r,ar,br,cr,dr,t1i,y1i,z1i,t2i,y2i,z2i,ai,bi,ci,di].
    Mode order of tmat: {p,l1,r1,d1,u1,l2,r2,d2,u2}.
    Mode order of gamma_dirac: {p,l1,r1,d1,u1,l2,r2,d2,u2,p_dag,l1_dag,r1_dag,u1_dag,d1_dag,l2_dat,r2_dag,u2_dag,d2_dag}.
    Mode order of gamma_maj: {p_1,p_2,l1_1,l1_2,r1_1,r1_2,d1_1,d1_2,u1_1,u1_2,l2_1,l2_2,r2_1,r2_2,d2_1,d2_2,u2_1,u2_2}.
    """

    _nparams_per_layer = 20
    ncopy = 2
    nvirtmodes_vertex = 8
    nvirtmodes_link = 4

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

        # Translation invariance (or variance)
        if unitcell_size not in [1, 2]:
            logger.error(
                "This ansatz only supports unitcell_size = 1 or 2. \
                         This can be adapted by adding in a specification in the config to map sites to parameters."
            )
            sys.exit(1)
        # map from site to index of independent parameters (default is unitcell_size = 1)
        self.site_params_dict = {site: 0 for site in range(self.lattice.size)}

        # For now, we use hard code the unitcell_size = 1 or 2 case
        # More general ways to do so are supported - just change these lines
        if unitcell_size == 2:
            for site in range(self.lattice.size):
                x, y = self.lattice.ind2coord(site)
                uc_ind = 1 if (x + y) % 2 else 0  # 0 for even sublattice, 1 for odd
                self.site_params_dict[site] = uc_ind
        self.unitcell_size = len(
            set(self.site_params_dict.values())
        )  # number of different sets of parameters across sites (min: 1, max: num_sites)
        if self.unitcell_size != unitcell_size:
            # It should be impossible to reach here
            raise ValueError("Inconsistent unitcell_size.")

        # U1 invariance
        # set to True if you want to enforce U(1) symmetry in the fermionic layers
        # (set to False to allow fermionic number to float between sectors)
        self.u1_symmetry = enforce_u1_symmetry

        # Constants used in the calculation of the electric energy
        prefactors = [[1, -1, 1.0j, 1.0j], [1, -1, 1.0j, 1.0j]]
        indices_layer_pg = [
            [(2, 4), (3, 5), (4, 5), (2, 3)],
            [(6, 0), (7, 1), (0, 1), (6, 7)],
        ]
        indices_layer_fermionic = [
            [(2, 0), (3, 1), (0, 1), (2, 3)],
            [(6, 4), (7, 5), (4, 5), (6, 7)],
        ]
        idxarr_lay_pg = get_pfaffian_arrays(indices_layer_pg, prefactors)
        idxarr_lay_fermionic = get_pfaffian_arrays(indices_layer_fermionic, prefactors)
        self.idxarr_vec = [idxarr_lay_pg] * self.num_pg_layer + [
            idxarr_lay_fermionic
        ] * self.num_fermionic_layer
        self.el_overall_factors = [
            -1 / 16
        ] * self.nlayer  # this arises due to normalization and the i^(# of modes/2) in the expression Tr[i^# * rho * (modes)]

    def make_pure_gauge(self):
        """Make the ansatz pure gauge by setting t-params to zero.

        This function is obsolete for this ansatz, and is kept for compatibility reasons.
        """
        t_indices = [0, 3, 10, 13]  # index of t1r, t2r, t1i, t2i in symbolvec
        for layer_ind in range(self.nlayer):
            for uc_ind in range(self.unitcell_size):
                for t_ind in t_indices:
                    coord = (layer_ind, uc_ind, t_ind)
                    self.paramvec[coord] = 0

    def enforce_parameter_conditions(self, mat):
        """Enforce conditions on parameters on each layer to get the required behaviour for the ansatz."""
        # The order of the parameters (for each layer) is [t1r,y1r,z1r,t2r,y2r,z2r,ar,br,cr,dr,t1i,y1i,z1i,t2i,y2i,z2i,ai,bi,ci,di]

        zeroed_params = []  # we'll save the indices of the zeroed parameters

        t_indices = [0, 3, 10, 13]  # index of t1r, t2r, t1i, t2i in symbolvec
        for layer_ind in range(self.num_pg_layer):
            for uc_ind in range(self.unitcell_size):
                for t_ind in t_indices:
                    coord = (layer_ind, uc_ind, t_ind)
                    if isinstance(mat, np.ndarray):  # TODO: handle jax better
                        mat[coord] = 0
                    else:
                        mat = mat.at[coord].set(0)
                    zeroed_params.append(coord)

        if self.u1_symmetry:
            zero_for_fermionic_layer = [
                3,
                13,
                1,
                2,
                4,
                5,
                11,
                12,
                14,
                15,
            ]  # index of t2r, t2i, y1r, z1r, y2r, z2r, y1i, z1i, y2i, z2i in symbolvec
        else:
            zero_for_fermionic_layer = []
        for layer_ind in range(self.num_pg_layer, self.nlayer):
            for uc_ind in range(self.unitcell_size):
                for ind in zero_for_fermionic_layer:
                    coord = (layer_ind, uc_ind, ind)
                    if isinstance(mat, np.ndarray):
                        mat[coord] = 0
                    else:
                        mat = mat.at[coord].set(0)
                    zeroed_params.append(coord)

        # save zeroed params
        self.zeroed_params = zeroed_params
        return

    def _create_symbolvec(self) -> List[sympy.Symbol]:
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
        The definition of T here is a result of an analytic consideration of global symmetries like rotational invariance, charge conjugation invarance, etc.
        The T matrix is given in terms of symbols to compute the derivative of the covariance matrices analytically via sympy.
        We do not have to type them explicitly anymore into the code.

        This is one of two analytic inputs into the code.
        The other input is the structure and the parametrization of the projectors.

        The mode order is: Psi, l_1, r_1, d_1, u_1, l_2, r_2, d_2, u_2

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
        """Generate the covariance matrix of the ungauged projectors.
        The mode order is {l1_1, l1_2, r1_1, r1_2, l2_1, l2_2, r2_1, r2_2}/{d1_1, d1_2, u1_1, u1_2, d2_1, d2_2, u2_1, u2_2}.
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
            List[xnp.ndarray]: Covariance matrices of the ungauged projector on a single link
        """

        # 2 if for 2D lattice
        dest_mixed = [0] * 2  # mixes copies
        dest_unmixed = [0] * 2  # does not mix copies

        # We want to give the projectors for the pure gauge part, which mix copies
        dest_mixed[Direction.X] = np.real_if_close(
            1.0j * np.kron(utils.paulix, np.kron(utils.pauliy, utils.paulix))
        )
        dest_mixed[Direction.Y] = np.real_if_close(
            1.0j * np.kron(utils.paulix, np.kron(utils.pauliy, utils.pauliz))
        )

        # We want to give the projectors for the fermionic part which don't mix copies (so as to preserve global U(1) symmetry)
        dest_unmixed[Direction.X] = np.array(
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

        dest_unmixed[Direction.Y] = np.array(
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

        return np.array(
            [dest_mixed] * self.num_pg_layer + [dest_unmixed] * self.num_fermionic_layer
        )
