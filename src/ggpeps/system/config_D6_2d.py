import sys
import sympy
import logging

import numpy as np
from scipy.linalg import block_diag

import ggpeps
from ggpeps import gauge
from ggpeps import modearray
from ggpeps.lattice import Direction

from .config_base import Config2DBase, generate_gauged_projector_terms


logger = logging.getLogger(ggpeps.LOGGER_NAME)


class D6System2D_Config(Config2DBase):
    """
    Configuration of the D2n system in 2D with 2 copies of virtual fermions on the links per layer.
    Each copy has 2 colors.
    Each layer can either be pure-gauge (in which case the t-params are zeroed out),
    or fermionic (in which case the y,z-params are zeroed out).

    Some general notes about conventions:

    Order of the paramvec: [t1r,y1r,z1r,t2r,y2r,z2r,ar,br,cr,dr,t1i,y1i,z1i,t2i,y2i,z2i,ai,bi,ci,di].
    Mode order of tmat: The mode order is:
        Psi_1,Psi_2, l1_1,l1_2, r1_1,r1_2,d1_1, d1_2, u1_1,u1_2, l2_1, l2_2, r2_1,r2_2, d2_1,d2_2, u2_1, u2_2
    Where the modes are labelled by directin{copy}_{color}
    Where the 1 and 2 virtual copies and the Psi are the m=0,
    and the 3 and 4 virtual copies and Psi2 are of the color m=1.

    Mode order of gamma_dirac:
        {p,l1,r1,d1,u1,l2,r2,d2,u2,p_dag,l1_dag,r1_dag,u1_dag,d1_dag,l2_dat,r2_dag,u2_dag,d2_dag}.
    Mode order of gamma_maj:
        {p_1,p_2,l1_1, l1_2, r1_1, r1_2, l2_1, l2_2, r2_1, r2_2,l3_1, l3_2, r3_1, r3_2, l4_1, l4_2, r4_1, r4_2,
        d1_1, d1_2, u1_1, u1_2, d2_1, d2_2, u2_1, u3_2,d3_1, d3_2, u3_1, u3_2, d4_1, d4_2, u4_1, u4_2}.
    """

    _nparams = 20
    ncopy = 2
    nvirtmodes_vertex = 16
    nvirtmodes_link = 8
    nphysmodes_site = 2  # number of physical modes per site
    ncolors = 2  # number of colors
    nvirtmodes_link_per_color = 4  # number of virtual modes per link per color

    def __init__(
        self,
        lattice,
        g_el,
        g_mag,
        g_int,
        g_mass,
        g_chem,
        num_pg_layer=2,
        num_fermionic_layer=0,
        mod_link_inds=(0,),
        unitcell_size=1,
        enforce_u1_symmetry=True,
    ) -> None:
        self.gaugemgr: gauge.D2nGauge
        super().__init__(
            gauge.D2nGauge(3),
            lattice,
            g_el,
            g_mag,
            g_int,
            g_mass,
            g_chem,
            num_pg_layer,
            num_fermionic_layer,
            mod_link_inds,
            unitcell_size,
            enforce_u1_symmetry,
        )

        # Translation invariance (or variance)
        if self.unitcell_size not in [1]:
            logger.error(
                "For Dn groups this ansatz only supports unitcell_size = 1. \
                This can be adapted by adding in a specification in the config to map sites to parameters."
            )
            sys.exit(1)

        self.init_el_energy_terms()

    def init_el_energy_terms(self) -> None:
        """Build idx_vec, coeffs_vec and constants_vec."""
        idx_vec = []
        coeffs_vec = []
        constants_vec = []

        for group_element in self.gaugemgr.group_elements_for_el_energy:
            # Pure Gauge (PG) ---
            idxarr_pg_h_0, const_pg_h_0 = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.X, group_element, site=0
            )
            idxarr_pg_h_1, const_pg_h_1 = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.X, group_element, site=1
            )
            idxarr_pg_v_0, const_pg_v_0 = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.Y, group_element, site=0
            )
            idxarr_pg_v_1, const_pg_v_1 = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.Y, group_element, site=1
            )

            # generate fermionic terms
            # Reuse the exact same variables for the fermionic terms
            idxarr_ferm_h_0, const_ferm_h_0 = idxarr_pg_h_0, const_pg_h_0
            idxarr_ferm_h_1, const_ferm_h_1 = idxarr_pg_h_1, const_pg_h_1
            idxarr_ferm_v_0, const_ferm_v_0 = idxarr_pg_v_0, const_pg_v_0
            idxarr_ferm_v_1, const_ferm_v_1 = idxarr_pg_v_1, const_pg_v_1

            pg_link_coeffs, pg_link_indices = [], []
            ferm_link_coeffs, ferm_link_indices = [], []
            pg_link_contants, ferm_link_constants = [], []

            for mod_link in self.mod_link_inds:
                coord, dir = self.lattice.ind2coord_dir(mod_link)
                site_parity = sum(coord) % 2
                is_vertical = dir == Direction.Y

                # Select the correct base terms based on direction and parity
                if is_vertical:
                    if site_parity == 0:
                        term_pg = idxarr_pg_v_0
                        term_ferm = idxarr_ferm_v_0
                        contant_pg = const_pg_v_0
                        contant_ferm = const_ferm_v_0
                    else:
                        term_pg = idxarr_pg_v_1
                        term_ferm = idxarr_ferm_v_1
                        contant_pg = const_pg_v_1
                        contant_ferm = const_ferm_v_1
                else:
                    if site_parity == 0:
                        term_pg = idxarr_pg_h_0
                        term_ferm = idxarr_ferm_h_0
                        contant_pg = const_pg_h_0
                        contant_ferm = const_ferm_h_0
                    else:
                        term_pg = idxarr_pg_h_1
                        term_ferm = idxarr_ferm_h_1
                        contant_pg = const_pg_h_1
                        contant_ferm = const_ferm_h_1

                pg_c, pg_i = self._bucket_sort_terms(term_pg)
                pg_link_coeffs.append(pg_c)
                pg_link_indices.append(pg_i)
                pg_link_contants.append(contant_pg)

                ferm_c, ferm_i = self._bucket_sort_terms(term_ferm)
                ferm_link_coeffs.append(ferm_c)
                ferm_link_indices.append(ferm_i)
                ferm_link_constants.append(contant_ferm)

            pg_base_constants = tuple(pg_link_contants)
            ferm_base_constants = tuple(ferm_link_constants)
            constants_vec.append(
                (pg_base_constants,) * self.num_pg_layer + (ferm_base_constants,) * self.num_fermionic_layer
            )

            pg_base_coeffs = tuple(pg_link_coeffs)
            ferm_base_coeffs = tuple(ferm_link_coeffs)

            # Stack layers: PG layers first, then Fermionic layers
            coeffs_vec.append((pg_base_coeffs,) * self.num_pg_layer + (ferm_base_coeffs,) * self.num_fermionic_layer)

            pg_base_indices = tuple(pg_link_indices)
            ferm_base_indices = tuple(ferm_link_indices)

            idx_vec.append((pg_base_indices,) * self.num_pg_layer + (ferm_base_indices,) * self.num_fermionic_layer)

        self.idx_vec = tuple(idx_vec)
        self.coeffs_vec = tuple(coeffs_vec)
        self.constants_vec = tuple(constants_vec)

    def make_pure_gauge(self):
        """Make the ansatz pure gauge by setting t-params to zero.

        This function is obsolete for this ansatz, and is kept for some tests.
        """
        t_indices = [0, 3, 10, 13]  # index of t1r, t2r, t1i, t2i in symbolvec
        for layer_ind in range(self.nlayer):
            for uc_ind in range(self.unitcell_size):
                for t_ind in t_indices:
                    coord = (layer_ind, uc_ind, t_ind)
                    self.paramvec[coord] = 0

    def get_zeroed_params(self):
        # The order of the parameters (for each layer) is
        #   [t1r,y1r,z1r,t2r,y2r,z2r,ar,br,cr,dr,t1i,y1i,z1i,t2i,y2i,z2i,ai,bi,ci,di]

        zeroed_params = []  # we'll save the indices of the zeroed parameters

        t_indices = [0, 3, 10, 13]  # index of t1r, t2r, t1i, t2i in symbolvec
        for layer_ind in range(self.num_pg_layer):
            for uc_ind in range(self.unitcell_size):
                for t_ind in t_indices:
                    coord = (layer_ind, uc_ind, t_ind)
                    zeroed_params.append(coord)

        if self.u1_symmetry:
            # index of t2r, t2i, y1r, z1r, y2r, z2r, y1i, z1i, y2i, z2i in symbolvec:
            zero_for_fermionic_layer = [3, 13, 1, 2, 4, 5, 11, 12, 14, 15]
        else:
            zero_for_fermionic_layer = []
        for layer_ind in range(self.num_pg_layer, self.nlayer):
            for uc_ind in range(self.unitcell_size):
                for ind in zero_for_fermionic_layer:
                    coord = (layer_ind, uc_ind, ind)
                    zeroed_params.append(coord)

        return tuple(zeroed_params)

    def _create_symbolvec(self) -> list[sympy.Symbol]:
        """Define all symbols of the T matrix as symbols.
        We will use the analytic expression of the T matrix to calculate the derivative of the
        covariance matrices analytically.

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

        The mode order is:
            Psi_1,Psi_2, l1_1, r1_1,d1_1,u1_1, l2_1, r2_1,d2_1,u2_1, l1_2, r1_2,d1_2,u1_2,l2_2, r2_2,d2_2,u2_2
        Where the modes are labelled by directin{copy}_{color}.
        The order {l,r,d,u} instead of {r,u,l,d} (used in some analytic calculations) because it
        eliminates the need for a lot of permutation matrices in the conversion from T to gamma_maj.
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
        tmat_symb_block = (
            sympy.Matrix(  # For a single color. Both colors have the same structure as well as parameters
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
        )
        # Define the 16x16 block diagonal tmat matrix for the two different colors in the order of:
        #  Psi1, l_1, r_1, d_1, u_1, l_2, r_2, d_2, u_2, Psi2, l_3, r_3, d_3, u_3, l_4, r_4, d_4, u_4,
        # where 3 and 4 are the m=2 color
        tmat_symb = sympy.BlockDiagMatrix(tmat_symb_block, tmat_symb_block)

        # Convert it to a regular Matrix
        tmat_symb = sympy.Matrix(tmat_symb)  # this is the T matrix in
        wrong_order = [
            "Psi1",
            "l_1",
            "r_1",
            "d_1",
            "u_1",
            "l_2",
            "r_2",
            "d_2",
            "u_2",
            "Psi2",
            "l_3",
            "r_3",
            "d_3",
            "u_3",
            "l_4",
            "r_4",
            "d_4",
            "u_4",
        ]
        correct_order = [
            "Psi1",
            "Psi2",
            "l_1",
            "r_1",
            "d_1",
            "u_1",
            "l_2",
            "r_2",
            "d_2",
            "u_2",
            "l_3",
            "r_3",
            "d_3",
            "u_3",
            "l_4",
            "r_4",
            "d_4",
            "u_4",
        ]
        perm = np.array(
            modearray.generate_permutation_matrix(
                wrong_order,
                correct_order,
            )
        )
        tmat_symb = np.transpose(perm) @ tmat_symb @ perm  # permute the modes to the correct order
        return tmat_symb

    def generate_gamma_gauge_neutral_dict(self):
        r"""Generate the covariance matrix of the ungauged projectors.
        The mode order is
        {l1_1_1,l1_2_1,r1_1_1,r1_2_1,l2_1_1,l2_2_1,r2_1_1,r2_2_1,l1_1_2,l1_2_2,r1_1_2,r1_2_2,l2_1_2,l2_2_2,r2_1_2,r2_2_2}
        /{d1_1_1,d1_2_1,u1_1_1,u1_2_1,d2_1_1,d2_2_1,u2_1_1,u2_2_1,d1_1_2,d1_2_2,u1_1_2,u1_2_2,d2_1_2,d2_2_2,u2_1_2,u2_2_2}.
        The naming convention here is <mode letter><number of copy>_<majorana mode>_<color>.
        We order first by link and then by copy.
        The sites are picked such that the left mode is right of the right modes,
        i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        This function returns the covariance matrices for ungauged projectors:
        Unlike the Z2, copies are always coupled to themselves without mixing different copies.
        Here we work in the same convention as Z2 w=exp(rl)and not w=exp(r^{\dagger}l^{\dagger}) as in some references.

        This method overwrites an abstract method in System2DBase.

        Returns:
            list[xnp.ndarray]: Covariance matrices of the ungauged projector on a single link
        """

        # 2 if for 2D lattice
        dest_unmixed = [0] * 2  # does not mix copies
        blockumixed_X = np.array(  # Block for a single color
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
        dest_unmixed[Direction.X] = block_diag(blockumixed_X, blockumixed_X)
        # Creating the matrix in the mode order of
        # {l1_1_1,l1_2_1,r1_1_1,r1_2_1,l2_1_1,l2_2_1,r2_1_1,r2_2_1,l1_1_2,l1_2_2,r1_1_2,r1_2_2,l2_1_2,l2_2_2,r2_1_2,r2_2_2}
        blockumixed_Y = np.array(
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
        dest_unmixed[Direction.Y] = block_diag(blockumixed_Y, blockumixed_Y)

        return np.array([dest_unmixed] * self.num_pg_layer + [dest_unmixed] * self.num_fermionic_layer)
