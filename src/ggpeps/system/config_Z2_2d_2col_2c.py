"""Z2 ansatz with a 2D representation that mixes 2 colors and 2 copies.

This is a diagnostic ansatz that mirrors the D6 (2 colors, 2 copies) layout but uses
Z_2 as the gauge group, represented by 2x2 matrices (D(+1) = I_2, D(-1) = sigma_x).
The T-matrix is the standard Z_2 9x9 block, doubled block-diagonally across the two
colors. The same parameters appear in both color blocks (no cross-color parameters,
matching the symmetry constraint discussed for D6).

Purpose: isolate whether the electric-energy discrepancy observed for D6 is specific
to D6 or appears for any 2D-rep gauging on top of the multi-color (color-diagonal-T)
structure.
"""

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


class Z2System2D_2col_Config(Config2DBase):
    """Z_2 ansatz with 2D rep, 2 copies, 2 colors.

    Layout mirrors D6System2D_Config: the T-matrix is block-diagonal in the color index
    and each block carries the standard Z_2 (1-color, 2-copy) parameters. The gauging
    uses a 2x2 representation, so M_{beta,alpha}(h) mixes colors exactly the way it
    does for D6 reflections.

    Parameter vector layout (per layer):
        [t1r, y1r, z1r, t2r, y2r, z2r, ar, br, cr, dr,
         t1i, y1i, z1i, t2i, y2i, z2i, ai, bi, ci, di]
    """

    _nparams = 20
    ncopy = 2
    nvirtmodes_vertex = 16  # 2 colors * (4 link directions * 2 modes) = 16
    nvirtmodes_link = 8  # 2 copies * 2 colors * (l/r or u/d) * 1 fermion per direction
    nphysmodes_site = 2  # one physical mode per color (mirrors D6 layout for symmetry)
    ncolors = 2
    nvirtmodes_link_per_color = 4

    def __init__(
        self,
        lattice,
        g_el,
        g_mag,
        g_int,
        g_mass,
        g_chem,
        ncopy=2,
        num_pg_layer=1,
        num_fermionic_layer=0,
        mod_link_inds=(0,),
        unitcell_size=1,
        enforce_u1_symmetry=True,
    ) -> None:
        self.gaugemgr: gauge.Z2RepGauge2D
        super().__init__(
            gauge.Z2RepGauge2D(),
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
        if ncopy != self.ncopy:
            raise ValueError(f"Z2System2D_2col_Config expects ncopy={self.ncopy}, got ncopy={ncopy}.")

        if self.unitcell_size not in [1]:
            logger.error("Z2System2D_2col only supports unitcell_size = 1.")
            sys.exit(1)

        self.init_el_energy_terms()

    # Same el-energy assembly as D6System2D_Config: per-link Majorana-polynomial
    # buckets keyed by (group_element, layer, link, term_size).
    def init_el_energy_terms(self) -> None:
        idx_vec = []
        coeffs_vec = []
        constants_vec = []

        for group_element in self.gaugemgr.group_elements_for_el_energy:
            # mix_copies=False matches the D6 convention (sigma = identity).
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

            idxarr_ferm_h_0, const_ferm_h_0 = idxarr_pg_h_0, const_pg_h_0
            idxarr_ferm_h_1, const_ferm_h_1 = idxarr_pg_h_1, const_pg_h_1
            idxarr_ferm_v_0, const_ferm_v_0 = idxarr_pg_v_0, const_pg_v_0
            idxarr_ferm_v_1, const_ferm_v_1 = idxarr_pg_v_1, const_pg_v_1

            pg_link_coeffs, pg_link_indices = [], []
            ferm_link_coeffs, ferm_link_indices = [], []
            pg_link_constants, ferm_link_constants = [], []

            for mod_link in self.mod_link_inds:
                coord, dir_ = self.lattice.ind2coord_dir(mod_link)
                site_parity = sum(coord) % 2
                is_vertical = dir_ == Direction.Y

                if is_vertical:
                    if site_parity == 0:
                        term_pg, const_pg = idxarr_pg_v_0, const_pg_v_0
                        term_ferm, const_ferm = idxarr_ferm_v_0, const_ferm_v_0
                    else:
                        term_pg, const_pg = idxarr_pg_v_1, const_pg_v_1
                        term_ferm, const_ferm = idxarr_ferm_v_1, const_ferm_v_1
                else:
                    if site_parity == 0:
                        term_pg, const_pg = idxarr_pg_h_0, const_pg_h_0
                        term_ferm, const_ferm = idxarr_ferm_h_0, const_ferm_h_0
                    else:
                        term_pg, const_pg = idxarr_pg_h_1, const_pg_h_1
                        term_ferm, const_ferm = idxarr_ferm_h_1, const_ferm_h_1

                pg_c, pg_i = self._bucket_sort_terms(term_pg)
                pg_link_coeffs.append(pg_c)
                pg_link_indices.append(pg_i)
                pg_link_constants.append(const_pg)

                ferm_c, ferm_i = self._bucket_sort_terms(term_ferm)
                ferm_link_coeffs.append(ferm_c)
                ferm_link_indices.append(ferm_i)
                ferm_link_constants.append(const_ferm)

            constants_vec.append(
                (tuple(pg_link_constants),) * self.num_pg_layer
                + (tuple(ferm_link_constants),) * self.num_fermionic_layer
            )
            coeffs_vec.append(
                (tuple(pg_link_coeffs),) * self.num_pg_layer + (tuple(ferm_link_coeffs),) * self.num_fermionic_layer
            )
            idx_vec.append(
                (tuple(pg_link_indices),) * self.num_pg_layer + (tuple(ferm_link_indices),) * self.num_fermionic_layer
            )

        self.set_el_energy_terms(idx_vec, coeffs_vec, constants_vec)

    def make_pure_gauge(self):
        """Zero the t-parameters in every layer (pure-gauge ansatz)."""
        t_indices = [0, 3, 10, 13]
        for layer_ind in range(self.nlayer):
            for uc_ind in range(self.unitcell_size):
                for t_ind in t_indices:
                    self.paramvec[layer_ind, uc_ind, t_ind] = 0

    def get_zeroed_params(self):
        zeroed_params = []
        t_indices = [0, 3, 10, 13]
        for layer_ind in range(self.num_pg_layer):
            for uc_ind in range(self.unitcell_size):
                for t_ind in t_indices:
                    zeroed_params.append((layer_ind, uc_ind, t_ind))

        if self.u1_symmetry:
            zero_for_fermionic_layer = [3, 13, 1, 2, 4, 5, 11, 12, 14, 15]
        else:
            zero_for_fermionic_layer = []
        for layer_ind in range(self.num_pg_layer, self.nlayer):
            for uc_ind in range(self.unitcell_size):
                for ind in zero_for_fermionic_layer:
                    zeroed_params.append((layer_ind, uc_ind, ind))
        return tuple(zeroed_params)

    def _create_symbolvec(self) -> list[sympy.Symbol]:
        names = [
            "t1r",
            "y1r",
            "z1r",
            "t2r",
            "y2r",
            "z2r",
            "ar",
            "br",
            "cr",
            "dr",
            "t1i",
            "y1i",
            "z1i",
            "t2i",
            "y2i",
            "z2i",
            "ai",
            "bi",
            "ci",
            "di",
        ]
        return [sympy.Symbol(n, real=True) for n in names]

    @property
    def tmat_symb(self):
        """Block-diagonal T-matrix: standard Z_2 9x9 block doubled across the two colors."""
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

        # Same 9x9 block as the standard Z_2 2-copy ansatz (config_Z2_2d_2c.py).
        # Mode order within one color: Psi, l, r, d, u (copy 1), l, r, d, u (copy 2).
        tmat_block = sympy.Matrix(
            [
                [0, -1.0j * t1, 1.0j * t1, t1, -t1, -1.0j * t2, 1.0j * t2, t2, -t2],
                [1.0j * t1, 0, 1.0j * y1, z1, 1.0j * z1, -1.0j * a, -1.0j * c, -1.0j * b, -1.0j * d],
                [-1.0j * t1, -1.0j * y1, 0, -1.0j * z1, -z1, 1.0j * c, 1.0j * a, 1.0j * d, 1.0j * b],
                [-t1, -z1, 1.0j * z1, 0, -y1, d, b, a, c],
                [t1, -1.0j * z1, z1, y1, 0, -b, -d, -c, -a],
                [1.0j * t2, 1.0j * a, -1.0j * c, -d, b, 0, 1.0j * y2, z2, 1.0j * z2],
                [-1.0j * t2, 1.0j * c, -1.0j * a, -b, d, -1.0j * y2, 0, -1.0j * z2, -z2],
                [-t2, 1.0j * b, -1.0j * d, -a, c, -z2, 1.0j * z2, 0, -y2],
                [t2, 1.0j * d, -1.0j * b, -c, a, -1.0j * z2, z2, y2, 0],
            ]
        )

        # Two color blocks with the SAME parameters: T = block_diag(T_block, T_block).
        tmat_symb = sympy.Matrix(sympy.BlockDiagMatrix(tmat_block, tmat_block))

        # Reorder modes to match D6: {Psi1, Psi2, l_1, r_1, ..., u_4} (color-outer for
        # the physical modes too). This matches the layout the rest of the system
        # machinery expects.
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
        perm = np.array(modearray.generate_permutation_matrix(wrong_order, correct_order))
        return np.transpose(perm) @ tmat_symb @ perm

    def generate_gamma_gauge_neutral_dict(self):
        """Block-diagonal across the two colors, single-color block matches D6's blockumixed.

        Mode order (same as D6):
            {l1_1_1, l1_2_1, r1_1_1, r1_2_1, l2_1_1, l2_2_1, r2_1_1, r2_2_1,
             l1_1_2, l1_2_2, r1_1_2, r1_2_2, l2_1_2, l2_2_2, r2_1_2, r2_2_2}
        i.e. <letter><copy>_<majorana>_<color>, color-outer.
        """
        dest_unmixed = [0] * 2
        # Same 8x8 single-color blocks as the D6 ansatz, in the canonical convention
        # w = exp(rl) (not w = exp(r^dag l^dag)) per the system convention.
        blockumixed_X = np.array(
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
