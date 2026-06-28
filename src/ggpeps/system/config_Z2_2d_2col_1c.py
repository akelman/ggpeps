"""Z2 ansatz with a 2D representation that mixes 2 colors and 1 copy.

Diagnostic ansatz: same as Z2System2D_2col_Config (config_Z2_2d_2col_2c.py) but with
ncopy=1 instead of ncopy=2. This removes all cross-copy parameters (a, b, c, d) and
their associated T-matrix blocks, leaving only (t1, y1, z1) per color.

Purpose: further isolate the electric-energy discrepancy. The Z2_2col (ncopy=2) ansatz
reproduces the D6 bug; this single-copy variant lets us test whether the bug requires
ncopy=2 (cross-copy mixing in the T-matrix) or already appears for ncopy=1.
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


class Z2System2D_2col_1copy_Config(Config2DBase):
    """Z_2 ansatz with 2D rep, 1 copy, 2 colors.

    Mirrors Z2System2D_2col_Config but with ncopy=1: the T-matrix is block-diagonal in
    the color index, each block carries the standard Z_2 single-copy 5x5 parameters
    (t1, y1, z1). No cross-copy parameters exist.

    Parameter vector layout (per layer):
        [t1r, y1r, z1r, t1i, y1i, z1i]
    """

    _nparams = 6
    ncopy = 1
    nvirtmodes_vertex = 8  # 2 colors * 1 copy * 4 link directions * 1 mode = 8
    nvirtmodes_link = 4  # 1 copy * 2 colors * 2 Majorana per link side = 4
    nphysmodes_site = 2  # one physical mode per color
    ncolors = 2
    nvirtmodes_link_per_color = 2  # 1 copy * 2 Majorana

    def __init__(
        self,
        lattice,
        g_el,
        g_mag,
        g_int,
        g_mass,
        g_chem,
        ncopy=1,
        num_pg_layer=1,
        num_fermionic_layer=0,
        mod_link_inds=(0,),
        unitcell_size=1,
        enforce_u1_symmetry=True,
        param_constraints="current",
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
            raise ValueError(f"Z2System2D_2col_1copy_Config expects ncopy={self.ncopy}, got ncopy={ncopy}.")
        if param_constraints != "current":
            raise ValueError(
                "Z2System2D_2col_1copy_Config only supports param_constraints='current'. "
                f"Got param_constraints={param_constraints!r}."
            )

        if self.unitcell_size not in [1]:
            logger.error("Z2System2D_2col_1copy only supports unitcell_size = 1.")
            sys.exit(1)

        self.init_el_energy_terms()

    def init_el_energy_terms(self) -> None:
        idx_vec = []
        coeffs_vec = []
        constants_vec = []

        for group_element in self.gaugemgr.group_elements_for_el_energy:
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
                        term_ferm, const_ferm = idxarr_pg_v_0, const_pg_v_0
                    else:
                        term_pg, const_pg = idxarr_pg_v_1, const_pg_v_1
                        term_ferm, const_ferm = idxarr_pg_v_1, const_pg_v_1
                else:
                    if site_parity == 0:
                        term_pg, const_pg = idxarr_pg_h_0, const_pg_h_0
                        term_ferm, const_ferm = idxarr_pg_h_0, const_pg_h_0
                    else:
                        term_pg, const_pg = idxarr_pg_h_1, const_pg_h_1
                        term_ferm, const_ferm = idxarr_pg_h_1, const_pg_h_1

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

        self.idx_vec = tuple(idx_vec)
        self.coeffs_vec = tuple(coeffs_vec)
        self.constants_vec = tuple(constants_vec)

    def make_pure_gauge(self):
        """Zero the t-parameters in every layer (pure-gauge ansatz)."""
        t_indices = [0, 3]  # t1r, t1i
        for layer_ind in range(self.nlayer):
            for uc_ind in range(self.unitcell_size):
                for t_ind in t_indices:
                    self.paramvec[layer_ind, uc_ind, t_ind] = 0

    def get_zeroed_params(self):
        zeroed_params = []
        t_indices = [0, 3]  # t1r, t1i
        for layer_ind in range(self.num_pg_layer):
            for uc_ind in range(self.unitcell_size):
                for t_ind in t_indices:
                    zeroed_params.append((layer_ind, uc_ind, t_ind))
        return tuple(zeroed_params)

    def _create_symbolvec(self) -> list[sympy.Symbol]:
        names = ["t1r", "y1r", "z1r", "t1i", "y1i", "z1i"]
        return [sympy.Symbol(n, real=True) for n in names]

    @property
    def tmat_symb(self):
        """Block-diagonal T-matrix: single-copy Z_2 5x5 block doubled across the two colors."""
        [t1r, y1r, z1r, t1i, y1i, z1i] = self.symbolvec
        t1 = t1r + 1.0j * t1i
        y1 = y1r + 1.0j * y1i
        z1 = z1r + 1.0j * z1i

        # Single-color, single-copy 5x5 block.
        # Mode order within one color: Psi, l, r, d, u (copy 1 only).
        tmat_block = sympy.Matrix(
            [
                [0, -1.0j * t1, 1.0j * t1, t1, -t1],
                [1.0j * t1, 0, 1.0j * y1, z1, 1.0j * z1],
                [-1.0j * t1, -1.0j * y1, 0, -1.0j * z1, -z1],
                [-t1, -z1, 1.0j * z1, 0, -y1],
                [t1, -1.0j * z1, z1, y1, 0],
            ]
        )

        # Two color blocks with the SAME parameters: T = block_diag(T_block, T_block).
        tmat_symb = sympy.Matrix(sympy.BlockDiagMatrix(tmat_block, tmat_block))

        # Reorder modes to color-outer: {Psi1, Psi2, l_1, r_1, d_1, u_1, l_2, r_2, d_2, u_2}.
        wrong_order = ["Psi1", "l_1", "r_1", "d_1", "u_1", "Psi2", "l_2", "r_2", "d_2", "u_2"]
        correct_order = ["Psi1", "Psi2", "l_1", "r_1", "d_1", "u_1", "l_2", "r_2", "d_2", "u_2"]
        perm = np.array(modearray.generate_permutation_matrix(wrong_order, correct_order))
        return np.transpose(perm) @ tmat_symb @ perm

    def generate_gamma_gauge_neutral_dict(self):
        """Block-diagonal across the two colors, single-color block is the 1-copy version.

        Mode order (same convention as D6/Z2_2col):
            {l1_1_1, l1_2_1, r1_1_1, r1_2_1, l1_1_2, l1_2_2, r1_1_2, r1_2_2}
        i.e. <letter><copy>_<majorana>_<color>, color-outer.
        """
        dest_unmixed = [0] * 2

        # Single-color, single-copy 4x4 block for horizontal (X) link covariance.
        # Taken from the top-left 4x4 of the 2-copy 8x8 single-color block.
        # Mode order: {l1_1, l1_2, r1_1, r1_2} — l and r Majorana modes for 1 copy.
        blockumixed_X_1c = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
            ]
        )
        # Same for vertical (Y) link covariance.
        blockumixed_Y_1c = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )

        # The naming convention in the 2-copy config uses blockumixed_X for Direction.Y
        # and blockumixed_Y for Direction.X — preserved here for consistency.
        dest_unmixed[Direction.X] = block_diag(blockumixed_X_1c, blockumixed_X_1c)
        dest_unmixed[Direction.Y] = block_diag(blockumixed_Y_1c, blockumixed_Y_1c)

        return np.array([dest_unmixed] * self.num_pg_layer + [dest_unmixed] * self.num_fermionic_layer)
