import sys
import sympy
import logging

import numpy as np
from scipy.linalg import block_diag

import ggpeps
from ggpeps import gauge
from ggpeps import modearray
from ggpeps import utils
from ggpeps.lattice import Direction

from .config_base import Config2DBase, generate_gauged_projector_terms, make_sigma_matrix

logger = logging.getLogger(ggpeps.LOGGER_NAME)


class D6System2D_Config(Config2DBase):
    """
    Configuration of the D2n system in 2D with ncopy copies of virtual fermions on the links per layer.
    Each copy has 2 colors.
    Each layer can either be pure-gauge (in which case the t-params are zeroed out),
    or fermionic (in which case the y,z-params are zeroed out).

    Some general notes about conventions:

    Order of the paramvec: a [t_c, y_c, z_c] triple per copy, then an [a_pq, b_pq, c_pq, d_pq]
    quadruple per copy pair, pairs ordered (1,2), (1,3), ..., (2,3), ...; all real parts
    before all imaginary parts. Both colors share the same parameters. For ncopy=2:
        [t1r,y1r,z1r,t2r,y2r,z2r,a12r,b12r,c12r,d12r,t1i,y1i,z1i,t2i,y2i,z2i,a12i,b12i,c12i,d12i].
    Mode order of tmat (ncopy=2 example):
        Psi1, Psi2, l1_1, r1_1, d1_1, u1_1, l2_1, r2_1, d2_1, u2_1, l1_2, r1_2, d1_2, u1_2, l2_2, r2_2, d2_2, u2_2
    Where the modes are labelled by direction{copy}_{color}: the physical modes of all
    colors first, then per color all its copies (copies of color m are often indexed
    (m-1)*ncopy+1 .. m*ncopy in code).

    Mode order of gamma_dirac (the output of tmat_to_covariance_matrix): the Dirac
    annihilation modes in the same order as tmat above, followed by the creation
    (daggered) modes:
        [Psi1, Psi2, l1_1, r1_1, d1_1, u1_1, l2_1, ..., u2_2,
         Psi1_dag, Psi2_dag, l1_1_dag, ... , u2_2_dag].
    Mode order of gamma_maj (after generate_smat): each Dirac mode `a` is replaced by its
    two adjacent Majoranas gamma^1 = a + a_dag, gamma^2 = i(a - a_dag), preserving the
    tmat order. So the physical Majoranas come first, then for each (color, copy) the 8
    Majoranas of l, r, d, u:
        Psi1^1, Psi1^2, Psi2^1, Psi2^2,
        l1_1^1, l1_1^2, r1_1^1, r1_1^2, d1_1^1, d1_1^2, u1_1^1, u1_1^2, l2_1^1, ... , u2_2^2.
    """

    nphysmodes_site = 2  # number of physical modes per site
    ncolors = 2  # number of colors

    def __init__(
        self,
        lattice,
        g_el,
        g_mag,
        g_int,
        g_mass,
        g_chem,
        ncopy=2,
        num_pg_layer=2,
        num_fermionic_layer=0,
        mod_link_inds=(0,),
        unitcell_size=1,
        enforce_u1_symmetry=True,
    ) -> None:
        if not (isinstance(ncopy, int) and ncopy >= 1):
            raise ValueError(f"ncopy must be a positive integer, got {ncopy}.")
        self.ncopy = ncopy
        self.nvirtmodes_vertex = 4 * self.ncolors * ncopy
        self.nvirtmodes_link = 2 * self.ncolors * ncopy
        self._nparams = 2 * ncopy * (2 * ncopy + 1)

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
        if self.unitcell_size not in [1, 2, -1]:
            logger.error("This ansatz only supports unitcell_size = 1, 2, or -1 (all sites independent). \
                This can be adapted by adding in a specification in the config to map sites to parameters.")
            raise ValueError("Invalid unitcell_size.")

        self.init_el_energy_terms()

    def init_el_energy_terms(self) -> None:
        """Build the per-group-element el-energy terms and hand them to set_el_energy_terms."""
        idx_vec = []
        coeffs_vec = []
        constants_vec = []
        drop_im = False
        if self.nlayer == 1:
            drop_im = True

        for group_element in self.gaugemgr.group_elements_for_el_energy:
            # Pure Gauge (PG) ---
            idxarr_pg_h_0, const_pg_h_0 = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.X, group_element, site=0, drop_imag=drop_im
            )
            idxarr_pg_h_1, const_pg_h_1 = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.X, group_element, site=1, drop_imag=drop_im
            )
            idxarr_pg_v_0, const_pg_v_0 = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.Y, group_element, site=0, drop_imag=drop_im
            )
            idxarr_pg_v_1, const_pg_v_1 = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.Y, group_element, site=1, drop_imag=drop_im
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

        self.set_el_energy_terms(idx_vec, coeffs_vec, constants_vec)

    def make_pure_gauge(self):
        """Make the ansatz pure gauge by setting t-params to zero.

        This function is obsolete for this ansatz, and is kept for some tests.
        """
        offset = self._nparams // 2
        for layer_ind in range(self.nlayer):
            for uc_ind in range(self.unitcell_size):
                for cop in range(self.ncopy):
                    self.paramvec[layer_ind, uc_ind, 3 * cop] = 0
                    self.paramvec[layer_ind, uc_ind, 3 * cop + offset] = 0

    def get_zeroed_params(self):
        """Return parameter coordinates that are forced to zero by the ansatz configuration.

        Coordinates are returned as (layer_ind, uc_ind, param_ind) and are later used by
        Config2DBase.enforce_parameter_conditions to zero the corresponding entries in-place.
        Both colors share the same parameters, so the coordinates carry no color index.
        NB: the paramvec order differs from Z2System2D_Config, which groups all t, then all
        y, then all z; here the params are grouped per copy (see _create_symbolvec).

        Pure-gauge layers:
            All t_c parameters are zeroed. These parameters couple the physical mode to the
            virtual modes, so they are not part of a pure-gauge layer.

        Fermionic layers with U(1) symmetry enforced:
            - even-labelled t parameters are zeroed;
            - all intra-copy y_c and z_c parameters are zeroed;
            - mixed-copy couplings are zeroed when the two copies have the same parity.

        If self.u1_symmetry is false, the fermionic-layer U(1) zeroing rules are not applied.
        Pure-gauge zeroing is always applied.
        """
        offset = self._nparams // 2  # offset to get index of imaginary part
        zeroed_params = []

        # Pure-gauge layers: all t params, real and imaginary parts.
        for layer_ind in range(self.num_pg_layer):
            for uc_ind in range(self.unitcell_size):
                for cop in range(self.ncopy):
                    zeroed_params.append((layer_ind, uc_ind, 3 * cop))
                    zeroed_params.append((layer_ind, uc_ind, 3 * cop + offset))

        # Fermionic layers: collect the real-part indices, imaginary twins added below.
        zero_for_fermionic_layer = []
        for cop in range(self.ncopy):
            if cop % 2 == 1:
                zero_for_fermionic_layer.append(3 * cop)  # t of even copy label (1-based)
            zero_for_fermionic_layer += [3 * cop + 1, 3 * cop + 2]  # y, z
        pair_start = 3 * self.ncopy
        for cop1 in range(self.ncopy):
            for cop2 in range(cop1 + 1, self.ncopy):
                if cop1 % 2 == cop2 % 2:
                    zero_for_fermionic_layer += list(range(pair_start, pair_start + 4))  # a, b, c, d
                pair_start += 4
        if self.u1_symmetry:
            for layer_ind in range(self.num_pg_layer, self.nlayer):
                for uc_ind in range(self.unitcell_size):
                    for ind in zero_for_fermionic_layer:
                        zeroed_params.append((layer_ind, uc_ind, ind))
                        zeroed_params.append((layer_ind, uc_ind, ind + offset))

        return tuple(zeroed_params)

    def _create_symbolvec(self) -> list[sympy.Symbol]:
        """Define all symbols of the T matrix as symbols.
        We will use the analytic expression of the T matrix to calculate the derivative of the
        covariance matrices analytically.

        The order of the symbols is a [t_c, y_c, z_c] triple per copy (t couples the physical
        and virtual modes, y and z couple a virtual copy to itself), then an
        [a_pq, b_pq, c_pq, d_pq] quadruple per copy pair (couplings between two virtual
        copies), pairs ordered (1,2), (1,3), ..., (2,3), ... .
        All real parts, then all imaginary parts. Copy numbering starts at 1.
        Both colors share the same parameters, so there is no color index.

        Returns:
            list: List of all analytic symbols
        """
        names = [f"{p}{cop}" for cop in range(1, self.ncopy + 1) for p in ("t", "y", "z")]
        names += [
            f"{p}{cop1}{cop2}"
            for cop1 in range(1, self.ncopy + 1)
            for cop2 in range(cop1 + 1, self.ncopy + 1)
            for p in ("a", "b", "c", "d")
        ]
        reals = [sympy.Symbol(name + "r", real=True) for name in names]
        imags = [sympy.Symbol(name + "i", real=True) for name in names]
        return reals + imags

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

        The mode order (ncopy=2 example):
            Psi_1,Psi_2, l1_1, r1_1,d1_1,u1_1, l2_1, r2_1,d2_1,u2_1, l1_2, r1_2,d1_2,u1_2,l2_2, r2_2,d2_2,u2_2
        Where the modes are labelled by direction{copy}_{color}: physical modes of all colors first,
        then per color, per copy: l, r, d, u.
        The order {l,r,d,u} instead of {r,u,l,d} (used in some analytic calculations) because it
        eliminates the need for a lot of permutation matrices in the conversion from T to gamma_maj.
        The permutation matrices are prone to errors.

        Returns:
            sympy.Matrix: Analytic T matrix of the fiducial state
        """
        # Complex parameters: pair each real symbol with its imaginary partner.
        offset = self._nparams // 2
        keys = [str(symb)[:-1] for symb in self.symbolvec[:offset]]
        vals = [self.symbolvec[i] + 1.0j * self.symbolvec[i + offset] for i in range(offset)]
        params = dict(zip(keys, vals))

        # Physical-virtual row for one copy (t couplings).
        Block_1 = sympy.Matrix([[-1.0j * params["t1"], 1.0j * params["t1"], params["t1"], -params["t1"]]])

        # Virtual-virtual within one copy (y, z couplings).
        Block_2 = sympy.Matrix(
            [
                [0, 1.0j * params["y1"], params["z1"], 1.0j * params["z1"]],
                [-1.0j * params["y1"], 0, -1.0j * params["z1"], -params["z1"]],
                [-params["z1"], 1.0j * params["z1"], 0, -params["y1"]],
                [-1.0j * params["z1"], params["z1"], params["y1"], 0],
            ]
        )

        t_blocks = [Block_1.subs(params["t1"], params[f"t{cop}"]) for cop in range(1, self.ncopy + 1)]
        yz_blocks = [
            Block_2.subs([(params["y1"], params[f"y{cop}"]), (params["z1"], params[f"z{cop}"])])
            for cop in range(1, self.ncopy + 1)
        ]

        # Virtual-virtual between two copies (a, b, c, d couplings); none for ncopy == 1.
        abcd_blocks = {}
        if self.ncopy > 1:
            Block_3 = sympy.Matrix(
                [
                    [-1.0j * params["a12"], -1.0j * params["c12"], -1.0j * params["b12"], -1.0j * params["d12"]],
                    [1.0j * params["c12"], 1.0j * params["a12"], 1.0j * params["d12"], 1.0j * params["b12"]],
                    [params["d12"], params["b12"], params["a12"], params["c12"]],
                    [-params["b12"], -params["d12"], -params["c12"], -params["a12"]],
                ]
            )
            for cop1 in range(self.ncopy):
                for cop2 in range(cop1 + 1, self.ncopy):
                    subs = [
                        (params["a12"], params[f"a{cop1 + 1}{cop2 + 1}"]),
                        (params["b12"], params[f"b{cop1 + 1}{cop2 + 1}"]),
                        (params["c12"], params[f"c{cop1 + 1}{cop2 + 1}"]),
                        (params["d12"], params[f"d{cop1 + 1}{cop2 + 1}"]),
                    ]
                    abcd_blocks[(cop1, cop2)] = Block_3.subs(subs)

        # Single-color block: physical row, then one row of 4-mode blocks per copy.
        # Both colors have the same structure as well as parameters.
        rows = [sympy.Matrix.hstack(sympy.zeros(1), *t_blocks)]
        for cop in range(self.ncopy):
            off_diag_below = [-abcd_blocks[(other, cop)].T for other in range(cop)]
            off_diag_above = [abcd_blocks[(cop, other)] for other in range(cop + 1, self.ncopy)]
            rows.append(sympy.Matrix.hstack(-t_blocks[cop].T, *off_diag_below, yz_blocks[cop], *off_diag_above))
        tmat_symb_block = sympy.Matrix.vstack(*rows)

        # Block diagonal over the colors, then reorder to the color-outer mode order.
        # Copy labels run over colors: copies of color m are (m-1)*ncopy + 1 .. m*ncopy.
        tmat_symb = sympy.Matrix(sympy.BlockDiagMatrix(*[tmat_symb_block] * self.ncolors))

        wrong_order = []
        correct_order = [f"Psi{m}" for m in range(1, self.ncolors + 1)]
        for m in range(1, self.ncolors + 1):
            link_labels = [
                f"{d}_{(m - 1) * self.ncopy + cop}" for cop in range(1, self.ncopy + 1) for d in ("l", "r", "d", "u")
            ]
            wrong_order += [f"Psi{m}"] + link_labels
            correct_order += link_labels
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
        The mode order (ncopy=2 example) is
        {l1_1_1,l1_2_1,r1_1_1,r1_2_1,l2_1_1,l2_2_1,r2_1_1,r2_2_1,l1_1_2,l1_2_2,r1_1_2,r1_2_2,l2_1_2,l2_2_2,r2_1_2,r2_2_2}
        /{d1_1_1,d1_2_1,u1_1_1,u1_2_1,d2_1_1,d2_2_1,u2_1_1,u2_2_1,d1_1_2,d1_2_2,u1_1_2,u1_2_2,d2_1_2,d2_2_2,u2_1_2,u2_2_2}.
        The naming convention here is <mode letter><number of copy>_<majorana mode>_<color>.
        We order first by link and then by copy.
        The sites are picked such that the left mode is right of the right modes,
        i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        This function returns the covariance matrices for ungauged projectors:
        Unlike the Z2, copies are always coupled to themselves without mixing different copies.
        Here we work in the same convention as Z2 w=exp(rl)and not w=exp(lr) as in some references.

        This method overwrites an abstract method in System2DBase.

        Returns:
            list[xnp.ndarray]: Covariance matrices of the ungauged projector on a single link
        """

        # 2 is for 2D lattice
        dest_unmixed = [0] * 2  # does not mix copies

        # Single-copy 4x4 blocks in the mode order {l_1, l_2, r_1, r_2} / {d_1, d_2, u_1, u_2}.
        gamma_x_base = np.real_if_close(1.0j * np.kron(utils.pauliy, utils.paulix))
        gamma_y_base = np.real_if_close(1.0j * np.kron(utils.pauliy, utils.pauliz))

        # Identity in copy space (no copy mixing); a fermionic-specific projector would use
        # a different copy-space matrix here.
        unmixed_copy = make_sigma_matrix(self.ncopy, mix_copies=False)
        block_x = np.real_if_close(np.kron(unmixed_copy, gamma_x_base))  # single-color block
        block_y = np.real_if_close(np.kron(unmixed_copy, gamma_y_base))
        dest_unmixed[Direction.X] = block_diag(*[block_x] * self.ncolors)
        dest_unmixed[Direction.Y] = block_diag(*[block_y] * self.ncolors)

        return np.array([dest_unmixed] * self.num_pg_layer + [dest_unmixed] * self.num_fermionic_layer)
