import sympy
import logging
import numpy as np

import ggpeps
from ggpeps import utils, gauge
from ggpeps.lattice import Direction

from .config_base import Config2DBase, generate_gauged_projector_terms

logger = logging.getLogger(ggpeps.LOGGER_NAME)


###################### Z2System2D ##########################


class Z2System2D_G4C_F4C_Config(Config2DBase):
    """Configuration of the Z2 system in 2D with 4 copies of virtual fermions on the links.
    More details about the mode order and the parameters can be found in the documentation of `Z2System2D2C`.

    Some general notes about conventions:

    Order of the paramvec: see the functions that create the symbolvec.
    Mode order of tmat: {p,l1,r1,d1,u1,l2,r2,d2,u2,l3,r3,d3,u3,l4,r4,d4,u4}.
    Mode order of gamma_dirac:
        {p,l1,r1,d1,u1,l2,r2,d2,u2,p_dag,l1_dag,r1_dag,u1_dag,d1_dag,l2_dat,r2_dag,u2_dag,d2_dag,l3,r3... }.
    Mode order of gamma_maj:
        {p_1,p_2,l1_1,l1_2,r1_1,r1_2,d1_1,d1_2,u1_1,u1_2,l2_1,l2_2,r2_1,r2_2,d2_1,d2_2,u2_1,u2_2,l3_1,l3_2... }.
    """

    _nparams = 2 * (4 + 2 * 4 + 4 * 3 * 2)
    ncopy = 4
    nvirtmodes_vertex = 16
    nvirtmodes_link = 8
    nphysmodes_site = 1
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
        num_fermionic_layer=1,
        mod_link_inds=(0,),
        unitcell_size=1,
        enforce_u1_symmetry=True,
    ) -> None:
        super().__init__(
            gauge.ZNGauge(2),
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

        if self.unitcell_size not in [1, 2, -1]:
            logger.error(
                "This ansatz only supports unitcell_size = 1, 2, or -1 (all sites independent). \
                This can be adapted by adding in a specification in the config to map sites to parameters."
            )
            raise ValueError("Invalid unitcell_size.")

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

            # generate fermionic terms
            idxarr_ferm_h_0, _ = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.X, group_element, site=0
            )
            idxarr_ferm_h_1, _ = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.X, group_element, site=1
            )
            idxarr_ferm_v_0, _ = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.Y, group_element, site=0
            )
            idxarr_ferm_v_1, _ = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.Y, group_element, site=1
            )

            pg_link_coeffs, pg_link_indices = [], []
            ferm_link_coeffs, ferm_link_indices = [], []

            for link_pos in range(len(self.mod_link_inds)):
                coord, dir = self.lattice.ind2coord_dir(link_pos)
                site_pairity = sum(coord) % 2
                is_vertical = dir == Direction.Y

                # Select the correct base terms based on direction and parity
                if is_vertical:
                    if site_pairity == 0:
                        term_pg = idxarr_pg_v_0
                        term_ferm = idxarr_ferm_v_0
                    else:
                        term_pg = idxarr_pg_v_1
                        term_ferm = idxarr_ferm_v_1
                else:
                    if site_pairity == 0:
                        term_pg = idxarr_pg_h_0
                        term_ferm = idxarr_ferm_h_0
                    else:
                        term_pg = idxarr_pg_h_1
                        term_ferm = idxarr_ferm_h_1

                pg_c, pg_i = self._bucket_sort_terms(term_pg)
                pg_link_coeffs.append(pg_c)
                pg_link_indices.append(pg_i)

                ferm_c, ferm_i = self._bucket_sort_terms(term_ferm)
                ferm_link_coeffs.append(ferm_c)
                ferm_link_indices.append(ferm_i)

            pg_base_coeffs = tuple(pg_link_coeffs)
            ferm_base_coeffs = tuple(ferm_link_coeffs)

            # Stack layers: PG layers first, then Fermionic layers
            coeffs_vec.append((pg_base_coeffs,) * self.num_pg_layer + (ferm_base_coeffs,) * self.num_fermionic_layer)

            pg_base_indices = tuple(pg_link_indices)
            ferm_base_indices = tuple(ferm_link_indices)

            idx_vec.append((pg_base_indices,) * self.num_pg_layer + (ferm_base_indices,) * self.num_fermionic_layer)

        self.idx_vec = tuple(idx_vec)
        self.coeffs_vec = tuple(coeffs_vec)

    def get_zeroed_params(self) -> tuple[tuple[int, int, int], ...]:
        offset = self._nparams // 2  # offset to get index of imaginary part
        zeroed_params = []  # we'll save the indices of the zeroed parameters

        # Zero out the parameters which are not used in the pure gauge layers
        for layer_ind in range(self.num_pg_layer):
            for uc_ind in range(self.unitcell_size):
                for t_ind in range(self.ncopy):
                    real_coord = (layer_ind, uc_ind, t_ind)
                    imag_coord = (layer_ind, uc_ind, t_ind + offset)
                    zeroed_params.append(real_coord)
                    zeroed_params.append(imag_coord)

        # Zero out the parameters which are not used in the fermionic layers
        y_inds = [ind for ind in range(self.ncopy, 2 * self.ncopy)]
        z_inds = [ind for ind in range(2 * self.ncopy, 3 * self.ncopy)]
        mixed_copy_inds = []
        countdown = list(range(self.ncopy - 1, 0, -1))
        for cop1 in range(self.ncopy):
            for cop2 in range(cop1 + 1, self.ncopy):
                if (cop1 % 2) == (cop2 % 2):
                    start = 3 * self.ncopy + (sum(countdown[:cop1]) + cop2 - cop1 - 1) * 4
                    inds = [ind for ind in range(start, start + 4)]  # a,b,c,d
                    mixed_copy_inds += inds

        zero_for_fermionic_layer = y_inds + z_inds + mixed_copy_inds
        if self.u1_symmetry:
            for layer_ind in range(self.num_pg_layer, self.nlayer):
                for uc_ind in range(self.unitcell_size):
                    for ind in zero_for_fermionic_layer:
                        real_coord = (layer_ind, uc_ind, ind)
                        imag_coord = (layer_ind, uc_ind, ind + offset)
                        zeroed_params.append(real_coord)
                        zeroed_params.append(imag_coord)

        return tuple(zeroed_params)

    def _create_symbolvec(self):
        """Define all symbols of the T matrix as symbols.
        We will use the analytic expression of the T matrix to calculate the derivative
        of the covariance matrices analytically.

        The order of the symbols is:
        1) params which couple the physical and virtual modes
        2) params which couple a set of virtual modes to themselves
        3) params which couple between two sets of virtual modes
        All real parts, then all imaginary parts.
        Copy numbering starts at 1 (no zero-indexing).

        TODO: this function is general enough to be used in other systems;
        it can probably be moved to the base class.

        Returns:
            list: List of all analytic symbols
        """

        # t params: couple physical to virtual modes
        t_params = []
        for cop in range(1, self.ncopy + 1):
            for com in ["r", "i"]:  # real or imaginary
                symbol = sympy.Symbol(f"t{cop}{com}", real=True)
                t_params.append(symbol)

        # y,z params: couple a virtual copy to itself
        y_params = []
        z_params = []
        for cop in range(1, self.ncopy + 1):
            for com in ["r", "i"]:  # real or imaginary
                symbol = sympy.Symbol(f"y{cop}{com}", real=True)
                y_params.append(symbol)

                symbol = sympy.Symbol(f"z{cop}{com}", real=True)
                z_params.append(symbol)

        # a,b,c,d params: couple a virtual copy to another virtual copy
        mixed_params = []
        for cop1 in range(1, self.ncopy + 1):
            for cop2 in range(cop1 + 1, self.ncopy + 1):
                for param in ["a", "b", "c", "d"]:
                    for com in ["r", "i"]:  # real or imaginary
                        symbol = sympy.Symbol(f"{param}{cop1}{cop2}{com}", real=True)
                        mixed_params.append(symbol)

        # Package all parameters, and ensure correct order
        all_params = t_params + y_params + z_params + mixed_params
        real_params = all_params[::2]
        imag_params = all_params[1::2]

        return real_params + imag_params

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
        # Create a dictionary of parameters
        offset = self._nparams // 2  # offset to get index of imaginary part
        keys = [str(symb)[:-1] for symb in self.symbolvec[:offset]]
        vals = [self.symbolvec[i] + 1j * self.symbolvec[i + offset] for i in range(offset)]
        params = {key: val for key, val in zip(keys, vals)}

        # Define the form blocks of the T matrix -
        # physical-virtual, virtual-virtual within the same copy, and virtual-virtual between copies

        # a) Physical-virtual
        # this is a row matrix (because of the transpose)
        Block_1 = sympy.Matrix([-1.0j * params["t1"], 1.0j * params["t1"], params["t1"], -params["t1"]]).T

        # b) Virtual-virtual within the same copy
        Block_2 = sympy.Matrix(
            [
                [0, 1.0j * params["y1"], params["z1"], 1.0j * params["z1"]],
                [-1.0j * params["y1"], 0, -1.0j * params["z1"], -params["z1"]],
                [-params["z1"], 1.0j * params["z1"], 0, -params["y1"]],
                [-1.0j * params["z1"], params["z1"], params["y1"], 0],
            ]
        )

        # c) Virtual-virtual between two different copies
        Block_3 = sympy.Matrix(
            [
                [
                    -1.0j * params["a12"],
                    -1.0j * params["c12"],
                    -1.0j * params["b12"],
                    -1.0j * params["d12"],
                ],
                [
                    1.0j * params["c12"],
                    1.0j * params["a12"],
                    1.0j * params["d12"],
                    1.0j * params["b12"],
                ],
                [params["d12"], params["b12"], params["a12"], params["c12"]],
                [-params["b12"], -params["d12"], -params["c12"], -params["a12"]],
            ]
        )

        # Generate all the blocks for all copies:
        # a) Physical-virtual blocks
        t_blocks = [Block_1.subs(params["t1"], params[f"t{i}"]) for i in range(1, self.ncopy + 1)]

        # b) Virtual-virtual within the same copy
        yz_blocks = [
            Block_2.subs([(params["y1"], params[f"y{i}"]), (params["z1"], params[f"z{i}"])])
            for i in range(1, self.ncopy + 1)
        ]

        # c) Virtual-virtual between different copies
        abcd_blocks = []
        for cop1 in range(1, self.ncopy + 1):
            for cop2 in range(cop1 + 1, self.ncopy + 1):
                # a12, b12, c12, d12
                a_sub = (params["a12"], params[f"a{cop1}{cop2}"])
                b_sub = (params["b12"], params[f"b{cop1}{cop2}"])
                c_sub = (params["c12"], params[f"c{cop1}{cop2}"])
                d_sub = (params["d12"], params[f"d{cop1}{cop2}"])
                subs = [a_sub, b_sub, c_sub, d_sub]
                block = Block_3.subs(subs)
                abcd_blocks.append(block)

        # We will constuct the T matrix row by row.
        # Each row will be a physical mode, or a virtual copy (4 virtual modes)

        # row corresponding to the physical mode
        data = [sympy.zeros(1)] + [t_block for t_block in t_blocks]
        first_row = sympy.Matrix.hstack(*data)

        # rows corresponding to the virtual modes
        rows = []
        countdown = list(range(self.ncopy - 1, 0, -1))
        for cop in range(self.ncopy):
            # We construct the row (really, 4 rows, since each copy has 4 modes, 1 for each link connected to a site)

            # mix copies - below diagonal blocks
            # TODO: this should be generalized to any number of copies
            off_diag1 = []
            if cop == 0:
                off_diag1 = []
            elif cop == 1:
                off_diag1 = [-abcd_blocks[0].T]
            elif cop == 2:
                off_diag1 = [
                    -abcd_blocks[1].T,
                    -abcd_blocks[3].T,
                ]
            elif cop == 3:
                off_diag1 = [
                    -abcd_blocks[2].T,
                    -abcd_blocks[4].T,
                    -abcd_blocks[5].T,
                ]
            else:
                raise ValueError("Invalid copy index.")

            # mix copies - above diagonal blocks
            start = sum(countdown[:cop])
            end = sum(countdown[: cop + 1])
            off_diag2 = abcd_blocks[start:end]

            # construct full row
            row = [-t_blocks[cop].T] + off_diag1 + [yz_blocks[cop]] + off_diag2
            rows.append(sympy.Matrix.hstack(*row))

        # Stack all rows together
        all_rows = [first_row] + rows
        tmat_symb = sympy.Matrix.vstack(*all_rows)

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
            list[np.ndarray]: Covariance matrices of the ungauged projector on a single link
        """

        dest_mixed = [0] * 2  # mixes copies
        dest_unmixed = [0] * 2  # does not mix copies

        zeros_8 = np.zeros((8, 8))

        # We want to give the projectors for the pure gauge part, which mix copies
        mixed_X = np.real_if_close(1.0j * np.kron(utils.paulix, np.kron(utils.pauliy, utils.paulix)))
        mixed_Y = np.real_if_close(1.0j * np.kron(utils.paulix, np.kron(utils.pauliy, utils.pauliz)))

        dest_mixed[Direction.X] = np.block([[mixed_X, zeros_8], [zeros_8, mixed_X]])
        dest_mixed[Direction.Y] = np.block([[mixed_Y, zeros_8], [zeros_8, mixed_Y]])

        # We want to give the projectors for the fermionic part which don't mix copies
        # (so as to preserve global U(1) symmetry)
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

        dest_unmixed[Direction.X] = np.block([[unmixed_X, zeros_8], [zeros_8, unmixed_X]])
        dest_unmixed[Direction.Y] = np.block([[unmixed_Y, zeros_8], [zeros_8, unmixed_Y]])

        return [dest_mixed] * self.num_pg_layer + [dest_unmixed] * self.num_fermionic_layer
