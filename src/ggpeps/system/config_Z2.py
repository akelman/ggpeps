import sympy
import logging
import numpy as np

import ggpeps
from ggpeps import utils, gauge
from ggpeps.lattice import Direction

from .config_base import Config2DBase, generate_gauged_projector_terms, make_sigma_matrix

logger = logging.getLogger(ggpeps.LOGGER_NAME)


###################### Generic Z2System2D Config ##########################


class Z2System2D_Config(Config2DBase):
    """Generic configuration of the Z2 system in 2D.

    This config describes a Z2 lattice gauge theory variational ansatz with
    ``ncopy`` copies of virtual fermions on each link.

    Supported copy numbers:
        The current gauged-projector construction supports ``ncopy == 1`` and
        even values of ``ncopy``. Odd values larger than 1 are not supported,
        because pure-gauge projectors mix copies pairwise through
        ``sigma = (1 <-> 2), (3 <-> 4), ...``.

    Parameter-vector convention:
        The parameter vector stores all real parts first and all imaginary
        parts second. For ``ncopy = n`` the symbolic order is:

        real part:
            t1r, ..., tnr,
            y1r, ..., ynr,
            z1r, ..., znr,
            a12r, b12r, c12r, d12r,
            a13r, b13r, c13r, d13r,
            ...,
            a(n-1)nr, b(n-1)nr, c(n-1)nr, d(n-1)nr

        imaginary part:
            t1i, ..., tni,
            y1i, ..., yni,
            z1i, ..., zni,
            a12i, b12i, c12i, d12i,
            a13i, b13i, c13i, d13i,
            ...,
            a(n-1)ni, b(n-1)ni, c(n-1)ni, d(n-1)ni

    Meaning of parameter families:
        - ``t_i`` couples the physical mode to virtual copy ``i``.
        - ``y_i`` and ``z_i`` couple virtual modes within copy ``i``.
        - ``a_ij``, ``b_ij``, ``c_ij`` and ``d_ij`` couple virtual copies
          ``i`` and ``j``.

    Mode order of ``tmat``:
        ``{p, l1, r1, d1, u1, l2, r2, d2, u2, ..., ln, rn, dn, un}``.

    Mode order of ``gamma_dirac``:
        Annihilation operators first, followed by creation operators, using the
        same copy ordering as ``tmat``:
        ``{p, l1, r1, d1, u1, ..., ln, rn, dn, un,
        p_dag, l1_dag, r1_dag, d1_dag, u1_dag, ..., ln_dag, rn_dag, dn_dag, un_dag}``.

    Mode order of ``gamma_maj``:
        Each Dirac mode is expanded into two Majorana modes in the same order:
        ``{p_1, p_2, l1_1, l1_2, r1_1, r1_2, d1_1, d1_2, u1_1, u1_2,
        ..., ln_1, ln_2, rn_1, rn_2, dn_1, dn_2, un_1, un_2}``.
    """

    def __init__(
        self,
        lattice,
        g_el,
        g_mag,
        g_int,
        g_mass,
        g_chem,
        ncopy=4,
        num_pg_layer=1,
        num_fermionic_layer=1,
        mod_link_inds=(0,),
        unitcell_size=1,
        enforce_u1_symmetry=True,
    ) -> None:
        """Initialize the generic Z2 2D config.

        Args:
            lattice: Two-dimensional lattice used by the system.
            g_el: Electric-energy coupling.
            g_mag: Magnetic-energy coupling.
            g_int: Gauge-matter interaction coupling.
            g_mass: Fermion mass coupling.
            g_chem: Chemical potential values for the fermionic layers. If
                ``None``, all chemical potentials are set to zero by the base
                class.
            ncopy: Number of virtual fermion copies per link. Supported values
                are ``1`` and even integers.
            num_pg_layer: Number of pure-gauge PEPS layers.
            num_fermionic_layer: Number of fermionic PEPS layers.
            mod_link_inds: Link indices on which electric-energy terms are
                measured.
            unitcell_size: Translation-invariance unit-cell convention. Allowed
                values are ``1`` for full translation invariance, ``2`` for a
                two-site checkerboard unit cell, and ``-1`` for independent
                parameters on every site.
            enforce_u1_symmetry: Whether to enforce the U(1)-symmetric
                fermionic-layer parameter constraints.
        """
        # The current gauged-projector construction mixes copies pairwise in pure-gauge layers
        # via sigma = (1 <-> 2), (3 <-> 4), ... Therefore odd ncopy > 1 is not supported
        # unless a different projector convention is implemented. The ncopy == 1 case is the
        # trivial unmixed case.
        if not (ncopy == 1 or ncopy % 2 == 0):
            raise ValueError("ncopy must be 1 or even.")

        self.ncopy = ncopy
        self.nvirtmodes_vertex = 4 * ncopy
        self.nvirtmodes_link = 2 * ncopy
        self.nphysmodes_site = 1
        self.ncolors = 1
        self._nparams = 2 * ncopy * (2 * ncopy + 1)
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
            logger.error("This ansatz only supports unitcell_size = 1, 2, or -1 (all sites independent). \
                This can be adapted by adding in a specification in the config to map sites to parameters.")
            raise ValueError("Invalid unitcell_size.")

        self.init_el_energy_terms()

    def init_el_energy_terms(self) -> None:
        """Build the electric-energy term data structures for this ansatz.

        The electric-energy contribution is evaluated by expanding the gauged
        projectors into Majorana monomials. We precompute the resulting index,
        coefficient and constant structures separately for:

        - pure-gauge layers, where projectors mix copies pairwise;
        - fermionic layers, where projectors do not mix copies in order to
          preserve the U(1)-symmetric convention;
        - horizontal and vertical links;
        - even and odd sublattice parity, since the faithful representation is
          conjugated on odd sites.

        The raw per-group-element structures (grouped Majorana-index tuples, matching
        coefficients, and scalar constant terms -- for Z2 the constants are expected to be
        zero) are handed to set_el_energy_terms, which keeps the constants and the derived
        unique-basis form used in evaluation.
        """
        idx_vec = []
        coeffs_vec = []
        constants_vec = []

        for group_element in self.gaugemgr.group_elements_for_el_energy:
            # Pure Gauge (PG)
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

            # generate fermionic terms
            idxarr_ferm_h_0, const_ferm_h_0 = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.X, group_element, site=0
            )
            idxarr_ferm_h_1, const_ferm_h_1 = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.X, group_element, site=1
            )
            idxarr_ferm_v_0, const_ferm_v_0 = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.Y, group_element, site=0
            )
            idxarr_ferm_v_1, const_ferm_v_1 = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.Y, group_element, site=1
            )

            pg_link_coeffs, pg_link_indices = [], []
            ferm_link_coeffs, ferm_link_indices = [], []
            pg_link_constants, ferm_link_constants = [], []

            for mod_link in self.mod_link_inds:
                coord, dir = self.lattice.ind2coord_dir(mod_link)
                site_parity = sum(coord) % 2
                is_vertical = dir == Direction.Y

                # Select the correct base terms based on direction and parity
                if is_vertical:
                    if site_parity == 0:
                        term_pg = idxarr_pg_v_0
                        term_ferm = idxarr_ferm_v_0
                        constant_pg = const_pg_v_0
                        constant_ferm = const_ferm_v_0
                    else:
                        term_pg = idxarr_pg_v_1
                        term_ferm = idxarr_ferm_v_1
                        constant_pg = const_pg_v_1
                        constant_ferm = const_ferm_v_1
                else:
                    if site_parity == 0:
                        term_pg = idxarr_pg_h_0
                        term_ferm = idxarr_ferm_h_0
                        constant_pg = const_pg_h_0
                        constant_ferm = const_ferm_h_0
                    else:
                        term_pg = idxarr_pg_h_1
                        term_ferm = idxarr_ferm_h_1
                        constant_pg = const_pg_h_1
                        constant_ferm = const_ferm_h_1

                pg_c, pg_i = self._bucket_sort_terms(term_pg)
                pg_link_coeffs.append(pg_c)
                pg_link_indices.append(pg_i)
                pg_link_constants.append(constant_pg)

                ferm_c, ferm_i = self._bucket_sort_terms(term_ferm)
                ferm_link_coeffs.append(ferm_c)
                ferm_link_indices.append(ferm_i)
                ferm_link_constants.append(constant_ferm)

            pg_base_constants = tuple(pg_link_constants)
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

    def make_pure_gauge(self) -> None:
        """Make the ansatz pure gauge by setting all t_i parameters to zero.

        The generic symbol order stores all real t_i parameters first, followed
        by the remaining real parameters, and then the corresponding imaginary
        parameters. Therefore the real t_i indices are 0, ..., ncopy - 1, and
        the imaginary t_i indices are offset, ..., offset + ncopy - 1.

        This method is kept for compatibility with tests and workflows that
        explicitly project an ansatz to the pure-gauge sector after assigning a
        parameter vector.
        """
        offset = self._nparams // 2
        t_indices = list(range(self.ncopy)) + [ind + offset for ind in range(self.ncopy)]

        for layer_ind in range(self.nlayer):
            for uc_ind in range(self.unitcell_size):
                for t_ind in t_indices:
                    coord = (layer_ind, uc_ind, t_ind)
                    self.paramvec[coord] = 0

    def get_zeroed_params(self) -> tuple[tuple[int, int, int], ...]:
        """Return parameter coordinates that are forced to zero by the ansatz configuration.

        Coordinates are returned as ``(layer_ind, uc_ind, param_ind)`` and are
        later used by ``Config2DBase.enforce_parameter_conditions`` to zero the
        corresponding entries in-place.

        Pure-gauge layers:
            All ``t_i`` parameters are zeroed. These parameters couple the
            physical mode to the virtual modes, so they are not part of a
            pure-gauge layer.

        Fermionic layers with U(1) symmetry enforced:
            The ansatz keeps only the U(1)-compatible subset of parameters:

            - even-labelled ``t`` parameters are zeroed;
            - all intra-copy ``y_i`` and ``z_i`` parameters are zeroed;
            - mixed-copy couplings are zeroed when the two copies have the same
              parity.

            In zero-based indexing this means that ``t_inds`` contains indices
            ``1, 3, 5, ...``, corresponding to copy labels ``2, 4, 6, ...``.

        If ``self.u1_symmetry`` is false, the fermionic-layer U(1) zeroing rules
        are not applied. Pure-gauge zeroing is always applied.
        """
        offset = self._nparams // 2  # offset to get index of imaginary part
        zeroed_params = []  # we'll save the indices of the zeroed parameters

        def add_real_imag_zeroed(layer_ind: int, uc_ind: int, ind: int) -> None:
            """Zero both real and imaginary components of one complex parameter."""
            zeroed_params.append((layer_ind, uc_ind, ind))
            zeroed_params.append((layer_ind, uc_ind, ind + offset))

        # Zero out the parameters which are not used in the pure gauge layers
        for layer_ind in range(self.num_pg_layer):
            for uc_ind in range(self.unitcell_size):
                for t_ind in range(self.ncopy):
                    add_real_imag_zeroed(layer_ind, uc_ind, t_ind)

        # Zero out the parameters which are not used in the fermionic layers.
        # Under the U(1)-symmetric convention, only odd-labelled copies couple directly
        # to the physical fermion via t_i. Hence, t_2, t_4, ... are forced to zero.
        # The y_i and z_i intra-copy virtual couplings are also forced to zero, and
        # mixed-copy couplings are allowed only between copies of opposite parity.
        t_inds = [ind for ind in range(self.ncopy) if ind % 2 == 1]
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

        zero_for_fermionic_layer = t_inds + y_inds + z_inds + mixed_copy_inds
        if self.u1_symmetry:
            for layer_ind in range(self.num_pg_layer, self.nlayer):
                for uc_ind in range(self.unitcell_size):
                    for ind in zero_for_fermionic_layer:
                        add_real_imag_zeroed(layer_ind, uc_ind, ind)

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

        Returns:
            list: List of all analytic symbols
        """

        # t params: couple physical to virtual modes
        t_params = []
        for cop in range(1, self.ncopy + 1):
            for com in ["r", "i"]:  # real or imaginary component
                symbol = sympy.Symbol(f"t{cop}{com}", real=True)
                t_params.append(symbol)

        # y,z params: couple a virtual copy to itself
        y_params = []
        z_params = []
        for cop in range(1, self.ncopy + 1):
            for com in ["r", "i"]:  # real or imaginary component
                symbol = sympy.Symbol(f"y{cop}{com}", real=True)
                y_params.append(symbol)

                symbol = sympy.Symbol(f"z{cop}{com}", real=True)
                z_params.append(symbol)

        # a,b,c,d params: couple a virtual copy to another virtual copy
        mixed_params = []
        for cop1 in range(1, self.ncopy + 1):
            for cop2 in range(cop1 + 1, self.ncopy + 1):
                for param in ["a", "b", "c", "d"]:
                    for com in ["r", "i"]:  # real or imaginary component
                        symbol = sympy.Symbol(f"{param}{cop1}{cop2}{com}", real=True)
                        mixed_params.append(symbol)

        # Package all parameters, and ensure correct order
        all_params = t_params + y_params + z_params + mixed_params
        real_params = all_params[::2]
        imag_params = all_params[1::2]

        return real_params + imag_params

    @property
    def tmat_symb(self):
        """Construct the symbolic fiducial-state T matrix.

        The T matrix encodes the Gaussian fiducial state before gauging. Its
        block structure is fixed by the analytic ansatz constraints, including
        lattice rotations, charge conjugation and the copy structure of the
        virtual modes. The entries are symbolic so that derivatives of the
        covariance matrices can be computed analytically with SymPy.

        The matrix is assembled block-wise for arbitrary supported ``ncopy``:

        - one physical-virtual block for each copy, controlled by ``t_i``;
        - one intra-copy virtual-virtual block for each copy, controlled by
          ``y_i`` and ``z_i``;
        - one inter-copy virtual-virtual block for each pair of copies,
          controlled by ``a_ij``, ``b_ij``, ``c_ij`` and ``d_ij``.

        The mode order is:
            ``Psi, l_1, r_1, d_1, u_1, l_2, r_2, d_2, u_2, ..., l_n, r_n, d_n, u_n``.

        The order ``{l, r, d, u}`` is used instead of ``{r, u, l, d}``, which
        appears in some analytic calculations, because it avoids extra
        permutation matrices when converting from ``T`` to ``gamma_maj``.

        Returns:
            sympy.Matrix: Symbolic T matrix of the fiducial state.
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

        # Generate all the blocks for all copies:
        # a) Physical-virtual blocks
        t_blocks = [Block_1.subs(params["t1"], params[f"t{i}"]) for i in range(1, self.ncopy + 1)]

        # b) Virtual-virtual within the same copy
        yz_blocks = [
            Block_2.subs([(params["y1"], params[f"y{i}"]), (params["z1"], params[f"z{i}"])])
            for i in range(1, self.ncopy + 1)
        ]

        # c) Virtual-virtual between two different copies.
        # For ncopy == 1 there are no mixed-copy blocks and therefore no a12,b12,c12,d12 symbols.
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
                    cop1_label = cop1 + 1
                    cop2_label = cop2 + 1
                    subs = [
                        (params["a12"], params[f"a{cop1_label}{cop2_label}"]),
                        (params["b12"], params[f"b{cop1_label}{cop2_label}"]),
                        (params["c12"], params[f"c{cop1_label}{cop2_label}"]),
                        (params["d12"], params[f"d{cop1_label}{cop2_label}"]),
                    ]
                    abcd_blocks[(cop1, cop2)] = Block_3.subs(subs)

        # We will construct the T matrix row by row.
        # Each row will be a physical mode, or a virtual copy (4 virtual modes)

        # row corresponding to the physical mode
        data = [sympy.zeros(1)] + [t_block for t_block in t_blocks]
        first_row = sympy.Matrix.hstack(*data)

        # rows corresponding to the virtual modes
        rows = []
        for cop in range(self.ncopy):
            # We construct the row (really, 4 rows, since each copy has 4 modes, 1 for each link connected to a site)

            # mix copies - below diagonal blocks
            off_diag1 = [-abcd_blocks[(other, cop)].T for other in range(cop)]

            # mix copies - above diagonal blocks
            off_diag2 = [abcd_blocks[(cop, other)] for other in range(cop + 1, self.ncopy)]

            # construct full row
            row = [-t_blocks[cop].T] + off_diag1 + [yz_blocks[cop]] + off_diag2
            rows.append(sympy.Matrix.hstack(*row))

        # Stack all rows together
        all_rows = [first_row] + rows
        tmat_symb = sympy.Matrix.vstack(*all_rows)

        return tmat_symb

    def generate_gamma_gauge_neutral_dict(self):
        """Generate covariance matrices for the ungauged link projectors.

        The returned matrices describe the ungauged projector placed on a single
        link before inserting the gauge-field action. Two projector conventions
        are needed:

        - mixed-copy projectors for pure-gauge layers, using the pairwise copy
          permutation ``sigma = (1 <-> 2), (3 <-> 4), ...)``;
        - unmixed-copy projectors for fermionic layers, preserving the copy
          labels and hence the U(1)-symmetric convention.

        For ``ncopy == 1`` both conventions reduce to the same 1-copy projector,
        since the mixed and unmixed copy-space matrices are both the identity.

        Mode order for horizontal links:
            ``{l1_1, l1_2, r1_1, r1_2, ..., ln_1, ln_2, rn_1, rn_2}``.

        Mode order for vertical links:
            ``{d1_1, d1_2, u1_1, u1_2, ..., dn_1, dn_2, un_1, un_2}``.

        The naming convention is ``<mode letter><copy index>_<Majorana index>``.
        For example, ``r3_2`` denotes the second Majorana mode of the right
        virtual mode in copy 3.

        The paired modes are chosen to live on the same physical link with
        opposite orientations: left/right for horizontal links and down/up for
        vertical links.

        Returns:
            list[np.ndarray]: One covariance-matrix dictionary entry per PEPS
            layer, with pure-gauge entries first and fermionic entries second.
        """

        dest_mixed = [0] * 2  # mixes copies
        dest_unmixed = [0] * 2  # does not mix copies

        gamma_x_base = np.real_if_close(1.0j * np.kron(utils.pauliy, utils.paulix))
        gamma_y_base = np.real_if_close(1.0j * np.kron(utils.pauliy, utils.pauliz))

        mixed_copy = make_sigma_matrix(self.ncopy, mix_copies=True)
        unmixed_copy = make_sigma_matrix(self.ncopy, mix_copies=False)

        # We want to give the projectors for the pure gauge part, which mix copies.
        # For ncopy == 1, make_sigma_matrix returns the 1x1 identity matrix, so mixed
        # and unmixed projectors coincide with the original one-copy projector.
        dest_mixed[Direction.X] = np.real_if_close(np.kron(mixed_copy, gamma_x_base))
        dest_mixed[Direction.Y] = np.real_if_close(np.kron(mixed_copy, gamma_y_base))

        # We want to give the projectors for the fermionic part which don't mix copies
        # (so as to preserve global U(1) symmetry).
        dest_unmixed[Direction.X] = np.real_if_close(np.kron(unmixed_copy, gamma_x_base))
        dest_unmixed[Direction.Y] = np.real_if_close(np.kron(unmixed_copy, gamma_y_base))

        return [dest_mixed] * self.num_pg_layer + [dest_unmixed] * self.num_fermionic_layer
