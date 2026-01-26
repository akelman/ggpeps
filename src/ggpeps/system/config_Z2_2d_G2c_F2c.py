import sympy
import logging

import numpy as np

import ggpeps
from ggpeps import utils, gauge
from ggpeps.lattice import Direction

from .config_base import Config2DBase, generate_gauged_projector_terms


logger = logging.getLogger(ggpeps.LOGGER_NAME)


class Z2System2D_G2C_F2C_Config(Config2DBase):
    """Configuration of the Z2 system in 2D with 2 copies of virtual fermions on the links per layer.
    Each layer can either be pure-gauge (in which case the t-params are zeroed out),
    or fermionic (in which case the y,z-params are zeroed out).

    Some general notes about conventions:

    Order of the paramvec: [t1r,y1r,z1r,t2r,y2r,z2r,ar,br,cr,dr,t1i,y1i,z1i,t2i,y2i,z2i,ai,bi,ci,di].
    Mode order of tmat: {p,l1,r1,d1,u1,l2,r2,d2,u2}.
    Mode order of gamma_dirac:
        {p,l1,r1,d1,u1,l2,r2,d2,u2,p_dag,l1_dag,r1_dag,u1_dag,d1_dag,l2_dat,r2_dag,u2_dag,d2_dag}.
    Mode order of gamma_maj:
        {p_1,p_2,l1_1,l1_2,r1_1,r1_2,d1_1,d1_2,u1_1,u1_2,l2_1,l2_2,r2_1,r2_2,d2_1,d2_2,u2_1,u2_2}.
    """

    _nparams = 20
    ncopy = 2
    nvirtmodes_vertex = 8
    nvirtmodes_link = 4
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
        """Build idxarr_vec (quad H0,H1,V0,V1 terms per layer)."""
        result = []
        for group_element in self.gaugemgr.group_elements_for_el_energy:
            # --- Pure Gauge Terms ---
            # 1. Horizontal, Site 0
            idxarr_lay_pg_h_0, _ = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, True, Direction.X, group_element, site=0
            )

            # 2. Horizontal, Site 1
            idxarr_lay_pg_h_1, _ = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, True, Direction.X, group_element, site=1
            )

            # 3. Vertical, Site 0
            idxarr_lay_pg_v_0, _ = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, True, Direction.Y, group_element, site=0
            )

            # 4. Vertical, Site 1
            idxarr_lay_pg_v_1, _ = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, True, Direction.Y, group_element, site=1
            )

            # --- Fermionic Terms ---
            # 1. Horizontal, Site 0
            idxarr_lay_pf_h_0, _ = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.X, group_element, site=0
            )

            # 2. Horizontal, Site 1
            idxarr_lay_pf_h_1, _ = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.X, group_element, site=1
            )

            # 3. Vertical, Site 0
            idxarr_lay_pf_v_0, _ = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.Y, group_element, site=0
            )

            # 4. Vertical, Site 1
            idxarr_lay_pf_v_1, _ = generate_gauged_projector_terms(
                self.ncopy, self.ncolors, False, Direction.Y, group_element, site=1
            )

            # Pair horizontal/vertical term-lists termwise for each layer kind
            # Structure: (H0, H1, V0, V1)
            zipped_pg = tuple(zip(idxarr_lay_pg_h_0, idxarr_lay_pg_h_1, idxarr_lay_pg_v_0, idxarr_lay_pg_v_1))
            zipped_pf = tuple(zip(idxarr_lay_pf_h_0, idxarr_lay_pf_h_1, idxarr_lay_pf_v_0, idxarr_lay_pf_v_1))

            result.append(tuple([zipped_pg] * self.num_pg_layer + [zipped_pf] * self.num_fermionic_layer))
            # Stack per-layer: first pure-gauge layers, then fermionic layers

        self.idxarr_vec = tuple(result)

    def make_pure_gauge(self) -> None:
        """Make the ansatz pure gauge by setting t-params to zero.

        This function is obsolete for this ansatz, and is kept for some tests.
        """
        t_indices = [0, 3, 10, 13]  # index of t1r, t2r, t1i, t2i in symbolvec
        for layer_ind in range(self.nlayer):
            for uc_ind in range(self.unitcell_size):
                for t_ind in t_indices:
                    coord = (layer_ind, uc_ind, t_ind)
                    self.paramvec[coord] = 0

    def get_zeroed_params(self) -> tuple[tuple[int, int, int], ...]:
        # The order of the parameters (for each layer) is:
        # [t1r,y1r,z1r,t2r,y2r,z2r,ar,br,cr,dr,t1i,y1i,z1i,t2i,y2i,z2i,ai,bi,ci,di]

        zeroed_params = []  # we'll save the indices of the zeroed parameters

        t_indices = [0, 3, 10, 13]  # index of t1r, t2r, t1i, t2i in symbolvec
        for layer_ind in range(self.num_pg_layer):
            for uc_ind in range(self.unitcell_size):
                for t_ind in t_indices:
                    coord = (layer_ind, uc_ind, t_ind)
                    zeroed_params.append(coord)

        if self.u1_symmetry:
            # index of t2r, t2i, y1r, z1r, y2r, z2r, y1i, z1i, y2i, z2i in symbolvec
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
        We will use the analytic expression of the T matrix to calculate the derivative
        of the covariance matrices analytically.

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
        return [t1r, y1r, z1r, t2r, y2r, z2r, ar, br, cr, dr, t1i, y1i, z1i, t2i, y2i, z2i, ai, bi, ci, di]

    @property
    def tmat_symb(self) -> sympy.Matrix:
        """Definition of the symbolic T matrix.
        The definition of T here is a result of an analytic consideration of global
        symmetries like rotational invariance, charge conjugation invarance, etc.
        The T matrix is given in terms of symbols to compute the derivative of the
        covariance matrices analytically via sympy.
        We do not have to type them explicitly anymore into the code.

        This is one of two analytic inputs into the code.
        The other input is the structure and the parametrization of the projectors.

        The mode order is: Psi, l_1, r_1, d_1, u_1, l_2, r_2, d_2, u_2

        The order {l,r,d,u} instead of {r,u,l,d} (used in some analytic calculations)
        because it eliminates the need for a lot of permutation matrices in the conversion from T to gamma_maj.
        The permutation matrices are prone to errors.

        Returns:
            sympy.Matrix: Analytic T matrix of the fiducial state
        """
        [t1r, y1r, z1r, t2r, y2r, z2r, ar, br, cr, dr, t1i, y1i, z1i, t2i, y2i, z2i, ai, bi, ci, di] = self.symbolvec
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
        return tmat_symb

    def generate_gamma_gauge_neutral_dict(self) -> np.ndarray:
        """Generate the covariance matrix of the ungauged projectors.
        The mode order is
            {l1_1, l1_2, r1_1, r1_2, l2_1, l2_2, r2_1, r2_2}
            or (for vertical links)
            {d1_1, d1_2, u1_1, u1_2, d2_1, d2_2, u2_1, u2_2}.
        The naming convention here is <mode letter><number of copy>_<majorana mode>.
        We order first by link and then by copy.
        The sites are picked such that the left mode is right of the right modes,
        i.e. they are sitting on the same link.
        The same is true for the for the up and down modes.

        This function returns two different covariance matrices for ungauged projectors:
        In the first, modes of copy 1 are coupled to modes of copy 2.
        In the second, the projectors don't mix copies (so as to preserve global U(1) symmetry).
        The first option is used for the pure-gauge layer, the second for the fermionic layer.

        We use Kronecker products to construct the covariance matrices concisely; the result
        is equivalent to hardcoding the matrices directly.

        This method overwrites an abstract method in Config2DBase.

        Returns:
            array: Covariance matrices of the ungauged projector on a single link
        """

        dest_mixed = []  # mixes copies
        dest_mixed.append(np.real(1.0j * np.kron(utils.paulix, np.kron(utils.pauliy, utils.paulix))))  # X direction
        dest_mixed.append(np.real(1.0j * np.kron(utils.paulix, np.kron(utils.pauliy, utils.pauliz))))  # Y direction

        dest_unmixed = []  # does not mix copies
        dest_unmixed.append(np.real(1.0j * np.kron(np.eye(2), np.kron(utils.pauliy, utils.paulix))))  # X direction
        dest_unmixed.append(np.real(1.0j * np.kron(np.eye(2), np.kron(utils.pauliy, utils.pauliz))))  # Y direction

        return np.array([dest_mixed] * self.num_pg_layer + [dest_unmixed] * self.num_fermionic_layer)
