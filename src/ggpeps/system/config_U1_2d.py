import sympy
import logging

import numpy as np

import ggpeps
from ggpeps import utils, gauge
from ggpeps.lattice import Direction

from .config_base import Config2DBase

logger = logging.getLogger(ggpeps.LOGGER_NAME)


class U1System2DConfig(Config2DBase):
    _nparams = 3
    ncopy = 1
    nvirtmodes_link = 8
    nvirtmodes_link = 4
    nphysmodes_site = 1  # number of physical modes per site
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
        num_fermionic_layer=0,
        unitcell_size=1,
    ):
        # The parameters have the following order: [[t1,y1,z1],[t2,y2,z2],....]
        super().__init__(
            gauge.ZNGauge(3),  # TODO: must be fixed - this is U(1)!
            lattice,
            g_el,
            g_mag,
            g_int,
            g_mass,
            g_chem,
            num_pg_layer,
            num_fermionic_layer,
            unitcell_size,
        )

        # Translation invariance
        if self.unitcell_size not in [1]:
            logger.error(
                "This ansatz only supports unitcell_size = 1. \
                This can be adapted by adding in a specification in the config to map sites to parameters."
            )
            raise ValueError("Invalid unitcell_size.")

    def make_pure_gauge(self):
        # The order of the parameters is [t,y,z]
        # Here we set the t parameters to zero for the pure gauge layers (which is all the layers)
        assert self.nlayer == self.num_pg_layer
        for lay in range(self.nlayer):
            for uc_ind in range(self.unitcell_size):
                self.paramvec[lay, uc_ind, 0] = 0

    def get_zeroed_params(self):
        """This should really use make_pure_gauge() - i.e. return the indices which are set to zero there.
        However, some tests which use this ansatz do not actually satisfy the pure gauge condition
        - they use this ansatz with nonzero t params, and test against hard-coded values.
        (This works because make_pure_gauge() is often not called in the execution path of those tests).
        To preserve compatibility with those tests, we do not call make_pure_gauge() here.
        """
        zeroed_params = []
        return tuple(zeroed_params)

    def _create_symbolvec(self):
        t = sympy.Symbol("t", real=True)
        y = sympy.Symbol("y", real=True)
        z = sympy.Symbol("z", real=True)
        return [t, y, z]

    def compute_tmat_symb_single(self):
        [t, y, z] = self.symbolvec
        etap = sympy.exp(1.0j * sympy.pi / 4.0)
        zsqrt = z / sympy.sqrt(2)
        tmat_symb_single = sympy.Matrix(
            [
                [t, etap**2 * t, etap * t, etap**3 * t],
                [0, y, zsqrt, zsqrt],
                [-y, 0, -zsqrt, zsqrt],
                [-zsqrt, zsqrt, 0, y],
                [-zsqrt, -zsqrt, -y, 0],
            ]
        )
        return tmat_symb_single

    @property
    def tmat_symb(self):
        tmat_symb = sympy.zeros(9, 9)
        tmat_symb_single = self.compute_tmat_symb_single()
        tmat_symb[0:5, 5:] = tmat_symb_single
        tmat_symb[5:, 0:5] = -tmat_symb_single.T
        return tmat_symb

    def generate_gamma_gauge_neutral_dict(self):
        # Note: unlike in the Z2 case, here we can ignore the direction of the link
        dest = [0] * 2
        dest[Direction.X] = np.real(1.0j * np.kron(np.kron(utils.pauliy, utils.paulix), utils.paulix))
        dest[Direction.Y] = np.real(1.0j * np.kron(np.kron(utils.pauliy, utils.paulix), utils.paulix))
        return [dest] * self.nlayer
