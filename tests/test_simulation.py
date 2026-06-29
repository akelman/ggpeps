import unittest

import numpy as np

import ggpeps
from ggpeps import lattice
from ggpeps import system
from ggpeps import utils
from ggpeps.caching import Cache
from ggpeps.evaluator_manager import EvaluatorManager
from ggpeps.minimizer import Minimizer, MinimizerConfig
from ggpeps.exacteval import ExactEvaluatorConfig
from ggpeps.mc import MonteCarloEvaluatorConfig

from ggpeps.system.system_Z2 import Z2System2D


# Helper: legacy G2 order for 2-copy Z2 system, for test parameter conversion
G2_ORDER_IN_G4_CONVENTION = [
    "t1r",
    "y1r",
    "z1r",
    "t2r",
    "y2r",
    "z2r",
    "a12r",
    "b12r",
    "c12r",
    "d12r",
    "t1i",
    "y1i",
    "z1i",
    "t2i",
    "y2i",
    "z2i",
    "a12i",
    "b12i",
    "c12i",
    "d12i",
]


def reorder_g2_paramvec_to_g4_order(paramvec, cfg):
    """Convert legacy G2-ordered test parameters to generic G4(ncopy=2) order."""
    target_order = [str(symbol) for symbol in cfg.symbolvec]
    return utils.reorder_parameter_vector(paramvec, G2_ORDER_IN_G4_CONVENTION, target_order, axis=-1)


class TestZ2System(unittest.TestCase):
    """This class tests hard-coded values that should be output by simulations.
    We compare only a few simulations, with as general settings as possible."""

    def setUp(self):
        pass

    def test_minimizer_against_ed(self):
        """Run a full minimization and compare with ED. This test takes a while."""

        # System config
        # Couplings
        g = 1.1
        g_int = 0.6
        mass = 0.3
        chem = [0.6]

        # Ansatz settings
        L = 2
        num_pg_layer = 1
        num_fermionic_layer = 1
        nlayer = num_pg_layer + num_fermionic_layer
        unitcell_size = 2
        enforce_u1 = False
        el_links = [0, 1]  # links on which to compute the electric energy
        gauge_fixing = -1  # maximal tree

        # Construct the system
        lat = lattice.Lattice2D(L, L, gf_num_of_rows=gauge_fixing)

        system_cfg = utils.make_z2_2copy_config(
            lat,
            g / 2,
            1.0 / (2.0 * g),
            g_int,
            mass,
            chem,
            num_pg_layer=num_pg_layer,
            num_fermionic_layer=num_fermionic_layer,
            mod_link_inds=el_links,
            unitcell_size=unitcell_size,
            enforce_u1_symmetry=enforce_u1,
        )  # 2 copy ansatz

        # Set parameters - the comparisons below depend on the parameters
        seed = 5
        rngstate = np.random.RandomState(seed)
        shape = (nlayer, unitcell_size, 20)
        paramvec = rngstate.rand(*shape)
        paramvec = reorder_g2_paramvec_to_g4_order(paramvec, system_cfg)
        system_cfg.paramvec = paramvec

        # Minimzer config
        min_cfg = MinimizerConfig()
        min_cfg.method = "BFGS"
        min_cfg.max_iter = 50
        min_cfg.alpha = 0.1
        min_cfg.tol = 1e-5

        # Evaluator
        ec_config = ExactEvaluatorConfig()
        ec_config.compute_grads = True
        nrunner = 0
        ec_mgr = EvaluatorManager(Z2System2D, system_cfg, ec_config, nrunner)

        # Minimize
        ggpeps.global_vars["cache"] = Cache(disable_cache=True)
        minimizer = Minimizer(min_cfg, ec_mgr)
        result = minimizer.minimize()
        ed_val = 3.4408931193537455
        self.assertAlmostEqual(result.value, ed_val, places=1)

    def test_exacteval_against_previous_version(self):
        """Run a full ExactEval and compare with a well tested previous version of the code.
        We compare against commit 566db49."""

        # System config
        # Couplings
        g = 0.8
        g_int = 0.9
        mass = 1.3
        chem = [0.2]

        # Ansatz settings
        L = 2
        num_pg_layer = 1
        num_fermionic_layer = 1
        nlayer = num_pg_layer + num_fermionic_layer
        unitcell_size = 2
        enforce_u1 = False
        el_links = [0, 1]  # links on which to compute the electric energy
        gauge_fixing = -1  # maximal tree

        # Construct the system
        lat = lattice.Lattice2D(L, L, gf_num_of_rows=gauge_fixing)

        system_cfg = utils.make_z2_2copy_config(
            lat,
            g / 2,
            1.0 / (2.0 * g),
            g_int,
            mass,
            chem,
            num_pg_layer=num_pg_layer,
            num_fermionic_layer=num_fermionic_layer,
            mod_link_inds=el_links,
            unitcell_size=unitcell_size,
            enforce_u1_symmetry=enforce_u1,
        )  # 2 copy ansatz

        # Set parameters - the comparisons below depend on the parameters
        seed = 12
        rngstate = np.random.RandomState(seed)
        shape = (nlayer, unitcell_size, 20)
        paramvec = rngstate.rand(*shape)
        paramvec = reorder_g2_paramvec_to_g4_order(paramvec, system_cfg)
        system_cfg.paramvec = paramvec

        # Evaluator
        ec_config = ExactEvaluatorConfig()
        ec_config.compute_grads = True
        nrunner = 0
        ec_mgr = EvaluatorManager(Z2System2D, system_cfg, ec_config, nrunner)

        # Evaluate
        result = ec_mgr.simulate()

        self.assertAlmostEqual(utils.get_obs_mean_df(result, "energy"), 14.756365491)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "mag_energy"), 3.898945142)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "el_energy"), 6.523328736)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "int_energy"), -0.00644230479)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "mass_energy"), 3.943588498)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "chem_energy"), 0.396945419)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "average_occupation")[0], 0.4961817739)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "wilson_loop_0-0_1x1"), 0.220211)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "square_string_0-0_1x1"), 0.08457018)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "FM_1x1"), 0.18021784)
        idx = (0, 0, 2)  # old G2 index 1 was y1r; in generic G4(ncopy=2) order y1r is index 2
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "grad_norm")[idx], -1.907041756)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "mag_energy_grad")[idx], 3.0555902689)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "el_energy_grad")[idx], 3.6992497863e-5)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "int_energy_grad")[idx], -1.064279918)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "mass_energy_grad")[idx], 0.24362347)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "chem_energy_grad")[idx], -0.0002859179)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "energy_grad")[idx], 2.234684899)

    def test_mceval_against_previous_version(self):
        """Run a full MCEval and compare with a well tested previous version of the code.
        We compare against the same results as found with an exacteval on commit 566db49."""

        # System config
        # Couplings
        g = 0.8
        g_int = 0.9
        mass = 1.3
        chem = [0.2]

        # Ansatz settings
        L = 2
        num_pg_layer = 1
        num_fermionic_layer = 1
        nlayer = num_pg_layer + num_fermionic_layer
        unitcell_size = 2
        enforce_u1 = False
        el_links = [0, 1]  # links on which to compute the electric energy
        gauge_fixing = 0  # no guage fixing for MC

        # Construct the system
        lat = lattice.Lattice2D(L, L, gf_num_of_rows=gauge_fixing)

        system_cfg = utils.make_z2_2copy_config(
            lat,
            g / 2,
            1.0 / (2.0 * g),
            g_int,
            mass,
            chem,
            num_pg_layer=num_pg_layer,
            num_fermionic_layer=num_fermionic_layer,
            mod_link_inds=el_links,
            unitcell_size=unitcell_size,
            enforce_u1_symmetry=enforce_u1,
        )  # 2 copy ansatz

        # Set parameters - the comparisons below depend on the parameters
        seed = 12
        rngstate = np.random.RandomState(seed)
        shape = (nlayer, unitcell_size, 20)
        paramvec = rngstate.rand(*shape)
        paramvec = reorder_g2_paramvec_to_g4_order(paramvec, system_cfg)
        system_cfg.paramvec = paramvec

        # Evaluator
        mc_config = MonteCarloEvaluatorConfig(warmup_steps=5000, meas_steps=5000, binsize=1, update_size_per_step=2)
        mc_config.compute_grads = True
        nrunner = 0
        mc_mgr = EvaluatorManager(Z2System2D, system_cfg, mc_config, nrunner)

        # Evaluate
        result = mc_mgr.simulate()

        self.assertAlmostEqual(utils.get_obs_mean_df(result, "energy"), 14.7563, places=0)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "mag_energy"), 3.8989, places=0)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "el_energy"), 6.5233, places=0)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "int_energy"), -0.0064, places=1)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "mass_energy"), 3.9436, places=0)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "chem_energy"), 0.3969, places=1)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "average_occupation")[0], 0.4962, places=1)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "wilson_loop_0-0_1x1"), 0.2202, places=1)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "square_string_0-0_1x1"), 0.08457, places=1)
        # self.assertAlmostEqual(utils.get_obs_mean_df(result, "FM_1x1"), 0.18022, places=1) # not implemented in MC
        idx = (0, 0, 2)  # old G2 index 1 was y1r; in generic G4(ncopy=2) order y1r is index 2
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "grad_norm")[idx], -1.9070, places=0)
        self.assertAlmostEqual(utils.get_obs_mean_df(result, "energy_grad")[idx], 2.2347, places=0)

    def test_system_against_previous_version(self):
        """Compare against the output of the code from a well tests previous version of the code.
        We compare against commit 566db49."""

        # Couplings
        g = 1.3
        g_int = 0.4
        mass = 0.7
        chem = [0.2]

        # Ansatz settings
        L = 2
        num_pg_layer = 1
        num_fermionic_layer = 1
        unitcell_size = 2
        enforce_u1 = False
        el_links = [0, 1]  # links on which to compute the electric energy
        gauge_fixing = -1  # maximal tree

        # Hard coded paramvec for which results were checked on trusted version of the code (commit 566db49)
        paramvec = np.asarray(
            [
                [
                    [
                        0.00000000e00,
                        7.20324493e-01,
                        1.14374817e-04,
                        0.00000000e00,
                        1.46755891e-01,
                        9.23385948e-02,
                        1.86260211e-01,
                        3.45560727e-01,
                        3.96767474e-01,
                        5.38816734e-01,
                        0.00000000e00,
                        6.85219500e-01,
                        2.04452250e-01,
                        0.00000000e00,
                        2.73875932e-02,
                        6.70467510e-01,
                        4.17304802e-01,
                        5.58689828e-01,
                        1.40386939e-01,
                        1.98101489e-01,
                    ],
                    [
                        0.00000000e00,
                        9.68261576e-01,
                        3.13424178e-01,
                        0.00000000e00,
                        8.76389152e-01,
                        8.94606664e-01,
                        8.50442114e-02,
                        3.90547832e-02,
                        1.69830420e-01,
                        8.78142503e-01,
                        0.00000000e00,
                        4.21107625e-01,
                        9.57889530e-01,
                        0.00000000e00,
                        6.91877114e-01,
                        3.15515631e-01,
                        6.86500928e-01,
                        8.34625672e-01,
                        1.82882773e-02,
                        7.50144315e-01,
                    ],
                ],
                [
                    [
                        9.88861089e-01,
                        7.48165654e-01,
                        2.80443992e-01,
                        7.89279328e-01,
                        1.03226007e-01,
                        4.47893526e-01,
                        9.08595503e-01,
                        2.93614148e-01,
                        2.87775339e-01,
                        1.30028572e-01,
                        1.93669579e-02,
                        6.78835533e-01,
                        2.11628116e-01,
                        2.65546659e-01,
                        4.91573159e-01,
                        5.33625451e-02,
                        5.74117605e-01,
                        1.46728575e-01,
                        5.89305537e-01,
                        6.99758360e-01,
                    ],
                    [
                        1.02334429e-01,
                        4.14055988e-01,
                        6.94400158e-01,
                        4.14179270e-01,
                        4.99534589e-02,
                        5.35896406e-01,
                        6.63794645e-01,
                        5.14889112e-01,
                        9.44594756e-01,
                        5.86555041e-01,
                        9.03401915e-01,
                        1.37474704e-01,
                        1.39276347e-01,
                        8.07391289e-01,
                        3.97676837e-01,
                        1.65354197e-01,
                        9.27508580e-01,
                        3.47765860e-01,
                        7.50812103e-01,
                        7.25997985e-01,
                    ],
                ],
            ]
        )

        # Construct the system
        lat = lattice.Lattice2D(L, L, gf_num_of_rows=gauge_fixing)

        # paramvec = np.random.rand(nlayer, unitcell_size, 20)
        cfg = utils.make_z2_2copy_config(
            lat,
            g / 2,
            1.0 / (2.0 * g),
            g_int,
            mass,
            chem,
            num_pg_layer=num_pg_layer,
            num_fermionic_layer=num_fermionic_layer,
            mod_link_inds=el_links,
            unitcell_size=unitcell_size,
            enforce_u1_symmetry=enforce_u1,
        )  # 2 copy ansatz
        paramvec = reorder_g2_paramvec_to_g4_order(paramvec, cfg)
        cfg.paramvec = paramvec
        system_z2 = system.Z2System2D(cfg)
        system_z2.cfg.enforce_parameter_conditions(system_z2.cfg.paramvec)

        # Check that enforcing parameter conditions didn't change the parameters, because they were already satisfied
        self.assertTrue(np.allclose(paramvec, system_z2.cfg.paramvec))

        # Set a non-trivial gauge configuration
        neutral_gauge = system_z2.cfg.gaugemgr.get_neutral_gauge_value()
        flux_gauge = system_z2.cfg.gaugemgr.get_representation(np.pi)
        config = np.asarray(
            [
                neutral_gauge,
                flux_gauge,
                flux_gauge,
                neutral_gauge,
                neutral_gauge,
                neutral_gauge,
                flux_gauge,
                neutral_gauge,
            ]
        )
        system_z2.update_gauge_full_system(config)

        # Get system observables
        mag_energy = system_z2.mag_energy
        el_energy = system_z2.el_energy
        int_energy = system_z2.int_energy
        mass_energy = system_z2.mass_energy
        chem_energy = system_z2.chem_energy
        total_energy = system_z2.energy
        occupations = system_z2.occupations_before_ph
        loop = system_z2.cfg.lattice.generate_all_wilson_loops((0, 1), [(1, 1)])
        wilson_loop = system_z2.compute_path(loop[0])
        idx = (1, 1, 5)  # a random index to compare, since it's not necessary to check all components
        norm_grad = system_z2.grad_over_norm_vec[idx]
        el_grad = system_z2.el_energy_op_grad_vec[idx]
        int_grad = system_z2.int_energy_op_grad_vec[idx]
        mass_grad = system_z2.mass_energy_op_grad_vec[idx]
        chem_grad = system_z2.chem_energy_op_grad_vec[idx]

        # Hard coded correct values
        expected_mag_energy = 3.0769230769230766
        expected_el_energy = 10.451819892201765
        expected_int_energy = -0.8681958598127099
        expected_mass_energy = 2.424562098759527
        expected_chem_energy = 0.39645909509164884
        expected_total_energy = 15.481568303163307
        expected_occupations = np.asarray([[0.86458734, 0.08913085, 0.17018682, 0.85839047]])
        expected_wilson_loop = -1.0
        expected_norm_grad = 0.7750323190625908
        expected_el_grad = 0.08908064243278789
        expected_int_grad = 0.3216484282058193
        expected_mass_grad = 0.28476820216866683
        expected_chem_grad = 0.10198218389629972

        # Comparisons
        self.assertAlmostEqual(mag_energy, expected_mag_energy)
        self.assertAlmostEqual(el_energy, expected_el_energy)
        self.assertAlmostEqual(int_energy, expected_int_energy)
        self.assertAlmostEqual(mass_energy, expected_mass_energy)
        self.assertAlmostEqual(chem_energy, expected_chem_energy)
        self.assertAlmostEqual(total_energy, expected_total_energy)
        self.assertTrue(np.allclose(occupations, expected_occupations))
        self.assertAlmostEqual(wilson_loop, expected_wilson_loop)
        self.assertAlmostEqual(norm_grad, expected_norm_grad)
        self.assertAlmostEqual(el_grad, expected_el_grad)
        self.assertAlmostEqual(int_grad, expected_int_grad)
        self.assertAlmostEqual(mass_grad, expected_mass_grad)
        self.assertAlmostEqual(chem_grad, expected_chem_grad)
