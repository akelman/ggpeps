"""
Main script to control the simulation.
Further details about the usage of the script can be found in README.md.
"""

# Imports
import os
import sys
import logging
import platform
from timeit import default_timer as timer

# Ensure that logs are not deduplicated, i.e. allow the same log message to be printed from different workers
os.environ["RAY_DEDUP_LOGS"] = "0"
import ray

import numpy as np

np.set_printoptions(linewidth=200)

import ggpeps
from ggpeps.caching import Cache
from ggpeps.system import U1System2DConfig
from ggpeps.system import Z2System2DConfig
from ggpeps.system import Z2System2D_G2C_F2C_Config
from ggpeps.system import Z2System2D_G4C_F4C_Config
from ggpeps.system import Z2System2D_G8C_F8C_Config
from ggpeps.system import D6System2D_Config
from ggpeps.system import Z2System2D
from ggpeps.system import D2nSystem2D

from ggpeps import utils
from ggpeps import lattice as lat
from ggpeps.measurement import Measurement
from ggpeps.nevmc import NEVMC_EvaluatorConfig
from ggpeps.mc import MonteCarloEvaluatorConfig
from ggpeps.exacteval import ExactEvaluatorConfig
from ggpeps.evaluator_manager import EvaluatorManager
from ggpeps.minimizer import Minimizer, MinimizerConfig

logger = logging.getLogger(ggpeps.LOGGER_NAME)

# set up to allow execution to end gracefully if process is signalled appropriately
import signal

INTERRUPT_EXIT_CODE = 10


def save_state_on_exit():
    args = ggpeps.global_vars["args"]
    cache: Cache = ggpeps.global_vars["cache"]

    cache_file = ggpeps.global_vars["args"].save_cache_dest
    if cache_file is not None and not cache.disable_cache:
        cache.save_cache_file(cache_file)
        logger.info(
            f"Saved cache file to {os.path.basename(cache_file)} in output folder."
        )
    return


def signal_handler(signum, frame):

    save_state_on_exit()
    logger.info(f"Received signal {signum}. Exiting.\n\n")
    sys.exit(INTERRUPT_EXIT_CODE)


# Register the signal handlers
signal.signal(signal.SIGTERM, signal_handler)
if platform.system().lower() != "windows":
    # Windows does define SIGUSR1
    signal.signal(signal.SIGUSR1, signal_handler)
signal.signal(signal.SIGINT, signal_handler)  # responds to CTRL-C


def args2logname(args, couplings: dict) -> str:
    """Convert arguments to a name for the log file

    Args:
        args (namespace): Namespace of arguments as provided by argparse
        couplings (dict): Dictionary of all couplings

    Returns:
        str: Filename of the log file
    """
    chem_str = "_".join([f"{val:.3f}" for val in couplings["g_chem"]])
    couplings_str = f"gel_{couplings['g_el']}_gmag_{couplings['g_mag']}_gint_{couplings['g_int']}_gmass_{couplings['g_mass']}_gchem_{chem_str}"

    if "exact" in args.mode:
        fname = f"log_{args.mode}_L_{args.L}x{args.L}_{couplings_str}.log"
    else:
        fname = f"log_{args.mode}_L_{args.L}x{args.L}_{couplings_str}_num_pg_layer_{args.num_pg_layer}_num_fermionic_layer_{args.num_fermionic_layer}_wsteps_{args.warmup_steps}_msteps_{args.meas_steps}.log"
    return os.path.join(args.output, fname)


def translate_parameters(
    system_cfg, params: str, rng_state: np.random.RandomState
) -> tuple[np.array, str]:
    """Translate the parameters given on the commandline to a form useful in the code

    Args:
        system_cfg (SystemConfig): Configuration of the system
        params (str): Parameters as given on the command line
        rng_state (np.random.RandomState): Input state of a PRNG

    Returns:
        np.array: Array of parameters that are suited for the simulation according to the command line parameters
    """
    shape = system_cfg.param_shape()
    if (
        params is not None
        and len(params) == 1
        and isinstance(params[0], str)
        and os.path.isfile(params[0])
    ):
        # The parameters are stored in a file and we can load them
        dest = np.load(params[0])
        dest = np.reshape(dest, shape)
        source = "command-line provided file"
    elif params is None or params == "rand":
        # No parameters are given and we randomize
        dest = rng_state.rand(*shape)
        source = "random state"
    else:
        # The parameters are listed explicitly in the command line
        dest = np.asarray(params, dtype=float)
        try:
            dest = dest.reshape(shape)
            source = "command-line provided parameters"
        except:
            logger.warning(
                "Reshape of provided parameters impossible. Starting with random parameters."
            )
            dest = rng_state.rand(shape)
            source = "random state"
    return dest, source


def validate_inputs(args) -> bool:

    if args.L % 2 != 0:
        logger.error(
            "The lattice dimension must currently be an even number."
        )  # this is important when staggering
        return False
    if args.ncopy == 1 and args.g_mass != 0:
        logger.error(
            "Not Implemented: the mass term has not yet been implemented for the 1 copy case."
        )
        return False
    if args.ncopy not in [1, 2, 4, 8]:
        logger.error("Not Implemented: only 1,2,4, or 8 copies are possible.")
        return False

    return True


def main(args):
    raw_command = " ".join(sys.argv)
    ind = raw_command.index("manager.py")
    raw_command = raw_command[ind:]

    # Make sure that the output directory is fine
    if os.path.exists(args.output):
        if not os.path.isdir(args.output):
            print(
                f"Output directory '{args.output}' exists and is not a directory. Aborting.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        os.makedirs(args.output)

    # Validate input arguments
    if not validate_inputs(args):
        sys.exit(1)

    # Set up ray before we actually start with the simulation
    # (i)  Ray uses randomness internally and we don't want it to mix up the setting of the seed
    # (ii) If ray is initialized after JAX is imported (which happens upon importing ggpeps),
    #      we get warnings about multithreading deadlocks,
    #      see: https://github.com/ray-project/ray/issues/44087
    if ggpeps.GPU_AVAILABLE and args.nrunner > 0:
        # TODO: is it necessary to specify the number of CPUs/GPUs here? Or is in eval manager enough?
        ray.init(num_cpus=args.nrunner, num_gpus=1)
    elif args.nrunner > 0:
        ray.init(num_cpus=args.nrunner)

    # Configure JAX
    import jax

    jax.config.update("jax_enable_x64", True)

    # GPU or CPU detection (compatible with ROCm GPUs)
    # If GPUs are available, use the first available GPU;
    # if not, default to using the CPU.
    available_devices_ = jax.devices()  # available_gpus = jax.devices('gpu')
    PREFERRED_DEVICE = available_devices_[0]
    device_name = PREFERRED_DEVICE.device_kind.lower()

    # Updated GPU detection heuristic
    if any(x in device_name for x in ["gpu", "nvidia", "amd", "rocm"]):
        ggpeps.GPU_AVAILABLE = True
        ggpeps.PREFERRED_DEVICE = PREFERRED_DEVICE
    else:
        ggpeps.GPU_AVAILABLE = False

    # Set up the simulation
    L = args.L
    g = args.g
    if args.g_el is None and g is not None:
        g_el = g / 2.0
    else:
        g_el = args.g_el
    if args.g_mag is None:
        if g is not None:
            g_mag = 1.0 / (2 * g)
        else:
            g_mag = 1 / (4 * g_el)
    else:
        g_mag = args.g_mag
    g_int = args.g_int
    g_mass = args.g_mass
    if args.g_chem is None:
        g_chem = np.zeros(args.num_fermionic_layer)
    else:
        g_chem = np.array(args.g_chem)
    if len(g_chem) != args.num_fermionic_layer:
        raise ValueError(
            "The number of chemical potentials must match the number of fermionic layers."
        )
    couplings = {
        "g_el": g_el,
        "g_mag": g_mag,
        "g_int": g_int,
        "g_mass": g_mass,
        "g_chem": g_chem,
    }

    # Set up the logger
    log_filename = args2logname(args, couplings)
    ggpeps.logger_file = log_filename
    utils.setup_logger(logger, log_filename, args.level)

    # Set up the MC Config
    if "nevmc" in args.mode:
        mc_config = NEVMC_EvaluatorConfig()
    else:
        mc_config = MonteCarloEvaluatorConfig()
    mc_config.warmup_steps = args.warmup_steps
    mc_config.meas_steps = args.meas_steps
    mc_config.binsize = args.binsize
    if args.use_systemsize_updates or args.update_size == "system":
        mc_config.update_size_per_step = 2 * L**2
    elif args.update_size == "halfsystem":
        mc_config.update_size_per_step = L**2
    elif args.update_size.isdecimal():
        mc_config.update_size_per_step = int(args.update_size)
    else:
        logger.error("Unrecognized value for update_size.")
        sys.exit(1)

    # Set up EC config
    ec_config = ExactEvaluatorConfig()

    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(np.iinfo(np.int32).max)

    # Log basic info
    logger.info(f"Git hash: {utils.get_git_hash()}")
    logger.info(f"Logging level: {args.level}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Seed: {seed}")  # used for both MC and randomizing parameters
    logger.info("======= RAW COMMAND ========")
    logger.info(raw_command)
    logger.info("============================")

    # We are focussing on 2 dimensions for the moment
    lattice = lat.Lattice2D(L, L, args.gauge_fixing)

    if args.fermions:
        # in python > 3.12 there is a flag to indicate deprecation, but we want to maintain support for earlier versions.
        logger.warning(
            "The --fermions flag is deprecated, will have no effect, and should no longer be used."
            "Instead specify the number of fermionic layers with --num_fermionic_layer."
        )

    # Depending on the parameters, we instantiate different systems/system_configs
    # Since they all share the same interface, we do not care much about the details of the system after this point
    if args.gauge_group == "Z2":
        system_type = Z2System2D

        if args.ncopy == 1:
            # Z2 system with one copy of virtual fermions on the links and no support for fermions
            if (
                args.num_fermionic_layer != 0
                or not np.allclose(g_mass, 0.0)
                or not np.allclose(g_int, 0.0)
                or not np.allclose(g_chem, 0.0)
            ):
                logger.error(
                    "Not Implemented: The 1 copy case does not support fermionic matter."
                )
                sys.exit(1)
            cfg_class = Z2System2DConfig
        elif args.ncopy == 2:
            cfg_class = Z2System2D_G2C_F2C_Config
        elif args.ncopy == 4:
            cfg_class = Z2System2D_G4C_F4C_Config
        elif args.ncopy == 8:
            cfg_class = Z2System2D_G8C_F8C_Config
        else:
            logger.error(
                "Not Implemented: Only 1, 2, 4, or 8 copies are currently supported for Z2."
            )
            sys.exit(1)
    elif args.gauge_group == "D6":
        system_type = D2nSystem2D
        cfg_class = D6System2D_Config
    elif args.gauge_group == "U1":
        logger.error("Not Implemented: The U1 gauge group is not currently working.")
        sys.exit(1)
    else:
        logger.error(
            "Not Implemented: Only the gauge groups Z2 and D6 are implemented and working."
        )
        sys.exit(1)

    # Create the system configuration of the appropriate type
    system_cfg = cfg_class(
        lattice,
        g_el,
        g_mag,
        g_int,
        g_mass,
        g_chem,
        num_pg_layer=args.num_pg_layer,
        num_fermionic_layer=args.num_fermionic_layer,
        unitcell_size=args.unitcell_size,
        enforce_u1_symmetry=not args.relax_u1,
    )

    # We use a local random number generator instead of the global numpy one to assure
    # reproducibility across different runs, even when using mulitple processes
    rngstate = np.random.RandomState(seed)
    mc_config.seed = seed

    # Translate the command line input to a valid parameter vector
    paramvec, param_source = translate_parameters(system_cfg, args.params, rngstate)
    system_cfg.paramvec = paramvec

    if isinstance(system_cfg, Z2System2DConfig) or isinstance(
        system_cfg, U1System2DConfig
    ):
        # This is a holdover for compatibility with these ansatz's,
        # since they have never been tested or used without being forced to be pure gauge.
        system_cfg.make_pure_gauge()

    # Enforce the required parameter conditions
    system_cfg.enforce_parameter_conditions(system_cfg.paramvec)

    # Switch to control the binning analysis on EOM (Error of mean)
    if args.no_bin_eom:
        Measurement.use_rebinning = False

    # The NEVMC approach does not use the standard minimizer methods
    if args.mode == "min-nevmc":
        args.method = "NEVMC"

    # Update Log
    # Log backend info - GPU/CPU, JAX/NUMPY, precision, etc.
    logger.info("======= BACKEND INFO =======")
    if ggpeps.GPU_AVAILABLE:
        # logger.info(f"Available JAX devices: {available_devices_}")
        logger.info(f"Found GPU, using device: {ggpeps.PREFERRED_DEVICE}.")
        # TODO: add basic GPU info
        # logger.info(f"GPU info: {ggpeps.PREFERRED_DEVICE.device_kind}"
    else:
        logger.info("No GPUs found, falling back to CPU.")
    logger.info(f"Numerical backend: {ggpeps.PREFERRED_BACKEND}")
    arr = ggpeps.xnp.array([1.2, 1.3])
    logger.info(f"Precision: {arr.dtype}")
    logger.info("============================")

    # Caching info
    logger.info("======= CACHE INFO ========")
    if args.ignore_cache:
        logger.info("Cache is disabled.")
    else:
        logger.info(f"Loaded cache from: {args.load_cache}")
        logger.info(f"Will save cache to: {args.save_cache_dest}")
    logger.info("============================")

    # System info
    logger.info("======= SYSTEM INFO ========")
    logger.info(f"L: {L}")
    logger.info(f"# of PG layers: {system_cfg.num_pg_layer}")
    logger.info(f"# of matter layers: {system_cfg.num_fermionic_layer}")
    logger.info(f"# of copies: {args.ncopy}")
    logger.info(f"fermions: {args.fermions}")
    if args.gauge_fixing == -1:
        logger.info(f"Gauge fixing: True - maximal tree")
    elif args.gauge_fixing == 0:
        logger.info(f"Gauge fixing: False")
    elif args.gauge_fixing == -2:
        logger.info(f"Gauge fixing: True - chessboard")
    else:
        logger.info(f"Gauge fixing: {args.gauge_fixing} rows fixed")
    logger.info(f"Unit cell size: {system_cfg.unitcell_size}")
    logger.info(f"Enforce U(1) number conservation: {not args.relax_u1}")
    logger.info(f"g (lambda): {g}")
    logger.info(f"g_el: {g_el}")
    logger.info(f"g_mag: {g_mag}")
    logger.info(f"g_int: {g_int}")
    logger.info(f"g_mass: {g_mass}")
    chem_str = ", ".join([f"{val:.3f}" for val in g_chem])
    logger.info(f"g_chem: {chem_str}")
    logger.info(f"Rebinning EOM: {Measurement.use_rebinning}")
    logger.info(f"Loaded parameters from: {param_source}")
    logger.info(f"Starting parameters: {paramvec}")
    logger.info("============================")

    # Mode info
    if "mc" in args.mode:
        logger.info("========= MC INFO ==========")
        logger.info(f"Seed: {mc_config.seed}")
        logger.info(f"Warmup steps: {mc_config.warmup_steps}")
        logger.info(f"Measurement steps: {mc_config.meas_steps}")
        logger.info(f"Bin size: {mc_config.binsize}")
        logger.info(
            f"Update size: {mc_config.update_size_per_step} (out of {2*L**2} total links)"
        )
        logger.info(
            f"Number of Ray runners: {args.nrunner} (zero indicates not using Ray)"
        )
        logger.info("============================")
        mc_config.warmup_log_freq = args.warmup_log_freq
        mc_config.run_log_freq = args.run_log_freq
    if "min" in args.mode:
        logger.info("====== MINIMIZER INFO ======")
        logger.info(f"Method: {args.method.upper()}")
        logger.info(f"Max Iterations: {args.maxiter}")
        logger.info(f"Convergence tolerance: {args.tol}")
        if args.method.upper() == "CUSTOM" or args.method.upper() == "NEVMC":
            # this is not used by the scipy minimizer, but by the custom and NEVMC minimizer
            logger.info(f"Learning rate: {args.alpha}")
        logger.info("============================")

    # Warn about potential/likely unintended choice of settings
    if not np.allclose(g_chem, 0):
        if args.unitcell_size == 1:
            logger.warning(
                "There is a non-zero chemical potential, but 1-site translation invariance. This may be unintended."
            )
        if not args.relax_u1:
            logger.warning(
                "There is a non-zero chemical potential, but U1 invariance. This may be unintended."
            )

    # Set up cache
    # and save the command line arguments to ggpeps global variable so that they are available everywhere
    cache = Cache(disable_cache=args.ignore_cache)
    if args.load_cache is not None and not cache.disable_cache:
        if os.path.isfile(args.load_cache):
            cache.load_cache_file(args.load_cache)
        else:
            logger.error(f"Unable to find cache file {args.load_cache}.")
    if args.save_cache_dest is not None and not cache.disable_cache:
        if not os.path.isabs(args.save_cache_dest):
            # Save the cache filename as an absolute path (so that it can be used throughout the code,
            # without needing to track the destination).
            args.save_cache_dest = os.path.join(
                args.output, os.path.basename(args.save_cache_dest)
            )
        cache.save_cache_dest = args.save_cache_dest
    ggpeps.global_vars["args"] = args
    ggpeps.global_vars["cache"] = cache

    # Call different functions depending on the mode specified via CLI
    if args.mode == "eval-mc":
        # Evaluate observables for a given set of parameters with Monte Carlo

        mc_config.compute_grads = args.compute_grads
        if cache.load_obj_from_local_cache("evaluator_manager") is not None:
            mc_mgr = cache.load_obj_from_local_cache("evaluator_manager")
            logger.info(f"Loaded evaluator manager from cache.")
        else:
            mc_mgr = EvaluatorManager(system_type, system_cfg, mc_config, args.nrunner)
        ggpeps.global_vars["eval_manager"] = mc_mgr  # save for global access

        start = timer()
        mc_result = mc_mgr.simulate()
        stop = timer()
        mc_result.print_stats()
        mc_result.save(output_dir=args.output)

        logger.info("==== Acceptance prob =======")
        logger.info(
            f"Acceptance probability: {mc_result.get_obs_mean('acceptance_prob')}"
        )
        logger.info("============================")
    elif args.mode == "min-mc":
        # Find the minimal energy (the optimal parameter vector) while evaluating the state with MC

        # Set the parameters of the minimizer
        min_cfg = MinimizerConfig()
        min_cfg.method = args.method.upper()
        min_cfg.max_iter = args.maxiter
        min_cfg.alpha = args.alpha
        min_cfg.tol = args.tol

        # Set up the evaluator
        if min_cfg.method in Minimizer.grad_methods or args.compute_grads:
            mc_config.compute_grads = True
        else:
            # no need to compute grads if not using a gradient-based method
            mc_config.compute_grads = False
        mc_mgr = EvaluatorManager(system_type, system_cfg, mc_config, args.nrunner)

        minimizer = Minimizer(min_cfg, mc_mgr)
        ggpeps.global_vars["minimizer"] = minimizer  # save for global access

        start = timer()
        result = minimizer.minimize()
        stop = timer()
        logger.info(result)
        minimizer.save(output_dir=args.output)
    elif args.mode == "eval-exact":
        # Evaluate observables for a given set of parameters with exact contraction

        ec_config.compute_grads = args.compute_grads
        ex_eval = EvaluatorManager(system_type, system_cfg, ec_config, args.nrunner)
        ggpeps.global_vars["eval_manager"] = ex_eval

        start = timer()
        dest = ex_eval.simulate()
        stop = timer()

        dest_dict = dest.obsdict
        dest.save(output_dir=args.output)
        for key, val in dest_dict.items():
            logger.info(f"{key}: {val}")
    elif args.mode == "min-exact":
        # Find the minimal energy (the optimal parameter vector) while evaluating the state with exact contractions

        min_cfg = MinimizerConfig()
        min_cfg.method = args.method.upper()
        min_cfg.max_iter = args.maxiter
        min_cfg.alpha = args.alpha
        min_cfg.tol = args.tol

        if min_cfg.method in Minimizer.grad_methods or args.compute_grads:
            ec_config.compute_grads = True
        else:
            # no need to compute grads if not using a gradient-based method
            ec_config.compute_grads = False

        ex_mgr = EvaluatorManager(system_type, system_cfg, ec_config, args.nrunner)

        minimizer = Minimizer(min_cfg, ex_mgr)
        ggpeps.global_vars["minimizer"] = minimizer

        start = timer()
        result = minimizer.minimize()
        stop = timer()
        logger.info(result)
        minimizer.save(output_dir=args.output)
    elif args.mode == "min-nevmc":
        # Find the minimal energy (the optimal parameter vector) while evaluating the state with NEVMC

        mc_config.compute_grads = True
        mc_mgr = EvaluatorManager(system_type, system_cfg, mc_config, args.nrunner)

        # Set the parameters of the minimizer according to the command line
        min_cfg = MinimizerConfig()
        min_cfg.max_iter = args.maxiter
        min_cfg.alpha = args.alpha
        min_cfg.tol = args.tol
        min_cfg.method = "NEVMC"

        minimizer = Minimizer(min_cfg, mc_mgr)
        ggpeps.global_vars["minimizer"] = minimizer  # save for global access

        start = timer()
        result = minimizer.minimize_NEVMC()
        stop = timer()

        logger.info(result)

        # Saving does not currently work with NEVMC, since last_result is overwritten
        # minimizer.save(output_dir=args.output)
    elif args.mode == "minmult-mc":
        """This mode has not been used in a while and might not work anymore.
        The port variable is intended for use with ray, but this does not currently work with the EvaluatorManager.
        It's possible the the port workaround is unneeded with current versions of ray.
        """
        # Optimize the parameters with multiple runs (useful if BFGS has problems with the Hessian)

        # Set the parameters of the minimizer according to the command line
        min_cfg = MinimizerConfig()
        min_cfg.method = args.method
        min_cfg.max_iter = args.maxiter
        min_cfg.alpha = args.alpha
        min_cfg.use_metric = args.use_metric

        start = timer()
        resultvec = []
        mc_config.compute_grads = True
        for i in range(args.minmult_iter):
            logger.info(f"Minimization iteration: {i:02d}")
            mc = EvaluatorManager(
                mc_config,
                system_type,
                system_cfg,
                args.nrunner,
                port=args.port,
            )  # TODO: port is not defined!!
            minimizer = Minimizer(mc, min_cfg)

            resultvec.append(minimizer.minimize())
            system_cfg.paramvec = resultvec[-1].paramvec
        stop = timer()
        # TODO: We can merge the resultvec to get a full result
        minimizer.save(output_dir=args.output)
        # We run a final iteration of the MC simulation with all observables
        mc_config.compute_grads = False
        mc_mgr = EvaluatorManager(
            mc_config,
            system_type,
            system_cfg,
            args.nrunner,
            port=args.port,
        )
        mc_result = mc_mgr.simulate()
        mc_result.save(output_dir=args.output)
    else:
        logger.error(f"Mode '{args.mode}' unknown.")

    # Save cache with all final computation results
    save_state_on_exit()

    # Log the time taken for the simulation
    logger.info("========== TIME ============")
    logger.info(f"The simulation took {stop - start}s.")
    logger.info(
        "============================\n\n"
    )  # add new lines to separate from next run


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""Possible modes: eval-mc, eval-exact, min-mc (minimize with MC), min-exact, minmult-mc. \
                    Possible logging levels: critical, error, warning, info, debug.""",
    )
    # Mode and lattice size
    parser.add_argument(
        "mode",
        type=str,
        choices=[
            "eval-mc",
            "eval-exact",
            "min-mc",
            "min-exact",
            "min-nevmc",
            "minmult-mc",
        ],
        help="Mode of the program",
    )
    parser.add_argument("gauge_group", type=str, choices=["Z2", "D6"])

    parser.add_argument("--L", type=int, help="Size of the square system (one side)")

    # Hamiltonian couplings
    parser.add_argument(
        "--g", type=float, default=None, help="coupling constant (equal to lambda)"
    )
    parser.add_argument(
        "--g_el",
        "--el",
        type=float,
        help="electric coupling constant (if not given, computed as g/2)",
    )
    parser.add_argument(
        "--g_mag",
        "--mag",
        type=float,
        help="magnetic coupling constant (if not given, computed as [2*g]^-1)",
    )
    parser.add_argument(
        "--g_int", "--int", type=float, default=0.0, help="gauge matter coupling"
    )
    parser.add_argument(
        "--g_mass", "--mass", "--m", type=float, default=0.0, help="matter coupling"
    )
    parser.add_argument(
        "--g_chem",
        "--chem",
        nargs="+",
        type=float,
        default=None,
        help="chemical potentials",
    )

    # Ansatz parameters
    parser.add_argument(
        "--num_pg_layer",
        default=1,
        type=int,
        help="Number of pure gauge PEPS layers for the variational state",
    )
    parser.add_argument(
        "--num_fermionic_layer",
        default=0,
        type=int,
        help="Number of matter PEPS layers for the variational state",
    )
    parser.add_argument(
        "--ncopy",
        default=1,
        type=int,
        help="Number of virtual fermions on the links per layer",
    )

    # Other system parameters
    parser.add_argument(
        "--params",
        nargs="+",
        help="Parameters passed as a starting configuration (Order for one copy: [t1r, t2r,..., y1r, y2r,..., z1r, z2r..., t1i, t2i, ..., y1i, ... z1i])",
    )
    parser.add_argument(
        "--fermions",
        action="store_true",
        default=False,
        help="Use an ansatz that allows for the inclusion of fermions",
    )  # TODO: this argument is no longer used, and can be removed after some time
    parser.add_argument(
        "--unitcell_size",
        type=int,
        default=1,
        help="Specify the size of the largest unit cell in the system. This determines the degree of translation invariance.",
    )
    parser.add_argument(
        "--relax_u1",
        action="store_true",
        default=False,
        help="Allow the system to disobey the global U(1) (fermionic number) symmetry.",
    )

    # Evaluator settings
    parser.add_argument(
        "--gauge_fixing",
        nargs="?",  # Optional value
        const=-1,  # Value when argument is used without a value
        type=int,  # Convert the input to an integer if provided
        default=0,  # Default value when argument is not used
        help="Gauge fixing: 0 if not provided (default), -1 if --gauge_fixing is used without a value - fix a maximal tree, or -2 to gauge fix like a chess boars, or any integer if we fix a specific number of rows.",
    )

    # Monte Carlo settings
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for the MC simulation and parameter initialization",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=int(1e5), help="Number of warmup steps"
    )
    parser.add_argument(
        "--meas_steps", type=int, default=int(1e5), help="Number of run steps"
    )
    parser.add_argument(
        "--binsize", default=1, type=int, help="Binsize used in the MC computation"
    )
    parser.add_argument(
        "--no_bin_eom",
        default=False,
        action="store_true",
        help="Use the standard EOM instead of a rebinning analysis",
    )
    parser.add_argument(
        "--use_systemsize_updates",
        action="store_true",
        default=False,
        help="Update every spin of the system between each update step. This option is kept for backwards compatibility",
    )
    parser.add_argument(
        "--update_size",
        type=str,
        default="1",
        help="The number of spins to update in each step (can be an integer, or one of: system, halfsystem)",
    )
    parser.add_argument(
        "--compute_grads",
        action="store_true",
        default=False,
        help="Compute grads even if in eval mode",
    )

    # Arguments for the minimizer
    parser.add_argument(
        "--method", type=str, default="bfgs", help="Minimization method"
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=100,
        help="Maximum number of steps for the minimizer",
    )
    parser.add_argument(
        "--alpha", "--lr", type=float, default=0.1, help="Learning rate"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Tolerance for convergence condition (e.g. minimal gradient to use as a stopping criterion)",
    )

    # Output settings
    parser.add_argument("--level", default="info", help="logging level")
    parser.add_argument(
        "--warmup_log_freq",
        type=int,
        default=50000,
        help="frequency at which to log completed warmup steps",
    )
    parser.add_argument(
        "--run_log_freq",
        type=int,
        default=50000,
        help="frequency at which to log completed run steps",
    )
    parser.add_argument("--output", type=str, default=".", help="Output Directory")

    # Cache settings
    parser.add_argument(
        "--load_cache",
        nargs="?",  # Optional value
        const="cache.pkl",  # Value when argument is used without a value
        default=None,  # Default value when argument is not used
        type=str,
        help="Load cache from the specified file. If no file provided, but the flag is present, the cache will be loaded from the default cache file (cache.pkl) if available.",
    )
    parser.add_argument(
        "--save_cache_dest",
        nargs="?",  # Optional value
        const="cache.pkl",  # Value when argument is used without a value
        default=None,  # Default value when argument is not used
        type=str,
        help="Save cache to the specified file. This will overwrite the old one if it exists.",
    )
    parser.add_argument(
        "--ignore_cache",
        action="store_true",
        default=False,
        help="Ignore the cache and do not load or save it. Overrides other cache settings.",
    )

    # Arguments for ray
    parser.add_argument(
        "--nrunner", type=int, default=0, help="Number of parallel MC runners"
    )

    args = parser.parse_args()
    main(args)
