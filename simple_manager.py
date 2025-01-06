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

# Ensure that logs are not deduplicated, i.e. the same log message can be printed from different workers
os.environ["RAY_DEDUP_LOGS"] = "0"
import ray

import numpy as np

np.set_printoptions(linewidth=200)

import ggpeps
from ggpeps.caching import Cache
from ggpeps.system import Z2System2DConfig
from ggpeps.system import Z2System2D2CConfig
from ggpeps.system import Z2System2D_G2C_F2C_Config
from ggpeps.system import Z2System2D_G2C_F4C_Config
from ggpeps.system import Z2System2D_8C_Config
from ggpeps.system import Z2System2D

from ggpeps import utils
from ggpeps import lattice as lat
from ggpeps.measurement import Measurement
from ggpeps.mc2 import MonteCarloEvaluatorConfig2
from ggpeps.evaluator_manager import EvaluatorManager
from ggpeps.minimizer import Minimizer, MinimizerConfig

logger = logging.getLogger(ggpeps.LOGGER_NAME)

# set up to allow execution to end gracefully if process is signalled appropriately
import signal

INTERRUPT_EXIT_CODE = 10


def save_state_on_exit():
    args = ggpeps.global_vars["args"]
    cache = ggpeps.global_vars["cache"]

    if not args.ignore_cache_eval:
        if "min" in args.mode:
            minimizer = ggpeps.global_vars["minimizer"]
            cache.add_obj_to_cache("evaluator_manager", minimizer.evaluator_manager)
            logger.info(f"Added evaluator manager to cache.")
        elif "eval" in args.mode:
            eval_manager = ggpeps.global_vars["eval_manager"]
            cache.add_obj_to_cache("evaluator_manager", eval_manager)
            logger.info(f"Added evaluator manager to cache.")

    cache_file = ggpeps.global_vars["args"].cache_file
    cache.save_cache_file(cache_file)
    logger.info(f"Saved cache file to {os.path.basename(cache_file)} in output folder.")
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
    couplings_str = f"gel_{couplings['g_el']}_gmag_{couplings['g_mag']}"

    fname = f"log_{args.mode}_L_{args.L}x{args.L}_{couplings_str}_wsteps_{args.warmup_steps}_msteps_{args.meas_steps}.log"
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
    nparams = system_cfg._nparams
    nlayer = system_cfg.nlayer
    if (
        params is not None
        and len(params) == 1
        and isinstance(params[0], str)
        and os.path.isfile(params[0])
    ):
        # The parameters are stored in a file and we can load them
        dest = np.load(params[0])
        dest = np.reshape(dest, (nlayer, -1))
        source = "command-line provided file"
    elif params is None or params == "rand":
        # No parameters are given and we randomize
        dest = rng_state.rand(nlayer, nparams)
        source = "random state"
    else:
        # The parameters are listed explicitly in the command line
        dest = np.asarray(params, dtype=float)
        try:
            dest = dest.reshape((nlayer, nparams))
            source = "command-line provided parameters"
        except:
            logger.warning(
                "Reshape of provided parameters impossible. Starting with random parameters."
            )
            dest = rng_state.rand(nlayer, nparams)
            source = "random state"
    return dest, source


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

    # GPU or CPU
    available_devices_ = jax.devices()  # available_gpus = jax.devices('gpu')
    PREFERRED_DEVICE = available_devices_[0]
    device_name = PREFERRED_DEVICE.device_kind.lower()
    if "gpu" in device_name or "nvidia" in device_name:  # heuristic
        ggpeps.GPU_AVAILABLE = True
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
    g_int = 0
    g_mass = 0
    g_chem = np.array([0])
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
    mc_config = MonteCarloEvaluatorConfig2()
    mc_config.warmup_steps = args.warmup_steps
    mc_config.meas_steps = args.meas_steps
    mc_config.binsize = args.binsize
    mc_config.gauge_fixing = args.gauge_fixing
    if args.use_systemsize_updates or args.update_size == "system":
        mc_config.update_size_per_step = 2 * L**2
    elif args.update_size == "halfsystem":
        mc_config.update_size_per_step = L**2
    elif args.update_size.isdecimal():
        mc_config.update_size_per_step = int(args.update_size)
    else:
        logger.error("Unrecognized value for update_size.")
        sys.exit(1)

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
    lattice = lat.Lattice2D(L, L)

    # Z2 system with one copy of virtual fermions on the links
    system_cfg = Z2System2DConfig(
        lattice,
        g_el,
        g_mag,
        g_int,
        g_mass,
        None,  # no chemical potential for this ansatz, which does not include matter
        num_pg_layer=1,
        num_fermionic_layer=0,
    )
    system_type = Z2System2D

    # We use a local random number generator instead of the global numpy one to assure
    # reproducibility across different runs, even when using mulitple processes
    rngstate = np.random.RandomState(seed)
    mc_config.seed = seed

    # Translate the command line input to a valid parameter vector
    paramvec, param_source = translate_parameters(system_cfg, args.params, rngstate)
    system_cfg.paramvec = paramvec

    # Ensure pure gauge (setting t parameter(s) to zero) if enabled
    system_cfg.make_pure_gauge()

    # Switch to control the binning analysis on EOM (Error of mean)
    if args.no_bin_eom:
        Measurement.use_rebinning = False

    # Update Log
    logger.info("======= SYSTEM INFO ========")
    logger.info(f"L: {L}")
    logger.info(f"# of PG layers: {system_cfg.num_pg_layer}")
    logger.info(f"# of matter layers: {system_cfg.num_fermionic_layer}")
    logger.info(f"# of copies: {1}")
    logger.info(f"fermions: {args.fermions}")
    logger.info(f"Gauge fixing: {args.gauge_fixing}")
    logger.info(f"g (lambda): {g}")
    logger.info(f"g_el: {g_el}")
    logger.info(f"g_mag: {g_mag}")
    logger.info(f"g_int: {g_int}")
    logger.info(f"g_mass: {g_mass}")
    logger.info(f"g_chem: {np.array2string(g_chem, separator=', ', precision=2)}")
    logger.info(f"Rebinning EOM: {Measurement.use_rebinning}")
    logger.info(f"Loaded parameters from: {param_source}")
    logger.info(f"Starting parameters: {paramvec}")
    logger.info("============================")

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
        if args.method.upper() == "CUSTOM":
            # these are only used by the custom (basic gradient descent) minimizer and are not passed to scipy
            logger.info(f"Learning rate: {args.alpha}")
            logger.info(f"Min grad: {args.min_grad}")
        logger.info("============================")

    # Set up cache
    # and save the command line arguments to ggpeps global variable so that they are available everywhere
    cache = Cache(args.mode)
    if not args.ignore_cache:
        cache.load_cache_file(args.cache_file)
        if args.ignore_cache_eval:
            cache.add_obj_to_cache("evaluator_manager", None)
    if not os.path.isabs(args.cache_file):
        # Save the cache filename as an absolute path (so that it can be used throughout the code,
        # without needing to track the destination).
        args.cache_file = os.path.join(args.output, os.path.basename(args.cache_file))
    ggpeps.global_vars["args"] = args
    ggpeps.global_vars["cache"] = cache

    # Call different functions depending on the mode specified via CLI
    if args.mode == "eval-mc2":
        # Evaluate observables for a given set of parameters with Monte Carlo

        mc_config.minimizer_mode = args.compute_grads
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
    elif args.mode == "min-mc2":
        # Find the minimal energy (the optimal parameter vector) while evaluating the state with MC

        mc_config.minimizer_mode = True
        mc_mgr = EvaluatorManager(system_type, system_cfg, mc_config, args.nrunner)

        # Set the parameters of the minimizer according to the command line
        min_cfg = MinimizerConfig()
        min_cfg.method = args.method.upper()
        min_cfg.max_iter = args.maxiter
        min_cfg.alpha = args.alpha
        min_cfg.min_grad = args.min_grad

        minimizer = Minimizer(min_cfg, mc_mgr)
        ggpeps.global_vars["minimizer"] = minimizer  # save for global access

        start = timer()
        result = minimizer.minimize()
        stop = timer()
        logger.info(result)
        minimizer.save(output_dir=args.output)
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
        epilog="""Possible logging levels: critical, error, warning, info, debug.""",
    )

    # Mode and lattice size
    parser.add_argument(
        "mode",
        type=str,
        choices=["eval-mc2", "min-mc2"],
        help="Mode of the program",
    )
    parser.add_argument("L", type=int, help="Size of the square system (one side)")

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

    # Ansatz parameters -- hard coded in main()

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
    )  # TODO: improve handling of pure-gauge and fermions arguments

    # Evaluator settings
    parser.add_argument("--gauge_fixing", action="store_true", default=False)

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
        "--no-bin-eom",
        default=False,
        action="store_true",
        help="Use the standard EOM instead of a rebinning analysis",
    )
    parser.add_argument(
        "--use-systemsize-updates",
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
        "--compute-grads",
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
        "--min-grad",
        type=float,
        default=1e-5,
        help="Minimal gradient to use as a stopping criterion",
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
        "--ignore_cache",
        action="store_true",
        default=False,
        help="Ignore the cache and start from scratch. A new cache will be saved (and overwrite the old one if it exists).",
    )
    parser.add_argument(
        "--ignore_cache_eval",
        action="store_true",
        default=False,
        help="Ignore the cache eval manager.",
    )
    parser.add_argument(
        "--cache_file",
        type=str,
        default="cache.pkl",
        help="Filename of the cache.",
    )

    # Arguments for ray
    parser.add_argument(
        "--nrunner", type=int, default=0, help="Number of parallel MC runners"
    )

    args = parser.parse_args()
    main(args)
