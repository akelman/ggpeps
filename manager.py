"""
Main script to control the simulation. 
Further details about the usage of the script can be found in README.md.
"""

# Imports 
import os
import sys
import logging
from timeit import default_timer as timer

os.environ["RAY_DEDUP_LOGS"] = "0" # Ensure that logs are not deduplicated, i.e. the same log message can be printed from different workers
import ray

import numpy as np
np.set_printoptions(linewidth=200)

import ggpeps
from ggpeps.caching import Cache
from ggpeps.system import Z2System2DConfig, Z2System2D
from ggpeps.system import Z2System2D2CConfig, Z2System2D2C
from ggpeps.system import Z2System2D_G2C_F2C_Config, Z2System2D_G2C_F2C
from ggpeps.system import Z2System2D_G2C_F4C_Config, Z2System2D_G2C_F4C
from ggpeps.system import Z2System2D_8C_Config, Z2System2D_8C

from ggpeps import utils
from ggpeps import lattice as lat
from ggpeps.measurement import Measurement
from ggpeps.mc import MonteCarloEvaluatorConfig
from ggpeps.evaluator_manager import EvaluatorManager
from ggpeps.minimizer import Minimizer, MinimizerConfig

logger = logging.getLogger(ggpeps.LOGGER_NAME)

# set up to allow execution to end gracefully if process is signalled appropriately
import signal
INTERRUPT_EXIT_CODE = 10

def save_state_on_exit():
    args = ggpeps.global_vars["args"]
    cache = ggpeps.global_vars["cache"]
    
    if "min" in args.mode:
        minimizer = ggpeps.global_vars["minimizer"]
        cache.add_obj_to_cache("evaluator_manager", minimizer.evaluator_manager)
        logger.info(f"Added evaluator manager to cache.")
    elif "eval" in args.mode:
        eval_manager = ggpeps.global_vars["eval_manager"]
        cache.add_obj_to_cache("evaluator_manager", eval_manager)
        logger.info(f"Added evaluator manager to cache.")

    cache.save_cache_file()
    logger.info(f"Saved cache file to {cache.cache_file}.")
    return

def signal_handler(signum, frame):

    save_state_on_exit()
    logger.info(f"Received signal {signum}. Exiting.\n\n")
    sys.exit(INTERRUPT_EXIT_CODE)

signal.signal(signal.SIGTERM, signal_handler) # register the signal handler
#signal.signal(signal.SIGUSR1, signal_handler) #TODO: fix for windows
signal.signal(signal.SIGINT, signal_handler) # responds to CTRL-C 


def args2logname(args, couplings: dict) -> str:
    """Convert arguments to a name for the log file

    Args:
        args (namespace): Namespace of arguments as provided by argparse
        couplings (dict): Dictionary of all couplings

    Returns:
        str: Filename of the log file
    """
    couplings_str = f"gel_{couplings['g_el']}_gmag_{couplings['g_mag']}_gint_{couplings['g_int']}_gmass_{couplings['g_mass']}"

    if "exact" in args.mode:
        fname = f"log_{args.mode}_L_{args.L}x{args.L}_{couplings_str}.log"
    else:
        fname = f"log_{args.mode}_L_{args.L}x{args.L}_{couplings_str}_nlayer_{args.nlayer}_wsteps_{args.warmup_steps}_msteps_{args.meas_steps}.log"
    return os.path.join(args.output, fname)

def translate_parameters(system_cfg, params: str, rng_state: np.random.RandomState):
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
    if params is not None and len(params)==1 and isinstance(params[0],str) and os.path.isfile(params[0]):
        # The parameters are stored in a file and we can load them
        dest = np.load(params[0])
        dest = np.reshape(dest,(nlayer,-1))
    elif params is None or params == "rand" :
        # No parameters are given and we randomize
        dest = rng_state.rand(nlayer, nparams)
    else:
        # The parameters are listed explicitly in the command line
        dest = np.asarray(params, dtype=float)
        try:
            dest = dest.reshape((nlayer, nparams))
        except:
            logger.warning("Reshape of provided parameters impossible. Starting with random parameters.")
            dest = rng_state.rand(nlayer, nparams)
    return dest

def validate_inputs(args) -> bool:

    if args.L % 2 != 0:
        logger.error("The lattice dimension must currently be an even number.") # this is important when staggering
        return False
    if args.ncopy == 1 and args.g_mass != 0:
        logger.error("Not Implemented: the mass term has not yet been implemented for the 1 copy case.")
        return False
    if args.ncopy not in [1,2,4,8]:
        logger.error("Not Implemented: only 1,2,4, or 8 copies are possible.")
        return False

    return True


def main(args):
    raw_command = ' '.join(sys.argv)
    ind = raw_command.index("manager.py")
    raw_command = raw_command[ind:]

    # Make sure that the output directory is fine
    if os.path.exists(args.output):
        if not os.path.isdir(args.output):
            print(f"Output directory '{args.output}' exists and is not a directory. Aborting.", file=sys.stderr)
            sys.exit(1)
    else:
        os.makedirs(args.output)

    # Set up the simulation
    L = args.L
    g = args.g
    if args.g_el is None and g is not None:
        g_el = g/2.0
    else:
        g_el = args.g_el
    if args.g_mag is None:
        if g is not None:
            g_mag = 1./(2*g)
        else:
            g_mag = 1/(4*g_el)
    else:
        g_mag = args.g_mag
    g_int = args.g_int
    g_mass = args.g_mass
    couplings = {"g_el":g_el, "g_mag":g_mag, "g_int":g_int, "g_mass":g_mass}

    # Set up the logger
    log_filename = args2logname(args, couplings)
    ggpeps.logger_file = log_filename
    utils.setup_logger(logger, log_filename, args.level)
    
    # Validate input arguments
    if not validate_inputs(args):
        sys.exit(1)

    # Set up ray before we actually start with the simulation
    # Ray uses randomness internally and we don't want it to mix up the setting of the seed
    if ggpeps.GPU_AVAILABLE and args.nrunner > 0:
        ray.init(num_cpus=args.nrunner, num_gpus=1)
    elif args.nrunner > 0:
        ray.init(num_cpus=args.nrunner)

    # Set up the MC Config
    mc_config = MonteCarloEvaluatorConfig()
    mc_config.warmup_steps = args.warmup_steps
    mc_config.meas_steps = args.meas_steps
    mc_config.binsize = args.binsize
    if args.use_systemsize_updates or args.update_size == "system":
        mc_config.update_size_per_step = 2*L**2
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
    logger.info(f"Seed: {seed}") # used for both MC and randomizing parameters
    logger.info("======= RAW COMMAND ========")
    logger.info(raw_command)
    logger.info("============================")

    # We are focussing on 2 dimensions for the moment
    lattice = lat.Lattice2D(L, L)

    # Depending on the parameters, we instantiate different systems
    # Since they all share the same interface, we do not care much about the details of the system after this point
    if args.fermions:
        if args.ncopy == 2:
            # Z2 system with 4 copies of virtual fermions on the links (2 for the pure gauge case, 2 for interacting with physical fermions)
            system_type = Z2System2D_G2C_F2C
            system_cfg = Z2System2D_G2C_F2C_Config(lattice, g_el, g_mag, g_int, g_mass, nlayer=args.nlayer)
        elif args.ncopy == 4:
            # Z2 system with 6 copies of virtual fermions on the links (2 for the pure gauge case, 4 for interacting with physical fermions)
            system_type = Z2System2D_G2C_F4C
            system_cfg = Z2System2D_G2C_F4C_Config(lattice, g_el, g_mag, g_int, g_mass, nlayer=args.nlayer)
        elif args.ncopy == 8:
            system_type = Z2System2D_8C
            system_cfg = Z2System2D_8C_Config(lattice, g_el, g_mag, g_int, g_mass, nlayer=args.nlayer)
        else:
            logger.error("Not Implemented: Only 2, 4, or 8 copies are possible with fermions.")
            sys.exit(1)
    else:
        if args.ncopy == 1:
            # Z2 system with one copy of virtual fermions on the links
            system_type = Z2System2D
            system_cfg = Z2System2DConfig(lattice, g_el, g_mag, g_int, g_mass, nlayer=args.nlayer)
        elif args.ncopy == 2:
            # Z2 system with two copies of virtual fermions on the links
            system_type = Z2System2D2C
            system_cfg = Z2System2D2CConfig(lattice, g_el, g_mag, g_int,  g_mass, nlayer=args.nlayer)
        else:
            logger.error("Not Implemented: Only 1, 2, or 4 copies are possible without fermions.")
            sys.exit(1)

    # We use a local random number generator instead of the global numpy one to assure
    # reproducibility across different runs, even when using mulitple processes
    rngstate = np.random.RandomState(seed)
    mc_config.seed = seed

    # Translate the command line input to a valid parameter vector
    paramvec = translate_parameters(system_cfg, args.params, rngstate)
    system_cfg.paramvec = paramvec

    # Ensure pure gauge (setting t parameter(s) to zero) if enabled
    if not args.fermions:
        system_cfg.make_pure_gauge()

    # Enforce the required parameter conditions to get the correct use of layers
    # This only has an effect for the ansatz's with fermions
    system_cfg.enforce_parameter_conditions(system_cfg.paramvec)

    # Switch to control the binning analysis on EOM (Error of mean)
    if args.no_bin_eom:
        Measurement.use_rebinning = False

    # Device selection: Checks if GPUs are available. If yes it uses the first available GPU;
    # if not, defaults to using the CPU.
    logger.info("========= GPU INFO =========")
    if ggpeps.GPU_AVAILABLE:
        logger.info(f"Found GPU, using {ggpeps.PREFERRED_DEVICE}.")
        # TODO: add basic GPU info
        # logger.info(f"GPU info: {ggpeps.PREFERRED_DEVICE.device_kind}"
    else:
        logger.info("No GPUs found, falling back to CPU.")
    logger.info("============================")

    # Update Log
    logger.info("======= SYSTEM INFO ========")
    logger.info(f"L: {L}")
    logger.info(f"# of layers: {system_cfg.nlayer}")
    logger.info(f"# of copies: {args.ncopy}")
    logger.info(f"fermions: {args.fermions}")
    logger.info(f"g (lambda): {g}")
    logger.info(f"g_el: {g_el}")
    logger.info(f"g_mag: {g_mag}")
    logger.info(f"g_int: {g_int}")
    logger.info(f"g_mass: {g_mass}")
    logger.info(f"Rebinning EOM: {Measurement.use_rebinning}")
    logger.info(f"Starting parameters: {paramvec}")
    logger.info("============================")
    
    if "mc" in args.mode:
        logger.info("========= MC INFO ==========")
        logger.info(f"Seed: {mc_config.seed}")
        logger.info(f"Warmup steps: {mc_config.warmup_steps}")
        logger.info(f"Measurement steps: {mc_config.meas_steps}")
        logger.info(f"Bin size: {mc_config.binsize}")
        logger.info(f"Update size: {mc_config.update_size_per_step} (out of {2*L**2} total links)")
        logger.info(f"Number of Ray runners: {args.nrunner} (zero indicates not using Ray)")
        logger.info("============================")
        mc_config.warmup_log_freq = args.warmup_log_freq
        mc_config.run_log_freq = args.run_log_freq
    if "min" in args.mode:
        logger.info("====== MINIMIZER INFO ======")
        logger.info(f"Method: {args.method.upper()}")
        logger.info(f"Max Iterations: {args.maxiter}")
        logger.info(f"Learning rate: {args.alpha}")
        logger.info(f"Min grad: {args.min_grad}")
        logger.info("============================")

    # Set up cache
    # and save the command line arguments to ggpeps global variable so that they are available everywhere
    cache = Cache(args.mode)
    ggpeps.global_vars["args"] = args
    ggpeps.global_vars["cache"] = cache
    if not args.ignore_cache:
        cache.load_cache_file(cache.cache_file)

    # Call different functions depending on the mode specified via CLI
    if args.mode == "eval-mc":
        # Evaluate observables for a given set of parameters with Monte Carlo
        
        mc_config.minimizer_mode = args.compute_grads
        if cache.load_obj_from_local_cache('evaluator_manager') is not None:
            mc_mgr = cache.load_obj_from_local_cache('evaluator_manager')
            logger.info(f"Loaded evaluator manager from cache.")
        else:
            mc_mgr = EvaluatorManager(system_type, system_cfg, mc_config, args.nrunner)
        ggpeps.global_vars["eval_manager"] = mc_mgr # save for global access
        
        start = timer()
        mc_result = mc_mgr.simulate()
        stop = timer()
        mc_result.print_stats()
        mc_result.save(output_dir = args.output)

        logger.info("==== Acceptance prob =======")
        logger.info(f"Acceptance probability: {mc_result.get_obs_mean('acceptance_prob')}")
        logger.info("============================")
    elif args.mode == "min-mc":
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
        ggpeps.global_vars["minimizer"] = minimizer # save for global access

        start = timer()
        result = minimizer.minimize()
        stop = timer()
        logger.info(result)
        minimizer.save(output_dir = args.output)
    elif args.mode == "eval-exact":
        # Evaluate observables for a given set of parameters with exact contraction
        ex_eval = EvaluatorManager(system_type, system_cfg, None, args.nrunner)
        
        start = timer()
        dest = ex_eval.simulate()
        stop = timer()
        
        dest_dict = dest.obsdict
        dest.save(output_dir=args.output)
        for key, val in dest_dict.items():
            logger.info(f"{key}: {val}")
    elif args.mode == "min-exact":
        # Find the minimal energy (the optimal parameter vector) while evaluating the state with exact contractions

        start = timer()
        ex_mgr = EvaluatorManager(system_type, system_cfg, None, args.nrunner)

        min_cfg = MinimizerConfig()
        min_cfg.method = args.method.upper()
        min_cfg.max_iter = args.maxiter
        min_cfg.alpha = args.alpha
        min_cfg.min_grad = args.min_grad

        minimizer = Minimizer(min_cfg, ex_mgr)
        ggpeps.global_vars["minimizer"] = minimizer

        start = timer()
        result = minimizer.minimize()
        stop = timer()
        logger.info(result)
        minimizer.save(output_dir = args.output)
    elif args.mode == "minmult-mc":
        """This mode has not been used in a while and might not work anymore.
        The port variable is intended for use with ray, but this does not currently work with the EvaluatorManager.
        It's possible the the port workaround is unneeded with current versions of ray."""

        # Optimize the parameters with multiple runs (useful if BFGS has problems with the Hessian)

        # Set the parameters of the minimizer according to the command line
        min_cfg = MinimizerConfig()
        min_cfg.method = args.method
        min_cfg.max_iter = args.maxiter
        min_cfg.alpha = args.alpha
        min_cfg.use_metric = args.use_metric

        start = timer()
        resultvec = []
        mc_config.minimizer_mode = True
        for i in range(args.minmult_iter):
            logger.info(f"Minimization iteration: {i:02d}")
            mc = EvaluatorManager(mc_config, system_type, system_cfg, args.nrunner, port=args.port) 
            minimizer = Minimizer(mc, min_cfg)

            resultvec.append(minimizer.minimize())
            system_cfg.paramvec = resultvec[-1].paramvec
        stop = timer()
        # TODO: We can merge the resultvec to get a full result
        minimizer.save(output_dir = args.output)
        # We run a final iteration of the MC simulation with all observables
        mc_config.minimizer_mode = False
        mc_mgr = EvaluatorManager(mc_config, system_type, system_cfg, args.nrunner, port=args.port)
        mc_result = mc_mgr.simulate()
        mc_result.save(output_dir = args.output)
    else:
        logger.error(f"Mode '{args.mode}' unknown.")

    # Save cache with all final computation results
    save_state_on_exit()

    # Log the time taken for the simulation
    logger.info("========== TIME ============")
    logger.info(f"The simulation took {stop - start}s.")
    logger.info("============================\n\n") # add new lines to separate from next run



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        epilog = """Possible modes: eval-mc, eval-exact, min-mc (minimize with MC), min-exact, minmult-mc. \
                    Possible logging levels: critical, error, warning, info, debug."""
    )

    # Mode and lattice size
    parser.add_argument("mode",
                        type=str,
                        choices=["eval-mc", "eval-exact", "min-mc", "min-exact", "minmult-mc"],
                        help="Mode of the program")
    parser.add_argument("L", type=int, help="Size of the square system (one side)")
    
    # Hamiltonian couplings
    parser.add_argument("--g", type=float, default=None, help="coupling constant (equal to lambda)")
    parser.add_argument("--g_el", "--el", type=float, help="electric coupling constant (if not given, computed as g/2)")
    parser.add_argument("--g_mag", "--mag", type=float, help="magnetic coupling constant (if not given, computed as [2*g]^-1)")
    parser.add_argument("--g_int", "--int", type=float, default=0.0, help="gauge matter coupling")
    parser.add_argument("--g_mass", "--mass", "--m", type=float, default=0.0, help="matter constant")

    # Ansatz parameters
    parser.add_argument("--nlayer", default=1, type=int,
                        help="Number of PEPS layers for the variational state")
    parser.add_argument("--ncopy", default=1, type=int,
                        help="Number of virtual fermions on the links per layer")

    # Other system parameters
    parser.add_argument("--params", nargs="+",
                        help="Parameters passed as a starting configuration (Order for one copy: [t1r, t2r,..., y1r, y2r,..., z1r, z2r..., t1i, t2i, ..., y1i, ... z1i])")
    parser.add_argument("--fermions", action="store_true", default=False, 
                        help="Use an ansatz that allows for the inclusion of fermions") # TODO: improve handling of pure-gauge and fermions arguments

    # Monte Carlo settings
    parser.add_argument("--seed", type=int, help="Seed for the MC simulation and parameter initialization")
    parser.add_argument("--warmup_steps", type=int, default=int(1e5), help="Number of warmup steps")
    parser.add_argument("--meas_steps", type=int, default=int(1e5), help="Number of run steps")
    parser.add_argument("--binsize", default=1, type=int, help="Binsize used in the MC computation")
    parser.add_argument("--no-bin-eom", default=False, action="store_true",
                        help="Use the standard EOM instead of a rebinning analysis")
    parser.add_argument("--use-systemsize-updates", action="store_true", default=False,
                        help="Update every spin of the system between each update step. This option is kept for backwards compatibility")
    parser.add_argument("--update_size", type=str, default="1",
                        help="The number of spins to update in each step (can be an integer, or one of: system, halfsystem)")
    parser.add_argument("--compute-grads", action="store_true", default=False,
                        help="Compute grads even if in eval mode")
    
    # Arguments for the minimizer
    parser.add_argument("--method", type=str, default="bfgs", help="Minimization method")
    parser.add_argument("--maxiter", type=int, default=100, help="Maximum number of steps for the minimizer")
    parser.add_argument("--alpha", "--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--min-grad", type=float, default=1e-5, help="Minimal gradient to use as a stopping criterion")
    
    # Output settings
    parser.add_argument("--level", default="info", help="logging level")
    parser.add_argument("--warmup_log_freq", type=int, default=50000, help="frequency at which to log completed warmup steps")
    parser.add_argument("--run_log_freq", type=int, default=50000, help="frequency at which to log completed run steps")
    parser.add_argument("--output", type=str, default='.', help="Output Directory")

    # Cache settings
    parser.add_argument("--ignore_cache", action="store_true", default=False, help="Ignore the cache and start from scratch. A new cache will be saved (and overwrite the old one).") 

    # Arguments for ray
    parser.add_argument("--nrunner", type=int, default=0, help="Number of parallel MC runners")
    
    args = parser.parse_args()
    main(args)
