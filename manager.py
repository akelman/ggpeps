"""Main script to control the simulation. 
Further details about the usage of the script can be found in README.md.
"""

# Imports 
import os
import sys
import ray
import logging
from timeit import default_timer as timer

import numpy as np
np.set_printoptions(linewidth=200)

from ggpeps.system import Z2System2D2CConfig, Z2System2D2C
from ggpeps.system import Z2System2DConfig, Z2System2D
from ggpeps.system import Z2System2D4C_Config, Z2System2D4C
from ggpeps.measurement import Measurement

from ggpeps import utils, exacteval
from ggpeps.minimizer import Minimizer, MinimizerConfig
from ggpeps.mc import MonteCarloEstimatorConfig, MonteCarloManager
from ggpeps import lattice as lat


def args2logname(args,params):
    """Convert arguments to a name for the log file

    Args:
        args (namespace): Namespace of arguments as provided by argparse
        params (dict): Dictionary of all couplings

    Returns:
        str: Filename of the log file
    """
    shorthands = {
        "min": "min",
        "minimize": "min",
        "eval": "eval",
        "exact": "exact",
        "minexact": "minexact"
    }
    if "exact" in args.mode:
        fname = f"log_{shorthands[args.mode]}_L_{args.L}x{args.L}_gel_{params['g_el']}_gmag_{params['g_mag']}_gint_{params['g_int']}_gmass{params['g_mass']}.log"
    else:
        fname = f"log_{shorthands[args.mode]}_L_{args.L}x{args.L}_gel_{params['g_el']}_gmag_{params['g_mag']}_gint_{params['g_int']}_gmass{params['g_mass']}_nlayer_{args.nlayer}_wsteps_{args.warmup_steps}_msteps_{args.meas_steps}.log"
    return os.path.join(args.output, fname)

def translate_parameters(system_cfg, params,rng_state):
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
    elif params is None or params=="rand" :
        # No parameters are given and we randomize
        dest = rng_state.rand(nlayer, nparams)
    else:
        # The parameters are listed explicitly in the command line
        dest = np.asarray(params, dtype=float)
        try:
            dest = dest.reshape((nlayer, nparams))
        except:
            logging.warning("Reshape of provided parameters impossible. Starting with random parameters.")
            dest = rng_state.rand(nlayer, nparams)
    return dest

def validate_inputs(args) -> bool:

    if args.L % 2 != 0:
        logging.error("The lattice dimension must currently be an even number.") # this is important when staggering
        return False
    if args.ncopy == 1 and args.g_mass != 0:
        logging.error("Not Implemented: the mass term has not yet been implemented for the 1 copy case.")
        return False
    if args.ncopy not in [1,2,4]:
        logging.error("Not Implemented: only 1,2, or 4 copies are possible.")
        return False

    return True

def main(args):

    # Make sure that the output directory is fine
    if os.path.exists(args.output):
        if not os.path.isdir(args.output):
            print(
                f"Output directory '{args.output}' exists and is not a directory. Aborting.", file=sys.stderr)
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
    params = {"g_el":g_el, "g_mag":g_mag, "g_int":g_int, "g_mass":g_mass}

    # Set up the logger
    h_stdout = logging.StreamHandler(stream=sys.stdout)
    h_stderr = logging.StreamHandler(stream=sys.stderr)
    h_stderr.addFilter(lambda record: record.levelno >= logging.WARNING)
    logging.basicConfig(
        level=args.level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args2logname(args,params)),
            h_stdout,
            h_stderr
        ]
    )

    # Validate input arguments
    if not validate_inputs(args):
        sys.exit(1)

    # Set up ray before we actually start with the simulation
    # Ray uses randomness internally and we don't want it to mix up the setting of the seed
    if args.nrunner > 0:
        ray.init(num_cpus=args.nrunner)

    # Set up the MC Config
    mc_config = MonteCarloEstimatorConfig()
    mc_config.warmup_steps = args.warmup_steps
    mc_config.meas_steps = args.meas_steps
    mc_config.binsize = args.binsize
    mc_config.use_systemsize_update = args.use_systemsize_updates

    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(np.iinfo(np.int32).max)

    # We use a local random number generator instead of the global numpy one to assure
    # reproducibility across different runs, even when using mulitple processes
    rngstate = np.random.RandomState(seed)
    mc_config.seed = seed

    logging.info(f"Git hash: {utils.get_git_hash()}")
    logging.info("========= MC INFO ==========")
    logging.info(f"Seed: {mc_config.seed}")
    logging.info(f"Warmup steps: {mc_config.warmup_steps}")
    logging.info(f"Measurement steps: {mc_config.meas_steps}")
    logging.info("============================")

    # We are focussing on 2 dimensions for the moment
    lattice = lat.Lattice2D(L, L)

    # Depending on the parameters, we instantiate different systems
    # Since they all share the same interface, we do not care much about the details of the system after this point
    if args.ncopy == 1:
        # Z2 system with one copy of virtual fermions on the links
        system_type = Z2System2D
        system_cfg = Z2System2DConfig(lattice, g_el, g_mag, g_int, g_mass, nlayer=args.nlayer)
    elif args.ncopy == 2:
        # Z2 system with two copies of virtual fermions on the links
        system_type = Z2System2D2C
        system_cfg = Z2System2D2CConfig(lattice, g_el, g_mag, g_int,  g_mass, nlayer=args.nlayer)
    elif args.ncopy == 4:
        # Z2 system with 4 copies of virtual fermions on the links (2 for the pure gauge case, 2 for interacting with physical fermions)
        system_type = Z2System2D4C
        system_cfg = Z2System2D4C_Config(lattice, g_el, g_mag, g_int, g_mass)
    else:
        logging.error("Not Implemented: Only 1, 2, or 4 copies are possible.")
        sys.exit(1)

    # Translate the command line input to a valid parameter vector
    paramvec = translate_parameters(system_cfg, args.params, rngstate)
    system_cfg.paramvec = paramvec

    # Ensure pure guage (setting t parameter to zero) if enabled
    if args.pure_gauge:
        system_cfg.make_pure_gauge()

    # For the ansatz with ncopy = 4, enforce the required parameter conditions to get the correct use of layers
    if args.ncopy == 4:
        system_cfg.enforce_parameter_conditions(system_cfg.paramvec)

    # Switch to control the binning analysis on EOM (Error of mean)
    if args.no_bin_eom:
        Measurement.use_rebinning = False

    logging.info("======= SYSTEM INFO ========")
    logging.info(f"L: {L}")
    logging.info(f"# of layers: {system_cfg.nlayer}")
    logging.info(f"# of copies: {args.ncopy}")
    logging.info(f"pure-gauge: {args.pure_gauge}")
    logging.info(f"g (lambda): {g}")
    logging.info(f"g_el: {g_el}")
    logging.info(f"g_mag: {g_mag}")
    logging.info(f"g_int: {g_int}")
    logging.info(f"g_mass: {g_mass}")
    logging.info(f"Rebinning EOM: {Measurement.use_rebinning}")
    logging.info(f"Starting parameters: {paramvec}")
    logging.info("============================")


    # Call different functions depending on the mode specified via CLI
    if args.mode == "eval":
        # Evaluate a given set of parameters with Monte Carlo
        mc_config.minimizer_mode = False
        mc_mgr = MonteCarloManager(mc_config, system_type, system_cfg, args.nrunner)
        start = timer()
        mc_result = mc_mgr.simulate()
        stop = timer()
        mc_result.print_stats()
        mc_result.save(output_dir = args.output)

        logging.info("==== Acceptance prob =======")
        logging.info(f"Acceptance probability: {mc_result.get_obs_mean('acceptance_prob')}")
        logging.info("============================")
    elif args.mode == "minimize" or args.mode == "min":
        # Find the minimal energy (the optimal parameter vector) while evaluating the state with MC
        logging.info("====== MINIMIZER INFO ======")
        logging.info(f"Max Iterations: {args.maxiter}")
        logging.info(f"Learning rate: {args.alpha}")
        logging.info(f"Method: {args.method.upper()}")
        logging.info("============================")

        mc_config.minimizer_mode = True
        mc_mgr = MonteCarloManager(mc_config, system_type, system_cfg, args.nrunner)
        # Set the parameters of the minimizer according to the command line
        min_cfg = MinimizerConfig()
        min_cfg.method = args.method.upper()
        min_cfg.max_iter = args.maxiter
        min_cfg.alpha = args.alpha
        min_cfg.min_grad = args.min_grad

        minimizer = Minimizer(min_cfg,mc_mgr)

        start = timer()
        result = minimizer.minimize()
        stop = timer()
        print(result)
        minimizer.save(output_dir = args.output)
    elif args.mode == "exact":
        # Evaluate a given set of parameters with exact contraction (equivalent to the mode "eval", just exact)
        system = system_type(system_cfg)
        start = timer()
        ex_eval = exacteval.ExactEvaluator(system)
        dest_dict = ex_eval.evaluate()
        stop = timer()
        ex_eval.save(output_dir=args.output)
        for key, val in dest_dict.items():
            print(f"{key}: {val}")
    elif args.mode == "minexact":
        # Find the minimal energy (the optimal parameter vector) while evaluating the state with exact contractions
        logging.info("====== MINIMIZER INFO ======")
        logging.info(f"Max Iterations: {args.maxiter}")
        logging.info(f"Learning rate: {args.alpha}")
        logging.info(f"Method: {args.method.upper()}")
        logging.info("============================")

        start = timer()
        ex_mgr = exacteval.ExactEvaluatorManager(system_type, system_cfg)

        min_cfg = MinimizerConfig()
        min_cfg.method = args.method.upper()
        min_cfg.max_iter = args.maxiter
        min_cfg.alpha = args.alpha
        min_cfg.min_grad = args.min_grad

        minimizer = Minimizer(min_cfg, ex_mgr, use_exact=True)

        start = timer()
        result = minimizer.minimize()
        stop = timer()
        print(result)
        minimizer.save(output_dir = args.output)
    elif args.mode == "minmult":
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
            logging.info(f"Minimization iteration: {i:02d}")
            mc = MonteCarloManager(mc_config, system_type, system_cfg, args.nrunner, port=args.port)
            minimizer = Minimizer(mc, min_cfg)

            resultvec.append(minimizer.minimize())
            system_cfg.paramvec = resultvec[-1].paramvec
        stop = timer()
        # TODO: We can merge the resultvec to get a full result
        minimizer.save(output_dir = args.output)
        # We run a final iteration of the MC simulation with all observables
        mc_config.minimizer_mode = False
        mc_mgr = MonteCarloManager(mc_config, system_type, system_cfg, args.nrunner, port=args.port)
        mc_result = mc_mgr.simulate()
        mc_result.save(output_dir = args.output)
    else:
        logging.error(f"Mode '{args.mode}' unkown.")

    logging.info("========== TIME ============")
    logging.info(f"The simulation took {stop - start}s")
    logging.info("============================")



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        epilog = """Possible modes: eval, minimize (min), exact, minexact, minmult. Possible logging levels: critical, error, warning, info, debug."""
    )

    # Mode and lattice size
    parser.add_argument("mode",
                        type=str,
                        choices=["eval", "min", "exact", "minexact", "minmult"],
                        help="Mode of the program")
    parser.add_argument("L", type=int, help="Size of the square system (one side)")
    
    # Hamiltonian parameters
    parser.add_argument("--g", type=float, default=1.0,
                        help="coupling constant (equal to lambda)")
    parser.add_argument("--g_el", type=float,
                        help="electric coupling constant (if not given, computed as g/2)")
    parser.add_argument("--g_mag", type=float,
                        help="magnetic coupling constant (if not given, computed as [2*g]^-1)")
    parser.add_argument("--g_int", type=float, default=0.0,
                        help="gauge matter coupling")
    parser.add_argument("--g_mass", type=float,
                        default=0.0, help="matter constant")

    # Ansatz parameters
    parser.add_argument("--nlayer",
                        default=1,
                        type=int,
                        help="Number of PEPS layers for the variational state")
    parser.add_argument("--ncopy",
                        default=1,
                        type=int,
                        help="Number of virtual fermions on the links")

    # Other system parameters
    parser.add_argument("--params",
                        nargs="+",
                        help="Parameters passed as a starting configuration (Order for one copy: [t1r, t2r,..., y1r, y2r,..., z1r, z2r..., t1i, t2i, ..., y1i, ... z1i])")
    parser.add_argument("--pure-gauge",
                        action="store_true",
                        default=False,
                        help="Force the coupling of physical and virtual fermions (t-parameters) to be 0")

    # Computation settings
    parser.add_argument("--seed", type=int, help="Seed for the MC simulation")
    parser.add_argument("--warmup_steps", type=int, default=int(1e5), help="Number of warmup steps")
    parser.add_argument("--meas_steps", type=int, default=int(1e5), help="Number of run steps")
    parser.add_argument("--level", default="info", help="logging level")
    parser.add_argument("--binsize", default=1, type=int, help="Binsize used in the MC computation")
    parser.add_argument("--output", type=str, default='.', help="Output Directory")
    parser.add_argument("--no-bin-eom",
                        default=False,
                        action="store_true",
                        help="Use the standard EOM instead of a rebinning analysis")
    parser.add_argument("--use-systemsize-updates",
                        action="store_true",
                        default=False,
                        help="Update every spin of the system between each update step")
    
    # Arguments for the minimizer
    parser.add_argument("--method", type=str, default="bfgs", help="Minimization method")
    parser.add_argument("--maxiter", type=int, default=100, help="Number of steps for the minimizer")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--min-grad", type=float, default=1e-5, help="Minimal gradient to use a stopping criterion")
    
    # Arguments for ray
    parser.add_argument("--nrunner", type=int, default=0, help="Number of parallel MC runners")
    
    args = parser.parse_args()
    main(args)
