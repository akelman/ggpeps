"""Main script to control the simulation. 
Further details about the usage of the script can be found in README.md.
"""
import os
import sys
from system import Z2System2D, Z2System2DConfig
from timeit import default_timer as timer
import ray
import utils
import logging
from minimizer import Minimizer
from mc import MonteCarloEstimator, MonteCarloEstimatorConfig, MonteCarloManager
import lattice as lat
import numpy as np
np.set_printoptions(linewidth=200)


def args2logname(args):
    shorthands = {"min": "min", "minimize": "min", "eval": "eval"}
    if args.g_mag == None:
        fname = "log_{}_L_{:02d}_g2_{:.3f}_gm_{:.3f}_t_{:.3f}_y_{:.3f}_z_{:.3f}_wsteps_{:06d}_msteps_{:06d}.log".format(
            shorthands[args.mode], args.L, args.g2, args.g_gm, args.t, args.y, args.z, args.warmup_steps, args.meas_steps)
    else:
        fname = "log_{}_L_{:02d}_g2_{:.3f}_gm_{:.3f}_gmag_{:.3f}_t_{:.3f}_y_{:.3f}_z_{:.3f}_wsteps_{:06d}_msteps_{:06d}.log".format(
            shorthands[args.mode], args.L, args.g2, args.g_gm, args.g_mag, args.t, args.y, args.z, args.warmup_steps, args.meas_steps)
    return fname


def main():
    #Set up ray before we actually start with the simulation
    #Ray uses randomness internally and we don't want it to mix up the setting of the seed
    if args.nrunner > 0:
        ray.init()

    #Set up the MC Config
    mc_config = MonteCarloEstimatorConfig()
    mc_config.warmup_steps = args.warmup_steps
    mc_config.meas_steps = args.meas_steps
    mc_config.binsize = args.binsize
    if args.seed is not None:
        mc_config.seed = args.seed

    #Set up the logger
    logging.basicConfig(
        level=args.level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args2logname(args)),
            logging.StreamHandler()
        ]
    )

    logging.info("Git hash: {}".format(utils.get_git_hash()))
    logging.info("========= MC INFO ==========")
    logging.info("Seed: {}".format(mc_config.seed))
    logging.info("Warmup steps: {}".format(mc_config.warmup_steps))
    logging.info("Measurement steps: {}".format(mc_config.meas_steps))
    logging.info("============================")

    #Set up the simulation
    np.random.seed(mc_config.seed)
    L = args.L
    g2 = args.g2
    g_gm = args.g_gm
    g_mag = args.g_mag
    # We are focussing on 2 dimensions for the moment
    lattice = lat.Lattice2D(L, L)

    paramdict = {"t": args.t, "y": args.y, "z": args.z}
    # TODO: This is now a specialized version that runs only Z2 System 2D.
    # We will have to make this more general at some point.
    system_cfg = Z2System2DConfig(paramdict, lattice, g2, g_gm, g_mag)

    logging.info("======= SYSTEM INFO ========")
    logging.info("L: {}".format(L))
    logging.info("t: {}".format(args.t))
    logging.info("y: {}".format(args.y))
    logging.info("z: {}".format(args.z))
    logging.info("g^2: {}".format(g2))
    logging.info("g_mag: {}".format(g_mag))
    logging.info("g_gm: {}".format(g_gm))
    #logging.info("Method: {}".format(method_str))
    logging.info("============================")

    mc = MonteCarloManager(mc_config, Z2System2D, system_cfg, args.nrunner)
    if args.mode == "eval":
        start = timer()
        mc_result = mc.simulate()
        stop = timer()
        mc_result.print_stats()
        mc_result.save()

        logging.info("==== Acceptance prob =======")
        logging.info("Acceptance probability: {}".format(
            mc_result.get_obs_mean("acceptance_prob")))
        logging.info("============================")
    elif args.mode == "minimize" or args.mode == "min":
        logging.info("====== MINIMIZER INFO ======")
        logging.info("Max Iterations: {}".format(args.maxiter))
        logging.info("Method: {}".format(args.method))
        logging.info("============================")

        minimizer = Minimizer(mc)
        #Set the parameters of the minimizer according to the command line
        minimizer.method = args.method
        minimizer.max_it = args.maxiter

        start = timer()
        result = minimizer.minimize()
        stop = timer()
        print(result)
        minimizer.save()

    logging.info("========== TIME ============")
    logging.info("The simulation took {}s".format(stop-start))
    logging.info("============================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""Possible modes: eval, minimize (min). Possible logging levels are critical, error, warning, info, debug."""
    )
    parser.add_argument("mode", type=str, help="Mode of the program")
    parser.add_argument(
        "L", type=int, help="Size of the square system (one side)")
    parser.add_argument("--g2", type=float, default=1.0,
                        help="coupling constant")
    parser.add_argument("--g_mag", type=float, help="coupling constant")
    parser.add_argument("--g_gm", type=float, default=0.0,
                        help="gauge matter coupling")
    parser.add_argument("--seed", type=int, help="Seed for the MC simulation")
    parser.add_argument("--warmup_steps", type=int,
                        default=int(1e5), help="Number of warmup steps")
    parser.add_argument("--meas_steps", type=int,
                        default=int(1e4), help="Number of run steps")
    parser.add_argument("--level", default="info", help="logging level")
    parser.add_argument("--binsize", default=1, type=int,
                        help="Binsize used in the MC computation")
    parser.add_argument("--y", default=0.5, type=float, help="initial y parameter")
    parser.add_argument("--z", default=0.5, type=float, help="initial z parameter")
    parser.add_argument("--t", default=0.0, type=float, help="initial t parameter: coupling to physical fermions")
    #Arguments for the minimizer
    parser.add_argument("--method", type=str,
                        default="custom", help="Minimization method")
    parser.add_argument("--maxiter", type=int, default=100,
                        help="Number of steps for the minimizer (if custom is used)")
    #Arguments for ray
    parser.add_argument("--nrunner", type=int, default=0,
                        help="Number of parallel MC runners")
    args = parser.parse_args()

    main()
