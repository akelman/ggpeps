import sys
import numpy as np
import itertools as it

from tqdm import tqdm

import exacteval
import lattice as lat
from system import Z2System2D, Z2System2DConfig
from system import Z2System2D2C, Z2System2D2CConfig
from mc import MonteCarloEstimatorConfig, MonteCarloManager


def main(args):
    L = args.L
    g_el = 1
    g_int = 0
    g_mag = 1
    # We are focussing on 2 dimensions for the moment
    lattice = lat.Lattice2D(L, L)

    paramiter = np.linspace(args.pmin, args.pmax, args.nstep)

    if args.ncopy == 1:
        system_cls = Z2System2D
        system_cfg = Z2System2DConfig(lattice, g_el, g_int, g_mag, nlayer=args.nlayer)
        nparams = system_cfg.nvarparams()
        if args.pure_gauge:
            paramproduct = it.product([0], *([paramiter] * (nparams - 1)))
        else:
            paramproduct = it.product(paramiter, repeat=nparams)
    elif args.ncopy == 2:
        system_cls = Z2System2D2C
        system_cfg = Z2System2D2CConfig(lattice, g_el, g_int, g_mag, nlayer=args.nlayer)
        nparams = system_cfg.nvarparams()
        if args.pure_gauge:
            paramproduct = it.product([0], *([paramiter] * 2), [0], *([paramiter] * 6))
        else:
            paramproduct = it.product(paramiter, repeat=nparams)
    else:
        print("Not Implemented: Only 1 or 2 copies are possible", file=sys.stderr)
        sys.exit(1)

    for paramvec in tqdm(paramproduct):
        system_cfg.paramvec = np.array([paramvec])
        system = system_cls(system_cfg)
        if args.exact:
            ex_eval = exacteval.ExactEvaluator(system)
            ex_eval.evaluate()
            ex_eval.save()
        else:
            mc_config = MonteCarloEstimatorConfig()
            mc_config.warmup_steps = args.warmup_steps
            mc_config.meas_steps = args.meas_steps
            if args.seed is not None:
                mc_config.seed = args.seed
            mc_config.compute_grads = False
            mc_mgr = MonteCarloManager(mc_config, system_cls, system_cfg, 0)
            mc_result = mc_mgr.simulate()
            fname_summary = "summary_mc_L_{:02d}-{:02d}_g2el_{:.3f}_int_{:.3f}_g2mag_{:.3f}_t_{}_y_{}_z_{}.pkl".format(
                system_cfg.lattice.nx,
                system_cfg.lattice.ny,
                system_cfg.g2_el,
                system_cfg.g_int,
                system_cfg.g2_mag,
                paramvec[0],
                paramvec[1],
                paramvec[2],
            )
            mc_result.save_summary(fname_summary)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("L", type=int, help="Size of the square system (one side)")

    parser.add_argument("--pmin", default=0.0, type=float, help="Minimal parameter")
    parser.add_argument("--pmax", default=2.0, type=float, help="Maximal parameter")
    parser.add_argument(
        "--pure-gauge",
        action="store_true",
        default=False,
        help="Force the coupling of physical and virtual fermions (t-parameters) to be 0",
    )
    parser.add_argument(
        "--nstep", default=10, type=int, help="Number of steps in the parameter"
    )

    # parser.add_argument("--g2", type=float, default=1.0,
    # help="coupling constant")
    # parser.add_argument("--g_mag", type=float, help="coupling constant")
    # parser.add_argument("--g_int", type=float, default=0.0,
    # help="gauge matter coupling")
    parser.add_argument("--seed", type=int, help="Seed for the MC simulation")
    parser.add_argument(
        "--warmup_steps", type=int, default=int(1e5), help="Number of warmup steps"
    )
    parser.add_argument(
        "--meas_steps", type=int, default=int(1e4), help="Number of run steps"
    )
    parser.add_argument(
        "--nlayer",
        default=1,
        type=int,
        help="Number of PEPS layers for the variational state",
    )
    parser.add_argument(
        "--ncopy", default=1, type=int, help="Number of virtual fermions on the links"
    )
    parser.add_argument(
        "--exact",
        default=False,
        action="store_true",
        help="Use exact contraction instead of MC",
    )

    args = parser.parse_args()
    main(args)
