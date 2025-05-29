import os
import re
import sys
from ggpeps import utils
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gzip


def main(args):
    f, axvec = plt.subplots(4, 1, figsize=(20, 13))
    for fname in args.fname:
        if os.path.isfile(fname):
            basename, ext = os.path.splitext(fname)
            if ext == ".gz":
                with gzip.open(fname, "rb") as infile:
                    dumpobj = pickle.load(infile)
                    obsvec = np.asarray(
                        dumpobj["mc"].obsdict[args.obs].get_timeseries()
                    )
            elif ext == ".txt":
                obsvec = np.genfromtxt(fname)
            else:
                print(f"Unkown file type {ext}. Aborting.", file=sys.stderr)
                sys.exit(1)
            rangevals, meanarr, eomarr, stdarr = utils.rebin_error(obsvec)
            axvec[0].plot(rangevals, meanarr, "o", label=fname)
            axvec[1].plot(rangevals, stdarr, "o")
            axvec[2].plot(rangevals, eomarr, "o")
            axvec[3].plot(np.abs(utils.autocorr_fft(obsvec))[0:200], "o")
        else:
            print(f"File '{fname}' not found.", file=sys.stderr)

    axvec[1].set_yscale("log")
    axvec[0].set_xscale("log")
    axvec[1].set_xscale("log")
    axvec[2].set_xscale("log")
    axvec[0].set_ylabel(f"Mean {args.obs}")
    axvec[0].legend()
    axvec[1].set_ylabel(f"STD {args.obs}")
    axvec[2].set_ylabel(f"EOM {args.obs}")
    axvec[3].set_ylabel(f"Autocorrelation {args.obs}")
    axvec[2].set_xlabel("len(bin)")
    axvec[3].set_xlabel(r"$\tau$")
    axvec[3].set_yscale("log")
    # f.tight_layout()
    if args.dest:
        plt.savefig(args.dest, dpi=300, bbox_inches="tight")
        plt.close()
    if args.show:
        plt.show()


if __name__ == "__main__":

    import argparse
    import glob

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--fname", nargs="+", help="MC pickle or txt file")
    # parser.add_argument("--obs", type=str, default="energy", help="Observable")
    # parser.add_argument("--show", type=bool, default=False, help="Display graph")
    # parser.add_argument("--dest", type=str, default=None, help="Destination filepath")

    # args = parser.parse_args()

    # main(args)
    base_dir = r"G:\My Drive\Research\MC\gauge_fixing_chess\g_2.5_el_1.2500_mag_0.2000_int_1.0_mass_1.0\comp_with_rows"
    L_vals = [2, 4, 6]
    c_vals = ["1","2","3","4","5","6", "c", "F", "T"]

    for L in L_vals:
        fnames = []
        for c in c_vals:
            folder = os.path.join(base_dir, f"L_{L}_gf_{c}")
            if not os.path.isdir(folder):
                continue
            fnames += glob.glob(os.path.join(folder, "*.pkl.gz*"))
        if not fnames:
            print(f"No files found for L={L}")
            continue

        parser = argparse.ArgumentParser()
        parser.add_argument("--fname", nargs="+", help="MC pickle or txt file")
        parser.add_argument("--obs", type=str, default="energy", help="Observable")
        parser.add_argument("--show", type=bool, default=False, help="Display graph")
        parser.add_argument(
            "--dest", type=str, default=None, help="Destination filepath"
        )

        args = parser.parse_args(
            args=[
                "--fname",
                *fnames,
                "--obs",
                "energy",
                "--dest",
                os.path.join(base_dir, f"binning_plot_L_{L}.pdf"),
            ]
        )

        print(f"Running main for L={L}, saving to {args.dest}")
        main(args)
