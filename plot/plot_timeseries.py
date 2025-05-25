"""This script plots the timeseries of given observable."""

import os
import sys
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt

from ggpeps import utils


def main(args, save_path=None):
    # Collect all dataframes into a single one
    obsvec = []
    for fname in args.fnames:
        if os.path.isfile(fname):
            with gzip.open(fname, "rb") as infile:
                data = pickle.load(infile)
                obsdict = data["mc"].obsdict
                try:
                    obsvec.append(obsdict[args.obs].get_timeseries())
                except KeyError:
                    print(f"The observable is not stored in file '{fname}'")
        else:
            print(f"File '{fname}' not found. Continuing anyway.", file=sys.stderr)

    # Plot the timeseries
    f, axvec = plt.subplots(2, 1, figsize=(20, 13))
    for obs in obsvec:
        axvec[0].plot(np.real(obs), "o")
        autocorr = np.abs(utils.autocorr_fft(np.real(obs)))
        axvec[1].plot(autocorr, "o")
    axvec[0].set_xlabel(r"$\tau$")
    axvec[0].set_ylabel(args.obs)
    axvec[1].set_xlabel(r"$\tau$")
    axvec[1].set_ylabel("Autocorrelation")
    axvec[1].set_yscale("log")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":

    # import argparse

    # parser = argparse.ArgumentParser(
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    # )

    # parser.add_argument("--fnames", nargs="+", help="Files to be loaded")
    # parser.add_argument(
    #     "--obs", type=str, default="energy", help="Observable to be plotted"
    # )

    # args = parser.parse_args()

    # main(args)
    import glob
    import os
    import argparse

    base_dir = r"G:\My Drive\Research\MC\gauge_fixing_chess\g_2.5_el_1.2500_mag_0.2000_int_1.0_mass_1.0\c2"
    L_vals = [2, 4, 6]
    c_vals = ["c2", "c", "F", "T"]

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
        parser.add_argument("--fnames", nargs="+", help="MC pickle or txt file")
        parser.add_argument("--obs", type=str, default="energy", help="Observable")
        parser.add_argument("--show", type=bool, default=False, help="Display graph")

        args = parser.parse_args(
            args=[
                "--fname",
                *fnames,
                "--obs",
                "energy",
            ]
        )

        print(f"Running main for L={L}")
        main(args, save_path=os.path.join(base_dir, f"timeseries_plot_L_{L}.pdf"))
