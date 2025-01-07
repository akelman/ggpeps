"""
As arguments this file receives the summary pkl file and the log file from a run in debug mode (only debug mode!).

This file plots three plots analysing the eom (error of mean - computed with autocorrelation and rebinning) - 
Dynamical mean of observable as a function of step number, EOM as a funcion of step number and EOM as a function of time.
"""

import os
import re
import sys
from ggpeps import utils
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime


import gzip


def main(args, save_path=None):
    f, axvec = plt.subplots(3, 1, figsize=(20, 13))
    for i in range(len(args.pkl_fname)):
        if os.path.isfile(args.pkl_fname[i]) and os.path.isfile(args.log_fname[i]):
            pkl_basename, pkl_ext = os.path.splitext(args.pkl_fname[i])
            log_basename, log_ext = os.path.splitext(args.log_fname[i])
            if pkl_ext == ".gz":
                with gzip.open(args.pkl_fname[i], "rb") as infile:
                    dumpobj = pickle.load(infile)
                    obsvec = np.asarray(
                        dumpobj["mc"].obsdict[args.obs].get_timeseries()
                    )
                    warmup_steps = dumpobj["mc"].cfg.warmup_steps
            else:
                print(f"Unkown file type {pkl_ext}. Aborting.", file=sys.stderr)
                sys.exit(1)
            if log_ext == ".log":
                with open(args.log_fname[i], "r") as infile:
                    content = infile.read()
                    # Define a regular expression to extract the date, time, and run number
                    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[DEBUG\] Run: (\d+)"
                    matches = re.findall(pattern, content)
                    start_time = datetime.strptime(
                        matches[0][0], "%Y-%m-%d %H:%M:%S,%f"
                    )
                    time = []
                    step_numbers = []
                    for match in matches:
                        timestamp = datetime.strptime(match[0], "%Y-%m-%d %H:%M:%S,%f")
                        time.append((timestamp - start_time).total_seconds())
                        step_numbers.append(int(match[1]) - warmup_steps)
            else:
                print(f"Unkown file type {log_ext}. Aborting.", file=sys.stderr)
                sys.exit(1)
            if (
                "grad" in args.obs
                and args.grad_ind is not None
                and args.layer_num is not None
            ):
                # if it is a gradient, we plot the graph just for the index.
                obsvec = obsvec[:, args.layer_num, args.grad_ind]
            dyn_mean, dyn_eom = compute_dynamic_eom_mean(obsvec, step_numbers)

            axvec[0].plot(step_numbers, dyn_mean, "o", label=args.pkl_fname[i])
            axvec[1].plot(time, dyn_eom, "o")
            axvec[2].plot(step_numbers, dyn_eom, "o")
        else:
            print(
                f"Files '{args.pkl_fname[i]}' or '{args.log_fname[i]}' not found.",
                file=sys.stderr,
            )

    axvec[0].set_ylabel(f"Dynamical Mean {args.obs}")
    axvec[0].legend()
    axvec[0].set_xlabel("step number")
    axvec[1].set_ylabel(f"Dynamical EOM {args.obs}")
    axvec[1].set_xlabel(f"time[sec]")
    axvec[1].set_yscale("log")
    axvec[1].set_xscale("log")
    axvec[2].set_ylabel(f"Dynamical EOM {args.obs}")
    axvec[2].set_yscale("log")
    axvec[2].set_xscale("log")
    axvec[2].set_xlabel(f"step number")

    # f.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def compute_dynamic_eom_mean(obsvec, step_numbers):
    """Compute dynamical mean and dynamical eom, i.e. mean and eom up to particular step number."""
    dyn_eom = []
    dyn_mean = []
    eom, decay_time = utils.autocorr_rebin_eom(obsvec)
    for step in step_numbers:
        dyn_array = obsvec[0 : step + 1]
        num_of_bins = step // decay_time
        mean = np.mean(dyn_array)
        if num_of_bins == 0:
            eom = utils.rebin_eom(dyn_array, 1)
        else:
            eom = utils.rebin_eom(dyn_array, num_of_bins)
        dyn_eom.append(eom)
        dyn_mean.append(mean)
    return dyn_mean, dyn_eom


if __name__ == "__main__":
    import glob
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_fname", nargs="+", help="MC pickle file")
    parser.add_argument("--obs", type=str, default="energy", help="Observable")
    parser.add_argument(
        "--grad_ind", type=int, default=None, help="Gradient index (default: 0)"
    )
    parser.add_argument(
        "--layer_num",
        type=int,
        default=None,
        help="Layer number - when calculating gradient",
    )
    parser.add_argument("--log_fname", nargs="+", help="MC log file - on debug mode")

    args = parser.parse_args()

    main(args)
