import os
import re
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ggpeps import utils


def parse_observables(observables: list[str]):
    """Given a list of strings, parse them into observable names and indices.
    The indices are used when the observable is an array.

    Example input: ["energy", "occupations[0,1]", "magnetization[2]"]
    Example outplut: [("energy", None), ("occupations", (0,1)), ("magnetization", (2,))]

    Args:
        observables (list[str]): observables to parse

    Returns:
        list[(str, tuple[int])]: parsed observables
    """
    pattern = re.compile(r"([a-zA-Z0-9_]+)(\[(.*?)\])?")  # captures name + inside brackets

    parsed_observables = []

    for item in observables:
        name, _, inside = pattern.fullmatch(item).groups()

        if inside is not None:
            idx = tuple([int(i) for i in inside.split(",")])
        else:
            idx = None
        parsed_observables.append((name, idx))
    return parsed_observables


def len_arr(x):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return len(x)
    else:
        return 1


def main(args):
    dfvec = []
    if args.ec is not None:
        for fname in args.ec:
            if os.path.isfile(fname):
                df_tmp = pd.read_pickle(fname)
                df_tmp["type"] = "EC"
                df_tmp["err"] = np.nan  # Append an empty column to make merges work
                if "nlayer" not in df_tmp.columns:
                    df_tmp["nlayer"] = utils.fname2nlayer(fname)
                if "ncopy" not in df_tmp.columns:
                    df_tmp["ncopy"] = utils.fname2ncopy(fname)
                dfvec.append(df_tmp)
    if args.mc is not None:
        for fname in args.mc:
            if os.path.isfile(fname):
                df_tmp = pd.read_pickle(fname)
                df_tmp["type"] = "MC"
                if "nlayer" not in df_tmp.columns:
                    df_tmp["nlayer"] = utils.fname2nlayer(fname)
                if "ncopy" not in df_tmp.columns:
                    df_tmp["ncopy"] = utils.fname2ncopy(fname)
                dfvec.append(df_tmp)
    df_mc_ec = pd.concat(dfvec)

    # Enrich dataset
    df_mc_ec["L"] = df_mc_ec["nx"].astype("str") + "x" + df_mc_ec["ny"].astype("str")
    df_mc_ec.rename(columns={"g2_el": "g_el", "g2_mag": "g_mag", "g2": "g"}, inplace=True)
    obsnamevec = df_mc_ec.name.unique()

    if "g" not in df_mc_ec.columns:
        df_mc_ec["g"] = df_mc_ec.g_el * 2
    if "g_mag" not in df_mc_ec.columns:
        df_mc_ec["g_mag"] = 1 / (df_tmp.g_el * 4)

    # Get exact data
    if args.exact is not None and os.path.isfile(args.exact):
        df_exact = pd.read_pickle(args.exact)
        df_exact["L"] = df_exact["nx"].astype("str") + "x" + df_exact["ny"].astype("str")
        df_exact["type"] = "ED"
        # Add numbers to the grouping columns to enable grouping
        df_exact["nlayer"] = -1
        df_exact["ncopy"] = -1

        # Adapt the naming convention between the ED and the MC/EC data
        df_exact.rename(columns={"g2_ham": "g", "value": "mean"}, inplace=True)
        df_exact.drop(columns=["nz", "gauge"], inplace=True)
        if "g_el" not in df_exact.columns:  # the next four lines should be handled in a more robust way
            df_exact["g_el"] = df_exact.g / 2
        if "g" not in df_exact.columns:
            df_exact["g"] = df_exact.g_el * 2

        df = pd.concat([df_mc_ec, df_exact])

        # Compute the differences
        df_mc_ec_approx = df_mc_ec.copy()
        df_ed_approx = df_exact.copy()
        df_mc_ec_approx.g = np.round(df_mc_ec_approx.g, decimals=3)
        df_ed_approx.g = np.round(df_ed_approx.g, decimals=3)
        df_merged = pd.merge(
            df_mc_ec_approx,
            df_ed_approx,
            on=["g", "L", "name", "nx", "ny"],
            suffixes=("_mc_ec", "_ed"),
        )
        df_merged["diff"] = df_merged["mean_mc_ec"] - df_merged["mean_ed"]
        df_diff = df_merged[
            [
                "name",
                "g",
                "ncopy_mc_ec",
                "nlayer_mc_ec",
                "L",
                "diff",
                "type_mc_ec",
                "err",
            ]
        ].copy()
        df_diff.rename(
            columns={
                "nlayer_mc_ec": "nlayer",
                "ncopy_mc_ec": "ncopy",
                "type_mc_ec": "type",
            },
            inplace=True,
        )
    else:
        df = df_mc_ec
        df_diff = None
        if args.diff:
            print(f"File {args.exact} not found. We need it for the differences. Aborting.")
            sys.exit(1)
        else:
            print(f"File {args.exact} not found. Skipping.")

    parsed_obs = parse_observables(args.obs)
    palette = sns.color_palette("husl", n_colors=len(args.obs))
    observable_colors = dict(zip(parsed_obs, palette))

    # Plot
    f, ax = plt.subplots(1, 1)
    for parsed_ob in parsed_obs:
        obs, idx = parsed_ob
        if obs in obsnamevec:
            df_filtered = df[df["name"] == obs]
            df_filtered.reset_index(drop=True, inplace=True)

            if args.diff:
                df_filtered = df_diff[df_diff["name"] == obs]
                df_filtered.reset_index(drop=True, inplace=True)

            for name, group in df_filtered.groupby(["type", "L", "nlayer", "ncopy"]):

                type_, L, nlayer, ncopy = name

                # Get x axis values
                if isinstance(group[args.xaxis].iloc[0], np.ndarray):
                    # Handle case where chosen values are an array
                    xaxis_values = group[args.xaxis].apply(lambda x: x[args.xaxis_ind])
                else:
                    xaxis_values = group[args.xaxis]

                # Get y axis values
                if args.diff:
                    yaxis_values = group["diff"]
                else:
                    yaxis_values = group["mean"]

                if isinstance(yaxis_values.iloc[0], np.ndarray):
                    # Handle case where chosen values are an array
                    yaxis_values = yaxis_values.apply(lambda x: x[*idx])

                # Set data label
                if not args.diff and isinstance(group["mean"].iloc[0], np.ndarray):
                    data_label = f"{type_}, obs={obs}, inds={idx}, L={L}"
                else:
                    data_label = f"{type_}, obs={obs}, L={L}"

                # Show errors for MC
                if type_ == "MC":
                    error = group["err"]  # this will not work for array observables
                else:
                    error = None

                # Set marker
                marker_fmt = "o"
                if type_ == "MC":
                    marker_fmt = "x"

                # Plot
                if type_ == "ED":
                    ax.plot(
                        xaxis_values,
                        yaxis_values,
                        label=data_label,
                        c=observable_colors[parsed_ob],
                    )
                else:
                    ax.errorbar(
                        xaxis_values,
                        yaxis_values,
                        fmt=marker_fmt,
                        yerr=error,
                        label=data_label,
                        c=observable_colors[parsed_ob],
                    )

    # Set axis properties
    if args.logx:
        ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")
    ax.set_xlabel(args.xaxis, fontsize=10)
    if args.diff:
        ax.set_ylabel("Value - ED", fontsize=10)
    else:
        ax.set_ylabel("Value", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_title(args.title)
    f.tight_layout()

    # Output
    if not args.no_save:
        if args.diff:
            f.savefig(f"summary_diff_{'-'.join(args.obs)}.pdf")
        else:
            f.savefig(f"summary_{'-'.join(args.obs)}.pdf")
    if args.show:
        plt.show()

    return


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ed", "--exact", help="ED data", dest="exact")
    parser.add_argument("--mc", nargs="+", help="MC data")
    parser.add_argument("--ec", nargs="+", help="EC data")
    parser.add_argument(
        "--diff",
        action="store_true",
        default=False,
        help="Plot the difference to the exact results",
    )
    parser.add_argument(
        "--logx",
        action="store_true",
        default=False,
        help="Use logarithmic scaling for x axis",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        default=False,
        help="Use logarithmic scaling for y axis",
    )
    parser.add_argument("--show", action="store_true", default=False, help="Show the plot")
    parser.add_argument("--no-save", action="store_true", default=False, help="Do not save the plot")
    parser.add_argument("--xaxis", type=str, default="g_el", help="Quantity to be plotted on the x axis")
    parser.add_argument(
        "--xaxis_ind",
        type=int,
        default="0",
        help="If --xaxis quantity is an array, use this index",
    )
    parser.add_argument(
        "--obs",
        type=str,
        nargs="+",
        default=["energy"],
        help="Observables to plot. Each observable can simple name the observable, "
        "or if the observable represents an array, be in the form obs_name[inds], "
        "where the appropriate inds will be plotted.",
    )

    parser.add_argument("--title", type=str, default="", help="Title for plot", dest="title")

    args = parser.parse_args()

    main(args)
