import os
import sys
import numpy as np

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from differential_heatmap import grad_heatmap

pd.options.mode.use_inf_as_na = True


def _draw_heatmap(*args, **kwargs):
    data = kwargs.pop("data")
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    ax = sns.heatmap(
        d,
        xticklabels=np.round(d.columns.tolist(), 2),
        yticklabels=np.round(d.index.tolist(), 2),
        **kwargs,
    )
    ax.invert_yaxis()


def is_3d(df):
    if "nz" in df.columns:
        nz = df["nz"].unique()[0]
        return nz > 1
    else:
        return False


def mask_impossible_values(df):
    df.loc[
        (np.isclose(abs(df.y + df.z), 1.0)) | (np.isclose(abs(df.y - df.z), 1.0)),
        "value",
    ] = np.nan


def plot_heatmap(df, obs, vmin=None, vmax=None, gradients=False):
    df_obs = df.loc[df["name"] == obs]
    df_obs.reset_index(inplace=True, drop=True)
    # if not is_3d(df_obs):
    # mask_impossible_values(df_obs)
    value_data = df_obs["mean"][df_obs["mean"].notnull()]
    if vmax is not None:
        maxval = float(vmax)
    else:
        maxval = float(value_data.max())

    if vmin is not None:
        minval = float(vmin)
    else:
        minval = float(value_data.min())
    print(f"Maximal value of {obs}: {value_data.max()}")
    print(f"Minimal value of {obs}: {value_data.min()}")
    plt.clf()

    if is_3d(df):
        fg = sns.FacetGrid(df_obs, col="z", col_wrap=3)
        fg.map_dataframe(_draw_heatmap, "x", "y", "mean", vmin=minval, vmax=maxval)
        fg.set_yticklabels(rotation=0)
        fg.set_xticklabels(rotation=90, visible=True)
        fig = fg.fig
    else:
        # The 1e6 is a dirty fix because the heatmap cannot deal with NaN data anymore
        d = df_obs.pivot_table(index="z", columns="y", values="mean", fill_value=1e6)
        darr = np.asarray(d.values)
        darr = darr.astype("float64")
        if gradients:
            im, cbar = grad_heatmap(sorted(df_obs.y.unique()), sorted(df_obs.z.unique()), darr)
            plt.xlabel("y")
            plt.ylabel("z")
            fig = plt.gcf()
        else:
            sns_heatmap = sns.heatmap(
                darr,
                xticklabels=np.round(d.columns.tolist(), 2),
                yticklabels=np.round(d.index.tolist(), 2),
                vmin=minval,
                vmax=maxval,
            )
            sns_heatmap.invert_yaxis()
            sns_heatmap.set_yticklabels(sns_heatmap.get_yticklabels(), rotation=0)
            sns_heatmap.set_xticklabels(sns_heatmap.get_xticklabels(), rotation=90)
            # This is a fix for matplotlib 3.1.1
            bottom, top = sns_heatmap.get_ylim()
            sns_heatmap.set_ylim(bottom - 0.5, top + 0.5)
            fig = sns_heatmap.get_figure()
    if not args.notitle:
        fig.subplots_adjust(top=0.9)
        fig.suptitle(obs)
    # plt.tight_layout()
    if gradients:
        filename = f"{obs}_grad.pdf"
    else:
        filename = f"{obs}.pdf"
    fig.savefig(filename, transparent=True)
    plt.close(fig)


def main(args):
    dfvec = []
    # Collect all the dataframes from different files into one dataframe
    for fname in args.fnames:
        if os.path.isfile(fname):
            tmp = pd.read_pickle(fname)
            dfvec.append(tmp)
        else:
            print(f"Could not open '{fname}'. Skipping file.", file=sys.stderr)
    df = pd.concat(dfvec)

    # Rename the dataframe columns 'val' to mean for the exact results
    if "val" in df.columns:
        df.rename(columns={"val": "mean"}, inplace=True)

    # Filter out illegal data
    df = df.dropna()

    # Filter out all the list-valued rows (gradients)
    df = df[df["mean"].apply(lambda x: not isinstance(x, list) and not isinstance(x, np.ndarray))]
    df = df.convert_dtypes()

    # Make explicit t, y and z columns
    if "y" not in df.columns:
        df["y"] = df["paramvec"].apply(lambda x: np.squeeze(x[:, 1]))
    if "z" not in df.columns:
        df["z"] = df["paramvec"].apply(lambda x: np.squeeze(x[:, 2]))
    if "t" not in df.columns:
        df["t"] = df["paramvec"].apply(lambda x: np.squeeze(x[:, 0]))

    print(df)

    obsverablevec = df.name.unique()
    if args.list:
        print("\n".join(sorted(obsverablevec)))
        sys.exit(0)
    if args.obs is not None:
        if args.obs in obsverablevec:
            plot_heatmap(df, args.obs, vmin=args.vmin, vmax=args.vmax, gradients=args.gradients)
        else:
            print(f"Observable '{args.obs}' has not been measured", file=sys.stderr)
    else:
        for obs in obsverablevec:
            plot_heatmap(df, obs, vmin=args.vmin, vmax=args.vmax, gradients=args.gradients)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fnames", nargs="+", help="Filenames")
    parser.add_argument("--vmax", default=None)
    parser.add_argument("--vmin", default=None)
    parser.add_argument("--obs", default=None)
    parser.add_argument("--gradients", dest="gradients", default=False, action="store_true")
    parser.add_argument("--list", dest="list", default=False, action="store_true")
    parser.add_argument("--no-title", dest="notitle", default=False, action="store_true")

    args = parser.parse_args()
    main(args)
