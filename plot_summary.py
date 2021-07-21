import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys


def main(args):
    dfvec = []
    for fname in args.fnames:
        if os.path.isfile(fname):
            df = pd.read_pickle(fname)
            dfvec.append(df)
    df = pd.concat(dfvec)
    df["L"] = df["nx"].astype("str") + "-" + df["ny"].astype("str")
    obsnamevec = df.name.unique()

    if args.exact is not None and os.path.isfile(args.exact):
        df_exact=pd.read_pickle(args.exact)
        df_exact["L"] = df_exact["nx"].astype("str") + "-" + df_exact["ny"].astype("str")

    f,ax=plt.subplots(1,1,figsize=(4.14,2.66))
    for obs in args.obs:
        if obs in obsnamevec:
            df_filtered=df[df.name==obs]
            df_filtered.reset_index(drop=True, inplace=True)

            for name, group in df_filtered.groupby("L"):
                ax.plot(group["g_el"],
                        group["mean"],
                        'o',
                        label="EC, {}, L={}".format(obs,name))

            # We can add the ED data to the plot to compare the curves
            if args.exact is not None and os.path.isfile(args.exact):
                df_exact_filtered=df_exact[df_exact.name==obs]
                for name, group in df_exact_filtered.groupby("L"):
                    ax.plot(df_exact_filtered["g_ham"],df_exact_filtered["value"],"-",label="ED, {}, L={}".format(obs,name))

    ax.set_xlabel("$g^2$", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig("summary_{}.pdf".format("-".join(args.obs)))
    if args.show:
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fnames", nargs="+", help="Filenames")
    parser.add_argument("--exact", help="ED data")
    parser.add_argument("--show", action="store_true", default=False, help="Show the plot")
    parser.add_argument("--obs",
                        type=str,
                        nargs="+",
                        default="energy",
                        help="Observable")

    args = parser.parse_args()

    main(args)
