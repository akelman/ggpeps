import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from ggpeps import utils

def len_arr(x):
    if isinstance(x,list) or isinstance(x,np.ndarray):
        return len(x)
    else:
        return 1

def main(args):
    dfvec = []
    if args.ec is not None:
        for fname in args.ec:
            if os.path.isfile(fname):
                df = pd.read_pickle(fname)
                df['type'] = "EC"
                if not "nlayer" in df.columns:
                    df["nlayer"] = utils.fname2nlayer(fname)
                if not "ncopy" in df.columns:
                    df["ncopy"] = utils.fname2ncopy(fname)
                dfvec.append(df)
    if args.mc is not None:
        for fname in args.mc:
            if os.path.isfile(fname):
                df = pd.read_pickle(fname)
                df['type'] = "MC"
                if not "nlayer" in df.columns:
                    df["nlayer"] = utils.fname2nlayer(fname)
                if not "ncopy" in df.columns:
                    df["ncopy"] = utils.fname2ncopy(fname)
                dfvec.append(df)
    df = pd.concat(dfvec)

    #Enrich dataset
    df["L"] = df["nx"].astype("str") + "-" + df["ny"].astype("str")
    df.rename(columns={"g_el":"g2_el", "g_mag":"g2_mag"},inplace=True)
    obsnamevec = df.name.unique()

    if args.exact is not None and os.path.isfile(args.exact):
        df_exact = pd.read_pickle(args.exact)
        df_exact["L"] = df_exact["nx"].astype("str") + "-" + df_exact["ny"].astype("str")

    #f,ax=plt.subplots(1,1,figsize=(4.14,2.66))
    f,ax=plt.subplots(1,1)
    for obs in args.obs:
        if obs in obsnamevec:
            df_filtered = df[df.name == obs]
            df_filtered.reset_index(drop=True, inplace=True)

            for name, group in df_filtered.groupby(["type","L","nlayer", "ncopy"]):
                type, L, nlayer, ncopy = name
                # We calculate the coupling g^2 from g2_el
                ax.plot(group["g2_el"]*2,
                        group["mean"],
                        'o',
                        label="{}, {}, L={}, nc={}, nl={}".format(type,obs,L,ncopy,nlayer))

            # We can add the ED data to the plot to compare the curves
            if args.exact is not None:
                if os.path.isfile(args.exact):
                    df_exact_filtered=df_exact[df_exact.name==obs]
                    for name, group in df_exact_filtered.groupby("L"):
                        ax.plot(df_exact_filtered["g2_ham"],df_exact_filtered["value"],"-",label="ED, {}, L={}".format(obs,name))
                else:
                    print(
                        f"File {args.exact} not found. Continuing without exact data.", file=sys.stderr)
    if args.logx:
        ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")
    ax.set_xlabel("$g^2$", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.legend(fontsize=8)
    f.tight_layout()
    if not args.no_save:
        f.savefig("summary_{}.pdf".format("-".join(args.obs)))
    if args.show:
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", help="ED data")
    parser.add_argument("--mc", nargs="+", help="MC data")
    parser.add_argument("--ec", nargs="+", help="EC data")
    parser.add_argument("--show", action="store_true", default=False, help="Show the plot")
    parser.add_argument("--no-save", action="store_true", default=False, help="Do not save the plot")
    parser.add_argument("--logx", action="store_true", default=False, help="Use logarithmic scaling for x axis")
    parser.add_argument("--logy", action="store_true", default=False, help="Use logarithmix scaling for y axis")
    parser.add_argument("--obs",
                        type=str,
                        nargs="+",
                        default=["energy"],
                        help="Observable")

    args = parser.parse_args()

    main(args)
