import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

#from matplotlib import rc,rcParams
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#rc('font',**{'family':'serif','sans-serif':['Palatino']})
#rc('text', usetex=True)
#rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']


def main(args):
    dfvec = []
    for fname in args.fnames:
        if os.path.isfile(fname):
            df = pd.read_pickle(fname)
            dfvec.append(df)
    df = pd.concat(dfvec)
    obsnamevec = df.name.unique()
    if args.obs in obsnamevec:
        df_filtered = df[df.name==args.obs]
        df_filtered.reset_index(drop=True, inplace=True)
        print(df_filtered)

        # The figure size is given in inches.
        # This is exactly a half column of an a4 page in 14 to 9
        f,ax = plt.subplots(1,1,figsize=(4.14,2.66))
        ax.errorbar(range(len(df_filtered)),
                    df_filtered["mean"],
                    fmt="o",
                    yerr=df_filtered["err"])
        ax.set_xlabel("Run", fontsize=10)
        ax.set_ylabel(f"{args.obs}", fontsize=10)
        f.tight_layout()
        f.savefig(f"interrun_convergence_{args.obs}.pdf")
        plt.show()
    else:
        print(f"Observable '{args.obs}' is not in the dataset", file=sys.stderr)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fnames", nargs="+", help="Filenames")
    parser.add_argument("--obs", type=str, default="energy", help="Observable")

    args = parser.parse_args()

    main(args)
