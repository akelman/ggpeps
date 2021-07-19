import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def main(args):
    dfvec=[]
    for fname in args.fnames:
        if os.path.isfile(fname):
            df=pd.read_pickle(fname)
            dfvec.append(df)
    df=pd.concat(dfvec) 
    obsnamevec=df.name.unique()
    if args.obs in obsnamevec:
        df_filtered=df[df.name==args.obs]
        df_filtered.reset_index(drop=True, inplace=True)
        print(df_filtered) 

        f,ax=plt.subplots(1,1)
        ax.errorbar(range(len(df_filtered)),df_filtered["mean"],yerr=df_filtered["err"])
        plt.show()
    else:
        print("Observable '{}' is not in the dataset".format(
            args.obs), file=sys.stderr)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fnames", nargs="+", help="Filenames")
    parser.add_argument("--obs", type=str, default="energy", help="Observable")

    args = parser.parse_args()

    main(args)
