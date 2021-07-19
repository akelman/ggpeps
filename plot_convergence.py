"""Plots the convergence of a given MC observable.
The value (and its error) is plotted against multiple MC runs with varying parameters.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os,sys

def args2fname():
    fname="convergence_{}.pdf".format(args.obs)
    return fname

def main():
    #Collect all dataframes into a single one
    dfvec=[]
    for fname in args.fnames:
        if os.path.isfile(fname):
            dfvec.append(pd.read_pickle(fname))
        else:
            print("File '{}' not found. Continuing anyway.".format(fname),file=sys.stderr)
    df=pd.concat(dfvec,ignore_index=True,sort=False)
    if not df.empty:
        #Plot the convergence
        df_filtered=df[df.name==args.obs]
        print(df_filtered)
        if not df_filtered.empty:
            f,ax=plt.subplots(1,1,figsize=(4.14,2.66))
            ax.set_xscale('log')
            ax.errorbar(df_filtered['meas_steps'],np.real(df_filtered['mean']),yerr=df_filtered['err'],fmt='o')
            ax.set_xlabel(r"$\tau$", fontsize=10)
            ax.set_ylabel(args.obs, fontsize=10)

            #Plot the exact value if given via exact contraction
            df_exact=df_filtered[df_filtered['warmup_steps'].isna()]
            if len(df_exact)>0:
                #Print the exact contraction value
                ax.axhline(np.real(df_exact['mean']))

            if args.show:
                plt.show()
            f.tight_layout()
            fname=args2fname()
            f.savefig(fname,bbox_inches='tight')
        else:
            print("No data for observable '{}'.".format(args.obs),file=sys.stderr)
    else:
        print("No data found in summary files.",file=sys.stderr)

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fnames", nargs="+", help="Files to be loaded")
    parser.add_argument("--obs", type=str, default="energy",
                        help="Observable to be plotted")
    parser.add_argument("--show", default=False, action="store_true", help="Show plot interactively")
    args=parser.parse_args()

    main()
