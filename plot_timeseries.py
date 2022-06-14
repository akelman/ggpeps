"""This script plots the timeseries of given observable.
"""
import matplotlib.pyplot as plt
import numpy as np
import gzip
import os,sys
import pickle
from ggpeps import utils

def main():
    #Collect all dataframes into a single one
    obsvec=[]
    for fname in args.fnames:
        if os.path.isfile(fname):
            with gzip.open(fname,"rb") as infile:
                data=pickle.load(infile)
                obsdict=data['mc'].obsdict
                try:
                    obsvec.append(obsdict[args.obs].get_timeseries())
                except KeyError:
                    print("The observable is not stored in file '{}'".format(fname))
        else:
            print("File '{}' not found. Continuing anyway.".format(fname),file=sys.stderr)

    # Plot the timeseries
    f,axvec=plt.subplots(2,1)
    for obs in obsvec:
        axvec[0].plot(np.real(obs),'o')
        autocorr=np.abs(utils.autocorr_fft(np.real(obs)))
        axvec[1].plot(autocorr,'o')
    axvec[0].set_xlabel(r"$\tau$")
    axvec[0].set_ylabel(args.obs)
    axvec[1].set_xlabel(r"$\tau$")
    axvec[1].set_ylabel("Autocorrelation")
    axvec[1].set_yscale('log')
    plt.show()

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fnames", nargs="+", help="Files to be loaded")
    parser.add_argument("--obs", type=str, default="energy",
                        help="Observable to be plotted")
    args=parser.parse_args()

    main()
