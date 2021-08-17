import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import re
import pickle
import utils
import gzip

def main(args):
    f,axvec=plt.subplots(4,1)
    for fname in args.fname:
        if os.path.isfile(fname):
            basename,ext=os.path.splitext(fname)
            if ext == ".gz":
                with gzip.open(fname,"rb") as infile:
                    dumpobj=pickle.load(infile)
                    obsvec=np.asarray(dumpobj["mc"].obsdict[args.obs].get_timeseries())
            elif ext==".txt":
                obsvec=np.genfromtxt(fname)
            else:
                print("Unkown file type {}. Aborting.".format(ext),file=sys.stderr)
                sys.exit(1)
            rangevals, meanarr, eomarr, stdarr=utils.rebin_error(obsvec)
            axvec[0].plot(rangevals,meanarr,'o',label=fname)
            axvec[1].plot(rangevals,stdarr,'o')
            axvec[2].plot(rangevals,eomarr,'o')
            axvec[3].plot(np.abs(utils.autocorr_fft(obsvec))[0:200],'o')
        else:
            print("File '{}' not found.".format(fname),file=sys.stderr)
    axvec[1].set_yscale("log")
    axvec[0].set_xscale("log")
    axvec[1].set_xscale("log")
    axvec[2].set_xscale("log")
    axvec[0].set_ylabel("Mean {}".format(args.obs))
    axvec[0].legend()
    axvec[1].set_ylabel("STD {}".format(args.obs))
    axvec[2].set_ylabel("EOM {}".format(args.obs))
    axvec[3].set_ylabel("Autocorrelation {}".format(args.obs))
    axvec[2].set_xlabel("len(bin)")
    axvec[3].set_xlabel(r"$\tau$")
    axvec[3].set_yscale("log")
    #f.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fname",nargs="+", help="MC pickle or txt file")
    parser.add_argument("--obs", type=str, default="energy", help="Observable")

    args = parser.parse_args()

    main(args)
