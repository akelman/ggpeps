import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import re

def fname2L(fname):
    pattern=r"(?<=L_)[\d]*"
    result = re.search(pattern, fname)
    return int(result.group(0))

def fname2g2(fname):
    """Extract the electric coupling from a filename"""
    pattern=r"(?<=gel_)[\d]*.[\d]"
    result = re.search(pattern, fname)
    return float(result.group(0))

def res2df(fname,res):
    resdict={}
    resdict["parametervec"]=res.parametervec
    resdict["energygrad"]=res.energygrad
    resdict["method"]=res.method
    resdict["energy"]=res.value
    resdict["converged"]=res.converged
    resdict["g2"]=fname2g2(fname)
    resdict["L"]=fname2L(fname)
    return pd.DataFrame(resdict)

def main(args):
    dfvec=[]
    for fname in args.fnames:
        if os.path.isfile(fname):
            minimizer_result=pd.read_pickle(fname)
            df=res2df(fname, minimizer_result)
            dfvec.append(df)
    df=pd.concat(dfvec) 

    f,ax=plt.subplots(1,1)
    for name, group in df.groupby("L"):
        ax.plot(group["g2"],group["energy"],'o',label="L={:02d}".format(name))
    ax.set_xlabel("$g^2$")
    ax.set_ylabel("Energy")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fnames", nargs="+", help="Filenames")
    parser.add_argument("--obs", type=str, default="energy", help="Observable")

    args = parser.parse_args()

    main(args)
