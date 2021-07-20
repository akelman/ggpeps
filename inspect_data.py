"""Print a summary of a data file. It can be a summary_*.pkl or a data_*.pkl.gz file """
import pickle
import pandas as pd
import numpy as np
import os, sys
import gzip
import utils

def print_output_mc_dump(version,mc):
    obsdict=mc.obsdict
    cfg=mc.cfg
    print("==================================== Data summary =====================================")
    print("\nVersion: {}\n".format(version))
    print("--------------------------------    Config   --------------------------------")
    print(cfg)
    print("-----------------------------------------------------------------------------\n")
    print("-------------------------------- Observables --------------------------------")
    keys=obsdict.keys()
    numentries=[str(len(obsdict[key])) for key in obsdict]
    data=list(zip(keys,numentries))
    utils.print_columns([["Name","# Measurements"]]+data,header=True)
    print("-----------------------------------------------------------------------------\n")
    print("=======================================================================================")

def main():
    if args.fname is not None and os.path.isfile(args.fname):
        fname_base=os.path.basename(args.fname)
        name,ext=os.path.splitext(fname_base)
        if ext == ".gz" and name.startswith("data"):
            #We are dealing with a full simulation file
            with gzip.open(args.fname,"rb") as infile:
                data=pickle.load(infile)
                version=data["version"]
                mc=data["mc"]
                print_output_mc_dump(version,mc)
        elif ext == ".pkl": 
            if name.startswith("summary"):
                # We are dealing with a summary file
                df=pd.read_pickle(args.fname)
                print(df)
            elif name.startswith("result_min"):
                with open(args.fname, "rb") as infile:
                    # We are dealing with a minimization result
                    data=pickle.load(infile)
                    print(data)
            else:
                #We don't know what to do
                print("Invalid file '{}'".format(args.fname),file=sys.stderr)
        elif ext== ".npy":
            # We are looking at a xivec or alphavec file, load it and display it
            vec_import=np.load(args.fname)
            print("Length: ({})".format(len(vec_import)))
            print(vec_import)
        else:
            #We don't know what to do
            print("Invalid file '{}'".format(args.fname),file=sys.stderr)
    else:
        print("File '{}' does not exist".format(args.fname),file=sys.stderr)

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("fname",type=str,help="filename to inspect")
    args=parser.parse_args()

    main()