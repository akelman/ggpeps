import pandas as pd
import sys
import os
from tqdm import tqdm

def main(args):
    if args.fnames is not None and len(args.fnames) > 0:
        for fname in tqdm(args.fnames):
            if os.path.isfile(fname):
                df = pd.read_pickle(fname)
                df.to_pickle(fname, protocol=args.pickle_protocol)
            else:
                print(f"File '{fname}' not found. Skipping.", file = sys.stderr)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fnames", nargs="+", help="Input filename")
    parser.add_argument("--pickle-protocol", default = 4)

    args = parser.parse_args()
    main(args)