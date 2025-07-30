import os
import pandas as pd
import matplotlib.pyplot as plt

from ggpeps import utils


def res2df(fname, res):
    resdict = {}
    resdict["paramvec"] = res.paramvec
    resdict["energygrad"] = res.energygrad
    resdict["method"] = res.method
    resdict["energy"] = res.value
    resdict["converged"] = res.converged
    resdict["g"] = utils.fname2g(fname)
    resdict["L"] = utils.fname2size(fname)
    return pd.DataFrame(resdict)


def main(args):
    dfvec = []
    for fname in args.fnames:
        if os.path.isfile(fname):
            minimizer_result = pd.read_pickle(fname)
            df = res2df(fname, minimizer_result)
            dfvec.append(df)
    df = pd.concat(dfvec)

    f, ax = plt.subplots(1, 1)
    for name, group in df.groupby("L"):
        ax.plot(group["g2"], group["energy"], "o", label=f"L={name:02d}")
    ax.set_xlabel("$g^2$")
    ax.set_ylabel("Energy")
    ax.legend()
    plt.show()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fnames", nargs="+", help="Filenames")
    parser.add_argument("--obs", type=str, default="energy", help="Observable")

    args = parser.parse_args()

    main(args)
