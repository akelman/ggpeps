import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_data(data:pd.DataFrame, xaxis:str, obs:str, restrictions:dict):

    for key, val in restrictions.items():
        data = data[data[key] == val]

    xvals = data.loc[:, xaxis].values
    yvals = data.loc[:, obs].values 

    return xvals, yvals

def main(args):
    
    f, ax = plt.subplots(1, 1)
    restrictions = {}
    for r in args.restrict:
        key, val = r.split("=")
        restrictions[key] = float(val)
    print(f"Plotting with the restrictions: {restrictions}")

    if args.ec_labels is None:
        args.ec_labels = ['']*len(args.ec)

    if args.ed is not None:
        data = pd.read_csv(args.ed)
        xvals, yvals = extract_data(data, args.xaxis, args.obs, restrictions)

        # reorder
        # this is important when points are connected by lines (as is done for ED data)
        xvals, yvals = zip(*sorted(zip(xvals, yvals)))         #xvals, yvals = map(list, zip(*sorted(zip(xvals, yvals), reverse=True)))

        ax.plot(xvals, yvals, label = f"ED, obs={args.obs}", c="orange")

    if args.ec is not None:
        for ind, ec_data in enumerate(args.ec):
            data = pd.read_csv(ec_data)
            xvals, yvals = extract_data(data, args.xaxis, args.obs, restrictions)
            ax.scatter(xvals, yvals, label = f"EC, obs={args.obs}, {args.ec_labels[ind]}")
    
    if args.mc is not None:
        xvals, yvals = extract_data(args.mc, args.xaxis, args.obs, restrictions)
        ax.errorbar(xvals, yvals, label = f"MC, obs={args.obs}")
        # TODO: add support for errorbars

    if args.logx:
        ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")
    
    ax.legend(fontsize=8)
    ax.set_xlabel(args.xaxis, fontsize=10)
    f.tight_layout()

    #if args.diff:
    #    ax.set_ylabel("Value - ED", fontsize=10)
    #else:
    ax.set_ylabel("Value", fontsize=10)
    
    if not args.no_save:
        f.savefig(f"summary_{'-'.join(args.obs)}.pdf")
    if args.show:
        plt.show()

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ed", "--exact", help="ED data")
    parser.add_argument("--mc", nargs="+", help="MC data")
    parser.add_argument("--ec", nargs="+", help="EC data")
    #parser.add_argument("--diff", action="store_true", default=False, help="Plot the difference to the exact results")
    parser.add_argument("--logx", action="store_true", default=False, help="Use logarithmic scaling for x axis")
    parser.add_argument("--logy", action="store_true", default=False, help="Use logarithmic scaling for y axis")
    parser.add_argument("--show", action="store_true", default=False, help="Show the plot")
    parser.add_argument("--no-save", action="store_true", default=False, help="Do not save the plot")
    parser.add_argument("--xaxis", type=str, default="el", help="Quantity to be plotted on the x axis")
    parser.add_argument("--obs", type=str, nargs="+", default=["energy"], help="Observables to plot")
    
    parser.add_argument("--ec_labels", nargs="+", help="Label tags for EC data")
    parser.add_argument("--restrict", nargs="+", help="Data restrictions - only plot data that has these couplings. Format: key=val, e.g. mass=1")

    args = parser.parse_args()

    main(args)
