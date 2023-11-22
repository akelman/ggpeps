import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def extract_data(data:pd.DataFrame, xaxis:str, obs, restrictions:dict, require_g:bool, get_error:bool = False):

    for key, val in restrictions.items():
        data = data[data[key] == val]
    
    if require_g:
        data = data[data["mag"] == 1/(4*data["el"])]

    xvals = data.loc[:, xaxis].values
    yvals_dict = {}
    yvals_err_dict = {}
    for ob in obs:
        yvals = data.loc[:, ob].values 
        yvals_dict[ob] = yvals

        # Get error
        if get_error:
            ob_error = f"{ob}_error"
            yvals_err_dict[ob] = data.loc[:, ob_error].values 

    return xvals, yvals_dict, yvals_err_dict


def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

def plot_3d(args):
    """Based on: https://matplotlib.org/stable/gallery/mplot3d/polys3d.html
    """
    ax = plt.figure().add_subplot(projection='3d')

    ob = args.obs[0]
    ec_data = args.ec[0]

    verts = []
    data = pd.read_csv(ec_data)
    int_couplings = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for interaction in int_couplings:
        restrictions = {"mass": -2.0, "int": interaction}
        xvals, yvals_dict, _ = extract_data(data, args.xaxis, args.obs, restrictions, args.require_g)
        yvals = yvals_dict[ob]
        shift = 0 #4*np.sqrt(2)
        yvals = [k + shift for k in yvals]
        verts.append(polygon_under_graph(xvals, yvals))

    facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))

    poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
    ax.add_collection3d(poly, zs=int_couplings, zdir='y')

    ax.set(xlim=(0, 1.5), ylim=(0, 1), zlim=(-10, 10),
        xlabel='el', ylabel='int', zlabel=ob)

    plt.show()


def plot(args):
    
    f, ax = plt.subplots(1, 1)
    restrictions = {}
    for r in args.restrict:
        key, val = r.split("=")
        restrictions[key] = float(val)
    print(f"Plotting with the restrictions: {restrictions}")

    if args.ec is not None and args.ec_labels is None:
        args.ec_labels = ['']*len(args.ec)

    if args.ed is not None:
        data = pd.read_csv(args.ed)
        xvals, yvals_dict, _ = extract_data(data, args.xaxis, args.obs, restrictions, args.require_g)

        for ob in args.obs:
            # reorder
            # this is important when points are connected by lines (as is done for ED data)
            yvals = yvals_dict[ob]
            xvals_m, yvals_m = zip(*sorted(zip(xvals, yvals))) # we define new variables so as not to change xvals while looping over the observables
            
            ax.plot(xvals_m, yvals_m, label = f"ED, obs={ob}")

    if args.ec is not None:
        for ind, ec_data in enumerate(args.ec):
            data = pd.read_csv(ec_data)
            xvals, yvals_dict, _ = extract_data(data, args.xaxis, args.obs, restrictions, args.require_g)
            for ob in args.obs:
                ax.scatter(xvals, yvals_dict[ob], label=f"EC, obs={ob}, {args.ec_labels[ind]}")
    
    if args.mc is not None:
        for ind, mc_data in enumerate(args.mc):
            data = pd.read_csv(mc_data)
            xvals, yvals_dict, yvals_err_dict = extract_data(data, args.xaxis, args.obs, restrictions, args.require_g, get_error=True)
            for ob in args.obs:
                ax.errorbar(xvals, yvals_dict[ob], yerr=yvals_err_dict[ob], label=f"MC, obs={ob}", marker='o', linestyle='')

    if args.logx:
        ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")
    
    # make a helpful title
    if args.title:
        title = args.title
    else:
        title = "Restrictions-"
        for key, val in restrictions.items():
            title += f"{key}={val}_"
        if args.require_g:
            title += "mag=1div(4*el)_"
    
    ax.legend(fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(args.xaxis, fontsize=10)
    #f.tight_layout()

    #if args.diff:
    #    ax.set_ylabel("Value - ED", fontsize=10)
    #else:
    ax.set_ylabel("Value", fontsize=10)
    
    if not args.no_save:
        f.savefig(f"summary_{title}{'-'.join(args.obs)}.pdf")
    if args.show:
        plt.show()

def main(args):
    plot(args)

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
    parser.add_argument("--require_g", action="store_true", default=False, help="Restricts data to cases where el and mag have the required relationship (both derived from g)")
    parser.add_argument("--title", type=str, default="", help="Plot title")

    args = parser.parse_args()

    main(args)
