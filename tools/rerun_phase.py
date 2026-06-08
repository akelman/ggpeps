#####################################################
# Script to restart simulations based on value of
# energy (or another observable) at each point
# compared to neighboring points in a phase diagram.
#
# created: Feb 2026
#
#####################################################

import os
import re
import glob
import subprocess

import numpy as np
import pandas as pd

import plotly.express as px


def len_arr(x):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return len(x)
    else:
        return 1


def fname2arg(fname: str, arg: str):
    """Extract the number of layers from a filename"""
    pattern = rf"(?<={arg}_)[\d]*"
    result = re.search(pattern, fname)
    if result is not None:
        return int(result.group(0))
    return None


def get_data(
    ec_files,
    mc_files,
):
    """Collect all data into a single dataframe for EC and MC. Collect ED data as well."""
    dfvec = []
    if ec_files is not None:
        for fname in ec_files:
            if os.path.isfile(fname):
                df_tmp = pd.read_pickle(fname)
                df_tmp["type"] = "EC"
                df_tmp["err"] = np.nan  # Append an empty column to make merges work
                if "nlayer" not in df_tmp.columns:
                    df_tmp["nlayer"] = fname2arg(fname, "nlayer")
                if "ncopy" not in df_tmp.columns:
                    df_tmp["ncopy"] = fname2arg(fname, "ncopy")
                dfvec.append(df_tmp)
    if mc_files is not None:
        for fname in mc_files:
            if os.path.isfile(fname):
                df_tmp = pd.read_pickle(fname)
                df_tmp["type"] = "MC"
                if "nlayer" not in df_tmp.columns:
                    df_tmp["nlayer"] = fname2arg(fname, "nlayer")
                if "ncopy" not in df_tmp.columns:
                    df_tmp["ncopy"] = fname2arg(fname, "ncopy")
                dfvec.append(df_tmp)
    df_mc_ec = pd.concat(dfvec)

    # Enrich dataset
    df_mc_ec["L"] = df_mc_ec["nx"].astype("str") + "x" + df_mc_ec["ny"].astype("str")
    df_mc_ec.rename(
        columns={
            "g2_el": "g_el",
            "g2_mag": "g_mag",
            "g2": "g",
        },
        inplace=True,
    )

    if "g" not in df_mc_ec.columns:
        df_mc_ec["g"] = df_mc_ec.g_el * 2
    if "g_mag" not in df_mc_ec.columns:
        df_mc_ec["g_mag"] = 1 / (df_tmp.g_el * 4)

    # different runs were done with versions of the code -
    # at some point the naming convention of Wilson loops was modified, we fix that here
    df_mc_ec.replace(
        {"wilson_00_11": "wilson_loop_0-0_1x1"},
        regex=True,
        inplace=True,
    )

    return df_mc_ec


def get_obs1(df, g, g_int, g_mass, g_chem, obs, column):
    data = df.loc[
        (np.isclose(df["g_int"], g_int))
        & (np.isclose(df["g_chem"], g_chem))
        & (np.isclose(df["g"], g))
        & (np.isclose(df["g_mass"], g_mass))
    ]
    val = data.loc[data["name"] == obs, column].values[0]
    return val


def get_obs2(df, g, g_int, g_mass, g_chem, obs, column):
    mask = (
        np.isclose(df["g_int"], g_int)
        & np.isclose(df["g_chem"], g_chem)
        & np.isclose(df["g"], g)
        & np.isclose(df["g_mass"], g_mass)
        & (df["name"] == obs)
    )
    return df.loc[mask, column].iat[0]


def get_obs(df_idx, g, g_int, g_mass, g_chem, column):
    return df_idx.loc[(g, g_int, g_mass, g_chem), column]


def get_neighbors(df, g, g_int, g_mass, g_chem, column, bounds):
    neighbors_vals = []
    neighbors_couplings = []
    for g_int_n in [g_int - 0.1, g_int + 0.1]:
        for chem_n in [g_chem - 0.1, g_chem + 0.1]:
            if (
                g_int_n < bounds["g_int"][0]
                or g_int_n > bounds["g_int"][1]
                or chem_n < bounds["g_chem"][0]
                or chem_n > bounds["g_chem"][1]
                or not (g, g_int_n.round(1), g_mass, chem_n.round(1)) in df.index
            ):
                continue
            val = get_obs(df, g, g_int_n.round(1), g_mass, chem_n.round(1), column)
            neighbors_vals.append(val)
            neighbors_couplings.append((g_int_n, chem_n))
    return neighbors_vals, neighbors_couplings


def is_outlier(value, neighbors):
    # thresh = 0.1  # 1.07
    # if abs(energy - neighbors_energy[idx]) > thresh * min(neighbors_energy):
    med = np.median(neighbors)
    diff = value - med
    mad = np.median(abs(neighbors - med))
    return diff / mad > 3


def main():

    # Data
    ec_files = None
    base_dir = "L4/mc/round4"
    mc_files = glob.glob(os.path.join(base_dir, r"g*/sum*.pkl"))

    modify_data = True  # Set to True to visually mark points for rerunning and skip actual rerunning for testing

    obs = "energy"
    column = "mean"
    df = get_data(ec_files, mc_files)
    df["g_chem"] = df["g_chem"].apply(lambda x: float(x) if isinstance(x, (list, np.ndarray)) else x)
    GRID_DECIMALS = 1

    for col in ["g", "g_int", "g_mass", "g_chem"]:
        df[col] = df[col].astype(float).round(GRID_DECIMALS)
    df_obs = df[df["name"] == obs].set_index(["g", "g_int", "g_mass", "g_chem"]).sort_index()
    num = 0

    g = 1.0
    g_mass = 1.0
    bounds = {
        "g_int": (0.0, 3.9),
        "g_chem": (0.0, 5.9),
    }

    os.chdir(base_dir)
    for g_int in np.linspace(0.0, 3.9, 40):
        for g_chem in np.linspace(0.0, 5.9, 60):
            chem_clean = g_chem.round(1)
            int_clean = g_int.round(1)

            if not (g, int_clean, g_mass, chem_clean) in df_obs.index:
                continue

            obs_val = get_obs(df_obs, g, int_clean, g_mass, chem_clean, column)
            neighbors_vals, neighbors_couplings = get_neighbors(
                df_obs, g, int_clean, g_mass, chem_clean, column, bounds
            )

            if is_outlier(obs_val, neighbors_vals):

                num += 1

                idx = np.argmin(neighbors_vals)
                print(f"Observable {obs} at g_int={g_int}, chem={chem_clean} is an outlier.")

                if modify_data:
                    df_obs.loc[(g, int_clean, g_mass, chem_clean), column] = 1000

                    continue

                neighbor_coupling = neighbors_couplings[idx]

                # move previous results to a new folder
                dirname = f"g_{g}_int_{g_int:.1f}_mass_{g_mass:.1f}_chem_{g_chem:.1f}"
                os.chdir(dirname)
                subprocess.run(["prepare_restart_job"], check=True)

                # res2params from neighbor
                neighbor_name = (
                    f"g_{g}_int_{neighbor_coupling[0]:.1f}_mass_{g_mass:.1f}_chem_{neighbor_coupling[1]:.1f}"
                )
                path = os.path.join("..", neighbor_name, "res*")
                matches = glob.glob(path)  # Expand wildcard
                if not matches:
                    raise FileNotFoundError(f"No files matched {path}")
                subprocess.run(["res2params", *matches], check=True)

                # rename params
                path = "*extracted*.npy"
                matches = glob.glob(path)
                if not matches:
                    raise FileNotFoundError(f"No files matched {path}")
                os.rename(matches[0], "paramvec.npy")

                # restart job
                subprocess.run(["sbatch", "slurmjob"], check=True)

                # change back to main directory
                os.chdir("..")

    print(f"Total number of points to rerun: {num}")
    plot = True
    if plot:
        fig = px.imshow(
            df_obs.pivot_table(index="g_chem", columns="g_int", values=column).astype(float),
            color_continuous_scale="viridis",
            labels=dict(x="X", y="Y", color="Value"),
        )

        fig.show()
    return


if __name__ == "__main__":

    main()
