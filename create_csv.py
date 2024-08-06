import os
import pandas as pd


def main(args):

    # cols = ['el', 'mag', 'int', 'mass', 'energy', 'el_energy', 'mag_energy', 'mass_energy', 'int_energy', 'mass_energy', 'tag']
    data = pd.DataFrame()
    obs = ['energy', 'el_energy', 'mag_energy',
           'int_energy', 'mass_energy', 'norm']

    for fname in args.files:
        if os.path.isfile(fname):
            df = pd.read_pickle(fname)
            vals = df.loc[:, 'mean']
            try:
                error = df.loc[:, 'err']
            except:
                error = vals
            keys = df.loc[:, 'name']
            tag = args.tag

            d = {'el': df.loc[0, 'g_el'], 'mag': df.loc[0, 'g_mag'],
                 'int': df.loc[0, 'g_int'], 'mass': df.loc[0, 'g_mass'], 'tag': tag}
            for key, val, error in zip(keys, vals, error):
                if key in obs:
                    d[key] = val
                    d[f"{key}_error"] = error

            # we must wrap the values in order to be able to create a dataframe from dict.
            for key, val in d.items():
                d[key] = [val]

            # data = data.append(d, ignore_index=True) # deprecated
            file_data = pd.DataFrame.from_dict(d, orient='columns')
            data = pd.concat([data, file_data])

    data.to_csv(args.out)
    print(f"Data: {data.shape}")

    return


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", help="Directory")
    parser.add_argument("--out", help="Output CSV file")
    parser.add_argument("--tag", type=str, default='',
                        help="Tag to add to each row of the CSV file")

    args = parser.parse_args()

    main(args)
