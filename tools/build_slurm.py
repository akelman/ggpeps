import os
import re


def name_from_path(prefix, path):
    return prefix + path


def folder2arg(foldername, arg):
    # handle gauge_fixing separately, since it is not a numeric argument
    if arg == "gf":
        pattern = rf"(?<={arg}_)([^_]+)"
        result = re.search(pattern, foldername)
        if result is not None:
            if result.group(0) == "T":
                return f"--gauge_fixing"
            elif result.group(0) == "F":
                return ""  # relies on the fact that the default value for gauge_fixing is False
            elif result.group(0) == "c":
                return f"--gauge_fixing -2"
            else:
                return f"--gauge_fixing {result.group(0)}"
        else:
            return ""

    # allow for any number of numeric values (ints or floats) separated by underscores
    pattern = rf"(?<={arg}_)(-?\d+(?:\.\d+)?(?:_-?\d+(?:\.\d+)?)*)"

    result = re.search(pattern, foldername)

    if result is not None:
        vals = " ".join(result.group(1).split("_"))
        return f"--{arg} {vals}"
    else:
        return ""


def format_slurmfile(slurmfile, name, folder, time, command, args, num_threads):
    for arg in args:
        command += " " + folder2arg(folder, arg)
    return slurmfile.format(jobname=name, time=time, command=command, ncpu=num_threads)


def main(args):
    if os.path.isfile(args.slurmfile):
        with open(args.slurmfile, mode="r") as infile:
            slurmfile_str = infile.read()

        for folder in args.folders:
            if os.path.isdir(folder):
                with open(os.path.join(folder, "slurmjob"), "w") as f:
                    name = name_from_path(args.prefix, folder)
                    slurmfile_format = format_slurmfile(
                        slurmfile_str,
                        name,
                        folder,
                        args.time,
                        args.command,
                        args.args,
                        args.num_threads,
                    )
                    print(slurmfile_format, file=f)
            else:
                print(f"Folder '{folder}' does not exist")

    else:
        print(f"Master slurmfile not found at '{args.slurmfile}'")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--folders",
        dest="folders",
        type=str,
        nargs="+",
        help="List of folders to modify",
        default=None,
    )
    parser.add_argument(
        "-i",
        "--slurmfile",
        dest="slurmfile",
        type=str,
        help="Master slurm file",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="time",
        type=str,
        help="time attribute (default: 24:00:00)",
        default="24:00:00",
    )
    parser.add_argument("-c", "--command", dest="command", type=str, help="command", default="")
    parser.add_argument(
        "-a",
        "--args",
        dest="args",
        nargs="+",
        help="args to parse from filepath",
        default=["el", "mag", "int", "mass"],
    )
    parser.add_argument(
        "-n",
        "--num_threads",
        dest="num_threads",
        type=str,
        help="number of OMP threads (default: 4)",
        default="4",
    )
    parser.add_argument("--prefix", type=str, help="name prefix (default: empty string)", default="")
    args = parser.parse_args()
    main(args)
