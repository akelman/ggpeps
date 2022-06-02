# Gaussian PEPS simulation

This repository contains the code for simulations with Gaussian Fermionic Projected Entangled Pair States (GGPEPS).
The aim is to simulation $Z_N$ lattice gauge theories.

## Installation
The code is written for Python 3 and tested to work with Python 3.8.

To make sure that all requirements of the package are fulfilled, the easiest way to use the code is to create a virtual environment and install the necessary packages independent of the python packages of the operating system

1. Creation of a virtual environment  
You can create the environment in a folder of your choice. 
For the rest of the tutorial, we assume it to be in `~/.pyenv/`
```
cd ~/.pyenv
python -m venv gaussiaenv
```
Assuming you are using bash or zsh, you can activate the environment with `source ~/.pyenv/gaussianenv/bin/activate`.
Upon activation, you will notice that your prompt changes.
As long as it is prefixed with `(gaussianenv)` the virtual environment is active.
The virtual environment can be deactivated with `deactivate`.

2. Cloning the code  
You can obtain the code by cloning the repo with
```
git clone git@gitlab.mpcdf.mpg.de:pemonts/gaussian-peps.git
```.
Note that you have to be a member of the project to clone it.
Cloning via SSH works only if you have added a (public) SSH key to the repository.

3. Preparation of the environment  
For the next step, please navigate into the repo that you just downloaded and activate the empty environment that we created in step 1.
In order to install all required packages for the simulation, execute
```
pip install -r requirements.txt
```

4. Modification of the PYTHONPATH  
Since we are working with a package (`ggpeps`), we have to add it to the PYTHONPATH in order to make Python aware of its existence.
We add the repository to the PYTHONPATH with the following command
```export PYTHONPATH=$PYTHONPATH:<path_to_repo>```
where `<path_to_repo>` is the location of the repository on your computer.

If you want to make the change persistent, i.e. it remains after closing the terminal, you can consider adding it to your `~/.bashrc` (for bash) or `~/.zshrc` (for zsh).

## Structure for the Code

The repository is split into two main parts: the package `ggpeps` and utility scripts in the main folder.

The package `ggpeps` contains the simulation code, i.e. the actual implementation of the physical problem.
All scripts in the main folder call parts of the package and provide the infrastructure to manage the simulations.

The package `ggpeps` is divided into several parts:

- `plot`: Helper scripts for plots
- `system`: Module containing all system implementations. Currently, two-dimensional systems for $Z_2$ are implemented with one and two copies of virtual fermions on the links are implemented.
The implementation of $U(1)$ is transferred from a C++ implementation and is not fully operational.
- `exacteval.py`: For small systems and finite gauge groups, the expectation values of the states can be evaluated exactly by contracting the full PEPS.
- `gauge.py`: Implementation of the gauge groups
- `lattice.py`: Implementation of two- and three-dimensional lattices
- `mc.py`: Implementation of the Monte Carlo sampling
- `measurement.py`: Measurement class to manage the timeseries data of the MC simulation
- `minimizer.py`: Minimizer class with a custom minimizer and a wrapper of the `scipy.optimize.minimize` function.
- `utils.py`: Utility functions

## Physical System

TODO: Fill

## Data Generation

The script `manager.py` is the central point for data generation. It supports different modes: `eval`, `exact`, `min` and `minexact`.

All modes are writing log files to disk and to console. 
The files are named according to the parameters that were provided via the commandline. 
In addition to the progress of the computation, they also store a git hash which enables to identify them with a certain version of the code later.

All data is stored in the form of pandas dataframes in pickle (`.pkl`) files.
While these files are convenient to work with in Python, they are a bit unintuitive to inspect on the commandline.
The tool `inspect_data.py` takes all output files generated with this code and displays them concisely.

In the following, we will describe the different modes in more detail.

`eval`: 
The evaluation mode computes the expectation value of a set of observables with given set of parameters using Monte Carlo.
To simulate a $2\times 2$ system with MC, we can call
```
python manager.py eval 2
```
The call generates three files: a log file, a data file and a summary file.
The log file is identical to the text printed on the console.
It is especially useful to check computations performed on a cluster.

The data file contains the full timeseries of the computation and can get quite large.
It is compressed by default to save disk space

The summary file is most relevant for most plots since it contains the mean values of observables including errors (computed via binning analysis).

`exact`:
The exact evaluation mode computes the expectation value of a set of observables with given set of parameters using exact contraction.
This works only for small systems of $L=2$.

```
python manager.py eval 2
```

`min`:
In minimization mode, the Kogut-Susskind Hamiltonian for the gauge theory in question is minimized by using different minimizers. 
The expectation values for a given set of parameters are computed with Monte Carlo.
The update according to the computed energy and gradients is controlled by the optimizer.
Currently, scipy optimizers like `BFGS` and a custom gradient based optimizer is available.

```
python manager.py min 2 --method BFGS
```

`minexact`:
For small systems, we can substitute the Monte Carlo evaluation part in the minimization (just as we did in `eval` mode) with an exact contraction.
Exact contraction are only available for systems of size 2x2.

```
python manager.py minexact 2 --method BFGS
```

For an overview of all command line parameters call `python manager.py --help`.

## Data Analysis/ Data Exploration

The data from different modes is stored in the form of pickled pandas dataframes.

A basic overview of the data can be obtained with 
```
python inspect_data.py <fname>
```

All scripts prefixed with `plot_*` will plot some aspect of the provided datasets.
The most used script is `plot_summary.py` which displays the data of a `summary_*.pkl` file.
Minimization and evalautions of single parameters produce summary files.

A typical can look like
```
python ../../../../plot_summary.py --ec summary_min* --obs el_energy mag_energy energy --show
```
The option `--show` displays the interactive matplotlib plot before saving the plot to disk.
This script is meant for data exploration and should not be used to produce paper-style plots.

Further information about the capabilities of `plot_summary.py` can be obtained with `python plot_summary.py --help`.

## Tests

The code is accompagnied by an extensive suit of tests which are located in the folder `tests`.
The full test-suite can be executed with
```
python -m unittest tests/test_*.py
```

If you want to execute a more specialized test, you can execute the files separately as well:
```
python -m unittest tests/test_lattice.py
```

## Known Issues

- Current implementation of U1 is not working properly
- Bogoliubov transform yields wrong results if used with fermions

## Ideas

- Add U1 system properly
- Add system in 3d
- Add option for DMRG like cylinder compression to obtain transfer matrices
- Make data file optional?
