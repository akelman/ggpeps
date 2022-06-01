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

TODO: Write more about the structure of the code

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
In minimization mode, the Kogut Susskind Hamiltonian for the gauge theory in question is minimized by using different minimizers. 
The expectation values for a given set of parameters are computed with Monte Carlo.
The update according to the computed energy and gradients is controlled by the optimizer.
Currently, scipy optimizers like `BFGS` and a custom gradient based optimizer is available.

TODO: Add example call

`minexact`:
For small systems, we can substitute the Monte Carlo evaluation part in the minimization (just as we did in `eval` mode) with an exact contraction.

TODO: Add example call

For an overview of all command line parameters call `python manager.py --help`.

## Data Analysis/ Data Exploration

The data from different modes is stored in the form of pickled dataframes.

A basic overview of the data can be obtained with 
```
python inspect_data.py <fname>
```

All scripts prefixed with `plot_*` will plot some aspect of the provided datasets.

TODO: More about plotting scripts

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

- Error in computation of electric energy of Z2 theories in 2d

## Ideas

- Add U1 system
- Add system in 3d
- Add option for DMRG like cylinder compression to obtain transfer matrices
- Make data file optional?
