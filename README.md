# Gaussian PEPS simulation

This repository contains the code for simulations with Gaussian Fermionic Projected Entangled Pair States (GGPEPS).
The aim is to simulate lattice gauge theories. Currently only $Z_N$ theories are operational.

## Installation
The code is written for Python 3 and tested to work with Python 3.9.
Earlier and later versions should work as well (though there may be issues relating to type hint in python < 3.9).

To make sure that all requirements of the package are fulfilled, the easiest way to use the code is to create a virtual environment and install the necessary packages independent of the python packages of the operating system.

1. **Create a virtual environment**
You can create the environment in a folder of your choice. 
For the rest of the tutorial, we assume it to be in `~/.pyenv/`.
    ```
    cd ~/.pyenv
    python -m venv gaussianenv
    ```
    Assuming you are using bash or zsh, you can activate the environment with `source ~/.pyenv/gaussianenv/bin/activate`.
    Upon activation, you will notice that your prompt changes.
    As long as it is prefixed with `(gaussianenv)` the virtual environment is active.
    The virtual environment can be deactivated with `deactivate`.
<br/>
2. **Clone the code**
You can obtain the code by cloning the repo with:
    ```
    git clone git@gitlab.mpcdf.mpg.de:pemonts/gaussian-peps.git
    ```
    Note that you have to be a member of the project to clone it.
    Cloning via SSH works only if you have added a (public) SSH key to the repository.
<br/>
3. **Install the pacakge**
For the next step, please navigate into the repo that you just downloaded and activate the empty environment that we created in step 1.
(If you intend to be able to use GPUs, see the note below.)

    To install all required packages for the simulation, execute
    ```
    pip install -e .
    ```
    This command installs the package as an editable package, i.e. all changes in the source code will be directly reflected in the installed package.

    If you do not intend to develop the code, you can install with
    ```
    pip install .
    ```

    Both commands will install the package `ggpeps` into your environment (and the necessary dependencies).

You can test your installation with opening a python console (just type `python`) and executing
```python
import ggpeps
ggpeps.__version__
```
The result should be a version string, e.g. `0.1+g38ac83d20240911`.

### Installation with GPUs
JAX is the library we use for running on GPUs. JAX must be installed wth jaxlib and connected to the correct versions of CUDA. The versions required will depend on what's available on a given cluster.

First, purge any loaded modules, with `module purge`.
Then load the appropriate modules for `python` and `cuda`.
To see the available modules, run `module avail`, to load a module run `module load <module>`, and to see a list of loaded modules run `module list`.

Once this is done, create and activate a virtual environment (as above).
Before installing the requirements, run `pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`, which will install JAX and connect it to the appropriate version of CUDA.
Then install the remaining requirements.

TODO: Add a build to `pyproject.toml` which works for JAX on systems with a GPU and CUDA.

# Development 

## Structure of the Code

The repository is split into two main parts: the package `ggpeps` and utility scripts in the main folder.

The package `ggpeps` contains the simulation code, i.e. the actual implementation of the physical problem.
All scripts in the main folder call parts of the package and provide the infrastructure to manage the simulations.

The package `ggpeps` is divided into several parts:

- `plot/`: Helper scripts for plots.
- `system/`: Module containing all system implementations. Currently, two-dimensional systems for $Z_2$ with one and two copies of virtual fermions on the links are implemented for the pure gauge case, and 2D systems with 2,4,8 copies of virtual fermions on the links for the fermionic case.
The system configs contain all information defining an ansatz (the $T$ matrix, $\Gamma_\text{in}$, etc.), while the system classes define observables, intermediate calculations, etc.
The implementation of $U(1)$ is transferred from a C++ implementation and is not fully operational.
- `exacteval.py`: For small systems and finite gauge groups, the expectation values of the states can be evaluated exactly by contracting the full PEPS.
- `gauge.py`: Implementation of the gauge groups
- `lattice.py`: Implementation of two- and three-dimensional lattices
- `mc.py`: Implementation of the Monte Carlo sampling
- `measurement.py`: Measurement class to manage the timeseries data of the MC simulation
- `minimizer.py`: Minimizer class with a custom minimizer and a wrapper of the `scipy.optimize.minimize` function.
- `utils.py`: Utility functions

Each implemented ansatz has it's own config class, each a subclass of Config2DBase (found in `system_base.py`). Currently, these are the implemented ansatz's (each located in a file of the same name):
- `system_u1_2d`: Not working - the implementation of $U(1)$ is transferred from a C++ implementation and is not fully operational.
- `system_z2_2d`: $Z_2$, 1 copy of virtual modes per layer, pure gauge.
- `system_z2_2d_2c`: $Z_2$, 2 copies of virtual modes per layer, pure gauge.
- `system_z2_2d_8c`: $Z_2$, 8 copies of virtual modes per layer. This is extremely impractical to run, even for 2x2 systems, due to the large number of virtual modes; it was built for testing purposes. Because there are so many parameters, this ansatz is more systematic in handling them.
- `system_z2_2d_G2c_F2c`: $Z_2$, 2 copies of virtual modes per layer (PG and matter layers), includes matter.
- `system_z2_2d_G2c_F4c`: *this is misnamed, and includes 4 copies per layer for both layers*. However the extra copies in the PG layer are set to zero, which makes it effectively 2 copies (though with matrix sizes, and computational cost, of 4 copies).

The pure gauge ansatz's all techincally contain a parameter for coupling to matter, but (a) it is manually set to zero, (b) other parts of the ansatz (e.g. the Gamma_in) do not obey the symmetries required for including matter.

## Code Formatting
Code is formatted using `black` with the default configuration.
To format your code, run 
```python
black .
```
from the main repository directory. 
To set up your editor to automatically format your code (e.g. on save), see [Black Editor Integrations](https://black.readthedocs.io/en/stable/integrations/editors.html).

## Data Generation

The script `manager.py` is the central point for data generation. It supports different modes: `eval`, and `min` where both can be evaluated with `exact` and `mc`.

All modes write log files to disk and to console. 
The files are named according to the parameters that were provided via the commandline. 
In addition to the progress of the computation, they also store a git hash which enables the user to identify which version of the code was used to generate particular data.

All data is stored in the form of pandas dataframes in pickle (`.pkl`) files.
While these files are convenient to work with in Python, they are a bit unintuitive to inspect on the commandline.
The tool `inspect_data.py` takes all output files generated with this code and displays them concisely.

In the following, we will describe the different modes in more detail.

`eval-mc`: 
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

`eval-exact`:
The exact evaluation mode computes the expectation value of a set of observables with given set of parameters using exact contraction.
This works only for small systems of $L=2$.

```
python manager.py eval 2
```

`min-mc`:
In minimization mode, the Kogut-Susskind Hamiltonian for the gauge theory in question is minimized by using different minimizers. 
The expectation values for a given set of parameters are computed with Monte Carlo.
The update according to the computed energy and gradients is controlled by the optimizer.
Currently, scipy optimizers like `BFGS` and a custom gradient based optimizer is available.

```
python manager.py min 2 --method BFGS
```

`min-exact`:
For small systems, we can substitute the Monte Carlo evaluation part in the minimization (just as we did in `eval` mode) with an exact contraction.
Exact contraction is only available for systems of size 2x2.

```
python manager.py minexact 2 --method BFGS
```

For an overview of all command line parameters call `python manager.py --help`.

## High Performance Computing

The repo includes several scripts to help with running many jobs on a computing cluster.
It can also interpret signals, e.g. as sent by slurm, to automatically cache and end a computation.


## Data Analysis / Exploration

The data from different modes is stored in the form of pickled pandas dataframes.

A basic overview of the data can be obtained with 
```
python inspect_data.py <fname>
```

All scripts prefixed with `plot_*` will plot some aspect of the provided datasets.
The most used script is `plot_summary.py` which displays the data of a `summary_*.pkl` file.
Minimization and evalautions of single parameters produce summary files.

A typical use can look like
```
python plot_summary.py --ec summary_min* --obs el_energy mag_energy energy --show
```
The option `--show` displays the interactive matplotlib plot before saving the plot to disk.
This script is meant for data exploration and should not be used to produce paper-style plots.

Further information about the capabilities of `plot_summary.py` can be obtained with `python plot_summary.py --help`.

## Tests

The code is accompagnied by an extensive suit of tests which are located in the folder `tests`.
The full test-suite can be executed with
```
python -m unittest
```
from the main project folder.

If you want to execute a more specialized test, you can execute the files separately as well:
```
python -m unittest tests/test_lattice.py
```

### Testing across multiple architectures
The package supports CPU and GPU operation.
To test both these modes independently, the tests are run with `nox` to run in different environments.
Additionally, it enables testing of the environment, coverage testing and lint testing.

The full `nox` test suite can be executed with
```
nox
```
It will run all so-called sessions. For an overview of available sessions, execute `nox --list`.
Individual sessions can be executed with `nox -s <name of session>`.


## Known Issues

- Current implementation of U1 is not working properly
- Bogoliubov transform yields wrong results if used with fermions (this is not used in any case)

## Ideas

- Add U1 system properly
- Add system in 3d
- Add option for DMRG like cylinder compression to obtain transfer matrices
- Make data file optional?

# Papers
The following is a list of papers that have used (versions of) this code:
1. Emonts et al, Finding the ground state of a lattice gauge theory with fermionic tensor networks: A $2+1\mathrm{D}$ ${\mathbb{Z}}_{2}$ demonstration, PRD vol 107 (2023).

There are also some papers that used a previous C++ implementation of this code.