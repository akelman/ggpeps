# Gauged Gaussian PEPS


This repository contains the code for simulations of lattice gauge theories (LGTs) with Gauged Gaussian Fermionic Projected Entangled Pair States (GGFPEPS).
Currently, $\mathbb{Z}_2$ theories are fully operational, and $D_n$ theories are a work in progress. The $U(1)$ implementation is not functional.

The purpose of this README is to provide:
1. a source of information for new team members;
2. an up-to-date high-level description of the structure of the project, and how to run simulations;
3. a record of papers that use this code and the theory around it. 

**Contents**
<!-- @import "[TOC]" {cmd="toc" depthFrom=2 depthTo=3 orderedList=true} -->

<!-- code_chunk_output -->

1. [Development](#development)
    1. [Installation](#installation)
    2. [Structure of the Code](#structure-of-the-code)
    3. [Code Style](#code-style)
    4. [Tests](#tests)
    5. [Known Issues](#known-issues)
    6. [Ideas](#ideas)
    7. [Contributing](#contributing)
2. [Use](#use)
    1. [Data Generation](#data-generation)
    2. [Reproducibility](#reproducibility)
    3. [High Performance Computing](#high-performance-computing)
    4. [Data Analysis / Exploration](#data-analysis--exploration)
3. [Papers](#papers)

<!-- /code_chunk_output -->



## Development 

### Installation
The code is written for Python 3 and tested to work with Python 3.9.
Earlier and later versions should work as well (though there may be issues relating to type hints in python < 3.9).

To make sure that all requirements of the package are fulfilled, the easiest way to use the code is to create a virtual environment and install the necessary packages independent of the python packages of the operating system.

1. **Create a virtual environment**
You can create the environment in a folder of your choice. 
For the rest of the tutorial, we assume it to be in `~/.pyenv/`.
    ```
    cd ~/.pyenv
    python -m venv ggpeps
    ```
    Assuming you are using `bash` or `zsh`, you can activate the environment with `source ~/.pyenv/ggpeps/bin/activate`. If you are using `csh`, instead use `source ~/.pyenv/ggpeps/bin/activate.csh`.
    Upon activation, you will notice that your prompt changes. As long as it is prefixed by `(ggpeps)` the virtual environment is active.
    The virtual environment can be deactivated with `deactivate`.
<br/>
2. **Clone the code**
You can obtain the code by cloning the repo with:
    ```
    git clone git@gitlab.com:patrick.emonts/gaussian-peps.git
    ```
    Note that you have to be a member of the project to clone it.
    Cloning via SSH works only if you have added a (public) SSH key to the repository.
<br/>
3. **Install the package**
For the next step, please navigate into the repo that you just downloaded and activate the empty environment that we created in step 1.
(If you intend to be able to use GPUs, see the note below.)

    To install all required packages for the simulation, execute
    ```
    pip install -e .
    ```
    This command installs the package as an editable package, i.e. all changes in the source code will be directly reflected in the installed package.

    If you wish to install the optional depencies, instead run
    ```
    pip install  -e .[dev,test]
    ```

    If you do not intend to edit the code, you can install with
    ```
    pip install .
    ```

    All commands will install the package `ggpeps` into your environment (and the necessary dependencies).

You can test your installation by opening a python console (just type `python`) and executing
```python
import ggpeps
ggpeps.__version__
```
The result should be a version string, e.g. `0.1.dev952+ga571e99.d20240918`, which can be interpreted as: `version 0.1` on the `dev` branch, which is `952` commits ahead of master, with the git commit hash beginning `a571e99`, on the date `2024-09-18`.

#### Installation with GPUs
Installation for use with GPUs can be tricky. JAX is the library we use for running on GPUs, and JAX must be be installed with jaxlib and connected to the GPU in the correct manner in order to function. The versions required will depend on what's available on a given cluster.

Our code has been tested and works with both NVIDIA and AMD GPUs.

**NVIDIA GPUs**: When working on a cluster, first, purge any loaded modules, with `module purge`.
Then load the appropriate modules for `python` and `cuda`.
To see the available modules run `module avail`, to load a module run `module load <module>`, and to see a list of loaded modules run `module list`.

Once this is done, create and activate a virtual environment (as above).
Before installing the package, run `pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`, which will install JAX and connect it to the appropriate version of CUDA.
Then install the package as usual as described above (e.g. `pip install -e .` in the appropriate directory).

NOTE: It may be necessary to do this installation on a node that has access to the GPU; otherwise JAX may not include the required support for CUDA despite the use of the above command.

**AMD GPUs**: For now, we have only successfully run simulations using pre-built containers with a version of JAX that connects to GPUs. This is heavily dependent on the setup on a particular cluster.


### Structure of the Code

The repository is split into two main parts: the package `ggpeps` and utility scripts in the main folder.

The package is contained in `src/ggpeps/` and contains the simulation code, i.e. the actual implementation of the physical problem.
All scripts in the main folder call parts of the package and provide the infrastructure to manage the simulations. This includes `tests/`, `tools/` which contains various bash and python scripts (especially to help manage batches of simulations on a cluster), and `plot/` with various plotting scripts.

The package `ggpeps` is divided into several parts:

- `system/`: Module containing all system implementations. Currently, only two-dimensional systems are supported, and each gauge group has its own class.
The system configs contain all information defining an ansatz (the $T$ matrix, $\Gamma_\text{in}$, etc.), while the system classes define observables, intermediate calculations, etc.
The implementation of $U(1)$ is transferred from a C++ implementation and is not fully operational.
- `exacteval.py`: For small systems with finite gauge groups, the expectation values of the states can be evaluated exactly by contracting the full PEPS.
- `gauge.py`: Implementation of the gauge groups.
- `lattice.py`: Implementation of two- and three-dimensional lattices.
- `mc.py`: Implementation of the Monte Carlo sampling.
- `measurement.py`: Measurement class to manage the timeseries data of the MC simulation.
- `minimizer.py`: Minimizer class with a custom minimizer and a wrapper of the `scipy.optimize.minimize` function.
- `utils.py`: Utility functions.

Each implemented ansatz has it's own config class, each a subclass of Config2DBase (found in `config_base.py`). Currently, these are the implemented ansatz's (each located in a file of the same name):
- `config_u1_2d`: Not working - the implementation of $U(1)$ is transferred from a C++ implementation and is not fully operational.
- `config_D6_2d`: $D_6$ ansatz.
- `config_z2`: $\mathbb{Z}_2$ ansatz.

Our code supports use on both CPU and GPU. To run on GPU, we use JAX ([documentation](https://docs.jax.dev/en/latest/index.html), [github](https://github.com/jax-ml/jax/blob/main/README.md)), which mostly follows the numpy syntax (see also [the Array API](https://data-apis.org/array-api/latest/index.html)).
This is handled by importing numpy/jax as `xnp` in `ggpeps/__init__.py`, and using `xnp` throughout the code.
The system configs only use `numpy`; they system classes use `xnp`. Objects which interface with the system (primarily `Evaluator` objects) can send numpy arrays (e.g. when updating a gauge field) but may receive jax or numpy arrays depending on the particular quantity they access.

In addition the `system/backend/` directory contains functions whose syntax depends on the use of numpy/jax.

NOTE: The pure gauge ansatz's all techincally contain a parameter for coupling to matter, but (a) it is manually set to zero, (b) other parts of the ansatz (e.g. the $\Gamma_{\text{in}}$) do not obey the symmetries required for including matter.

### Code Style
Code is formatted using `black` with the default configuration, except that the maximum allowed line length is 119.
To format your code, run `python black . --line-length 119` from the main repository directory. 
To set up your editor to automatically format your code (e.g. on save), see [Black Editor Integrations](https://black.readthedocs.io/en/stable/integrations/editors.html). Black is listed in the package dependencies under the `dev` tag (thus it is available if the package was installed with the dev dependencies, otherwise it must be installed manually: `pip install black`).

The goal of type hinting is to improve readability and reasoning about the code. Only the `src` directory is typed.
We use `mypy` for static type checking, though it is not strictly enforced. A `nox` session is used to validate type hints (see below for details on `nox`).
Occasionally `assert` statements or `# type: ignore` comments are used to address type errors - the goal is to improve documentation and readability; in cases where those aims are better served by ignoring a typing error, we do so.

### Tests

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

#### Testing across multiple architectures
The package supports CPU and GPU operation.
To test both these modes independently, the tests are run with `nox` to run in different environments.
Additionally, it enables testing of the environment, coverage, type hints, and lint testing.

The full `nox` test suite can be executed with
```
nox
```
It will run all so-called sessions. For an overview of available sessions, execute `nox --list`. 
Individual sessions can be executed with `nox -s <name of session>`.


### Known Issues

- Current implementation of U1 is not working properly
- Bogoliubov transform yields wrong results if used with fermions (this transform is not used in the current implementation)

### Ideas

- Extend to other gauge groups: $Z_N$, fix $U(1)$, finish $D_n$, etc.
- Add support for 3D systems.
- Add option for DMRG like cylinder compression to obtain transfer matrices

### Contributing

We would be very happy to hear ideas for improving our code.
Pull requests would be appreciated for minor improvements; for  major updates we would appreciate hearing from any potential contributor as early as possible.


## Use

### Data Generation

The script `manager.py` is the central point for data generation. It supports different modes: `eval` and `min` where both can be evaluated with `exact` and `mc`.

To run with JAX (whether on CPU or GPU), first export the environment variable: `export GGPEPS_BACKEND=jax` (it can also be set to `numpy`, but numpy will also be used by default regardless).
If using multiple runners in conjuction with a GPU (we have not tested using multiple GPUs simultaneously, though this should only require small changes), then memory issues can arise. JAX preallocates 75% of GPU memory upon startup; with multiple runners, each runner tries to allocate this memory, causing a crash. This can be solved using an environment variable: `XLA_PYTHON_CLIENT_MEM_FRACTION=.XX` where `XX` should be $1/\text{nrunner}$, rounded down if necessary. Note that this only addresses preallocation, and the program may crash if a runner tries to request more memory. See the [JAX documentation on GPU Memory Allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html) for more information.

All modes write log files to disk and to console. 
The files are named according to the parameters that were provided via the commandline. 
In addition to the progress of the computation, they also store a git hash which enables the user to identify which version of the code was used to generate particular data.

All data is stored as pandas dataframes in pickle (`.pkl`) files.
While these files are convenient to work with in Python, they are a bit unintuitive to inspect on the commandline.
The tool `inspect_data.py` takes all output files generated with this code and displays them concisely.

In the following, we will describe the different modes in more detail.

`eval-mc`: 
The evaluation mode computes the expectation value of a set of observables with given set of parameters using Monte Carlo.
To simulate a $2\times 2$ system with MC, we can run
```
python manager.py eval-mc 2
```
The call generates three files: a log file, a data file, and a summary file.
The log file is identical to the text printed on the console.
It is especially useful to check computations performed on a cluster.

The data file contains the full timeseries of the computation and can get quite large.
It is compressed by default to save disk space.

The summary file is most relevant for most plots since it contains the mean values of observables including errors (computed via binning analysis).

`eval-exact`:
The exact evaluation mode computes the expectation value of a set of observables with given set of parameters using exact contraction.
This works only for small systems of $L=2$. For systems of size $L=4$, it may also be possible to run in `exact` mode if gauge fixing is turned on, though this will still be slower than MC with the default number of steps. 

```
python manager.py eval-exact 2
```

`min-mc`:
In minimization mode, the Kogut-Susskind Hamiltonian for the gauge theory in question is minimized. 
The expectation values for a given set of parameters are computed with Monte Carlo.
The update according to the computed energy and gradients is controlled by the optimizer.
Several different minimizers are available. 
Currently, scipy optimizers (such as `BFGS`) as well as a custom gradient based optimizer are available.

```
python manager.py min-mc 2 --method BFGS
```

`min-exact`:
For small systems, we can substitute the Monte Carlo evaluation part in the minimization (just as we did in `eval` mode) with an exact contraction.
Exact contraction is only practical for systems of size 2x2.

```
python manager.py min-exact 2 --method BFGS
```

For an overview of all command line parameters call `python manager.py --help`.

### Reproducibility
Successive runs produce identical output, provided they use the same version of the code with the same command-line arguments, and also use the same seed (if the parameters are provided, and no Monte Carlo is used, randomness should have no effect, and so the seed does not matter).

The exception is related to caching, which can interfere with the randomness as well as the minimizer.


### High Performance Computing

The repo includes several scripts to help with running many jobs on a computing cluster.
It can also interpret signals, e.g. as sent by slurm, to automatically cache and end a computation.


### Data Analysis / Exploration

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

Further information about the capabilities of `plot_summary.py` can be obtained with `python plot_summary.py --help`.


## Papers

The following is a list of papers that have used (versions of) this code:
1. Emonts et al, Finding the ground state of a lattice gauge theory with fermionic tensor networks: A $2+1\mathrm{D}$ ${\mathbb{Z}}_{2}$ demonstration, PRD vol 107 (2023).

There are also some papers that used a previous C++ implementation of this code.