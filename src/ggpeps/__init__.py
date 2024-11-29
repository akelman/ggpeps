# Global vars
global_vars = {}

# Logger setup
LOGGER_NAME = "ggpeps"
logger_file = None


import os as os_

# Set numerical backend
GPU_AVAILABLE = False  # Default to False
AVAILABLE_NUMERICAL_BACKENDS = ["numpy", "jax"]
if "GGPEPS_BACKEND" in os_.environ:
    if os_.environ["GGPEPS_BACKEND"] in AVAILABLE_NUMERICAL_BACKENDS:
        PREFERRED_BACKEND = os_.environ["GGPEPS_BACKEND"]
    else:
        raise ValueError(f"Unknown backend: {os_.environ['GGPEPS_BACKEND']}")
elif GPU_AVAILABLE:
    PREFERRED_BACKEND = "jax"
else:
    PREFERRED_BACKEND = "numpy"
del os_

if PREFERRED_BACKEND == "numpy":
    import numpy as xnp
    import scipy as xscipy
elif PREFERRED_BACKEND == "jax":
    import jax.numpy as xnp
    import jax.scipy as xscipy
else:
    raise ValueError(f"Unknown backend: {PREFERRED_BACKEND}")


# Importing modules
from ggpeps import caching as caching
from ggpeps import evaluator_manager as evaluator_manager
from ggpeps import evaluator as evaluator
from ggpeps import exacteval as exacteval
from ggpeps import gauge as gauge
from ggpeps import lattice as lattice
from ggpeps import mc as mc
from ggpeps import measurement as measurement
from ggpeps import minimizer as minimizer
from ggpeps import system as system
from ggpeps import utils as utils

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")
