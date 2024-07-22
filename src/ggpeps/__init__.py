# Global vars
global_vars = {}

# Logger setup
LOGGER_NAME = 'ggpeps'
logger_file = None


# Configure JAX 
import jax as jax_
# TODO: does this need to be here?
# The following line ensures that JAX is configured to 64-bit precision.
# Without this line, some of the precision tests do not pass.
jax_.config.update("jax_enable_x64", True)

# GPU or CPU
available_devices_ = jax_.devices() # available_gpus = jax.devices('gpu')
PREFERRED_DEVICE = available_devices_[0] # eventually should not be used in our code
if 'gpu' in available_devices_[0].device_kind:
    GPU_AVAILABLE = True
else: 
    GPU_AVAILABLE = False
del jax_
del available_devices_

import os as os_
# Set numerical backend
AVAILABLE_NUMERICAL_BACKENDS = ['numpy', 'jax']
if "GGPEPS_BACKEND" in os_.environ:
    if os_.environ["GGPEPS_BACKEND"] in AVAILABLE_NUMERICAL_BACKENDS:
        PREFERRED_BACKEND = os_.environ["GGPEPS_BACKEND"]
    else:
        raise ValueError(f"Unknown backend: {os_.environ['GGPEPS_BACKEND']}")
elif GPU_AVAILABLE:
    PREFERRED_BACKEND = 'jax'
else:
    PREFERRED_BACKEND = 'numpy'
del os_

if PREFERRED_BACKEND == 'numpy':
    import numpy as xnp
elif PREFERRED_BACKEND == 'jax':
    import jax.numpy as xnp
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