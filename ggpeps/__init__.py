import os

# Global vars
global_vars = {}

# Logger setup
LOGGER_NAME = 'ggpeps'
logger_file = None

# Configure JAX 
import jax

# TODO: does this need to be here?
# The following line ensures that JAX is configured to 64-bit precision.
# Without this line, some of the precision tests do not pass.
jax.config.update("jax_enable_x64", True)

# GPU or CPU
available_devices = jax.devices() # available_gpus = jax.devices('gpu')
PREFERRED_DEVICE = available_devices[0] # eventually should not be used in our code
if 'gpu' in available_devices[0].device_kind:
    GPU_AVAILABLE = True
else: 
    GPU_AVAILABLE = False

# Set numerical backend
AVAILABLE_NUMERICAL_BACKENDS = ['numpy', 'jax']
if "GGPEPS_BACKEND" in os.environ:
    if os.environ["GGPEPS_BACKEND"] in AVAILABLE_NUMERICAL_BACKENDS:
        PREFERRED_BACKEND = os.environ["GGPEPS_BACKEND"]
    else:
        raise ValueError(f"Unknown backend: {os.environ['GGPEPS_BACKEND']}")
elif GPU_AVAILABLE:
    PREFERRED_BACKEND = 'jax'
else:
    PREFERRED_BACKEND = 'numpy'

if PREFERRED_BACKEND == 'numpy':
    import numpy as xnp
elif PREFERRED_BACKEND == 'jax':
    import jax.numpy as xnp
else:
    raise ValueError(f"Unknown backend: {PREFERRED_BACKEND}")
