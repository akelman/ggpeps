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
available_gpus = jax.devices('gpu')
if len(available_gpus) > 0:
    PREFERRED_DEVICE = available_gpus[0] # we only use one GPU
    GPU_AVAILABLE = True
else: # If GPUs are not available, fall back to the CPU.
    PREFERRED_DEVICE = jax.devices('cpu')[0]
    GPU_AVAILABLE = False

# Set numerical backend
AVAILABLE_NUMERICAL_BACKENDS = ['numpy', 'jax']
PREFERRED_BACKEND = 'numpy'
if GPU_AVAILABLE:
    PREFERRED_BACKEND = 'jax'

if PREFERRED_BACKEND == 'numpy':
    import numpy as xnp
elif PREFERRED_BACKEND == 'jax':
    import jax.numpy as xnp
else:
    raise ValueError(f"Unknown backend: {PREFERRED_BACKEND}")
