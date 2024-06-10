# Global vars
global_vars = {}

# Logger setup
LOGGER_NAME = 'ggpeps'
logger_file = None

# GPU or CPU
import jax

# TODO: does this need to be here?
# The following line ensures that JAX is configured to 64-bit precision.
# Without this line, some of the precision tests do not pass.
jax.config.update("jax_enable_x64", True)

try:
    available_gpus = jax.devices('gpu') # Get the list of available GPUs
    PREFERRED_DEVICE = available_gpus[0] # Use the first available GPU as the preferred device
    GPU_AVAILABLE = True
except RuntimeError:
    # If GPUs are not available, fall back to the CPU.
    PREFERRED_DEVICE = jax.devices('cpu')[0]
    GPU_AVAILABLE = False