# Global vars
global_vars = {}

# Logger setup
logger_name = 'ggpeps' # this is hard-coded in several places in the code
logger_file = None

# GPU or CPU
import jax
# config.update("jax_enable_x64", True) # TODO: does this need to be here
try:
    available_gpus = jax.devices('gpu') # Get the list of available GPUs
    PREFERRED_DEVICE = available_gpus[0] # Use the first available GPU as the preferred device
    GPU_AVAILABLE = True
except RuntimeError:
    # If GPUs are not available, fall back to the CPU.
    PREFERRED_DEVICE = jax.devices('cpu')[0]
    GPU_AVAILABLE = False