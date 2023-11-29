# GPU or CPU
import jax
try:
    available_gpus = jax.devices('gpu') # Get the list of available GPUs
    PREFERRED_DEVICE = available_gpus[0] # Use the first available GPU as the preferred device
    GPU_AVAILABLE = True
except RuntimeError:
    # If GPUs are not available, fall back to the CPU.
    PREFERRED_DEVICE = jax.devices('cpu')[0]
    GPU_AVAILABLE = False
