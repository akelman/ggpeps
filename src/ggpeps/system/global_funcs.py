import ggpeps

from .global_funcs_numpy import BackendNumpy_Z2
from .global_funcs_jax import BackendJax_Z2

############## SELECT APPROPRIATE VERSION ##############
if ggpeps.PREFERRED_BACKEND == "jax":
    backend = BackendJax_Z2()
else:
    backend = BackendNumpy_Z2()

# TODO: create JAX versions of the following functions
# update_gauge_ind = update_gauge_ind_numpy
