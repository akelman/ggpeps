import ggpeps

from ggpeps.system.backend_base import BackendBase
from ggpeps.system.backend_numpy import BackendNumpy
from ggpeps.system.backend_jax import BackendJax

############## SELECT APPROPRIATE VERSION ##############
backend: BackendBase
if ggpeps.PREFERRED_BACKEND == "jax":
    backend = BackendJax()
else:
    backend = BackendNumpy()
