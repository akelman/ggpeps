import ggpeps

from ggpeps.system.backend_base import BackendBase
from ggpeps.system.backend_numpy import BackendNumpy_Z2
from ggpeps.system.backend_jax import BackendJax_Z2

############## SELECT APPROPRIATE VERSION ##############
if ggpeps.PREFERRED_BACKEND == "jax":
    backend: BackendBase = BackendJax_Z2()
else:
    backend: BackendBase = BackendNumpy_Z2()
