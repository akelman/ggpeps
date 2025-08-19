import ggpeps

from ggpeps.system.backend_base import BackendBase
from ggpeps.system.backend_numpy import BackendNumpy_Z2
from ggpeps.system.backend_jax import BackendJax_Z2

############## SELECT APPROPRIATE VERSION ##############
backend: BackendBase
if ggpeps.PREFERRED_BACKEND == "jax":
    backend = BackendJax_Z2()
else:
    backend = BackendNumpy_Z2()
