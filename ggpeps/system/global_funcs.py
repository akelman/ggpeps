import ggpeps

from .global_funcs_numpy import *
from .global_funcs_jax import *

############## SELECT APPROPRIATE VERSION ##############
if ggpeps.PREFERRED_BACKEND == 'jax':
    calculate_lognormvec = calculate_lognormvec_jax
    compute_grad_over_norm = compute_grad_over_norm_jax
    compute_el_grad_vec = compute_el_grad_vec_jax
    extract_partial_covmats = extract_partial_covmats_jax
else:
    calculate_lognormvec = calculate_lognormvec_numpy
    compute_grad_over_norm = compute_grad_over_norm_numpy
    compute_el_grad_vec = compute_el_grad_vec_numpy
    extract_partial_covmats = extract_partial_covmats_numpy

# TODO: create JAX versions of the following functions
#update_gauge_ind = update_gauge_ind_numpy
