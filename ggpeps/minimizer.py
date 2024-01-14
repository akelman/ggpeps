import os
import pickle

import numpy as np
from scipy.optimize import minimize

from ggpeps import utils
from ggpeps import logger


####################### Caching #######################

class Cache:
    def __init__(self, cache_file: str = 'cache.pkl'):
        self.cache_version = 0.1
        self.cache_file: str = cache_file
        self.cache_data: dict = {'cache_version': self.cache_version,
                                 'git_hash': utils.get_git_hash(),
                                 'minimizer': None,
                                 'evaluator_manager': None,
                                 'system': None,
                                 'energy': {}, 
                                 'energy_grad': {}} 

    def paramvec2key(self, paramvec: np.ndarray):
        return paramvec.data.tobytes()

    def key2paramvec(self, key: bytes):
        return np.frombuffer(key)

    def add_to_cache(self, paramvec: np.ndarray, obs: str, val: float):
        key = self.paramvec2key(paramvec)
        obs_cache = self.cache_data[obs]
        obs_cache[key] = val

        #logger.debug(f"Added {obs} to cache for paramvec {paramvec}")
        if len(obs_cache) > 1000: # 1000 is an arbitrary threshold
            logger.warn(f"Cache for obs {obs} is large.")

        # Save to pickle file
        with open(self.cache_file, "wb") as outfile:
            pickle.dump(self.cache_data, outfile)

    def load_from_local_cache(self, paramvec: np.ndarray, obs: str):
        obs_cache = self.cache_data[obs]
        for key in obs_cache.keys():
            if np.allclose(self.key2paramvec(key), paramvec):
                return obs_cache[key]
        return None

    def load_cache_file(self, cache_file: str):
        # TODO: once we include other objects in the cache,
        #       this function should check that cached objects have the same configs
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as infile:
                self.cache_data = pickle.load(infile)
        return self.cache_data


####################### Minimizer #######################

class MinimizerResult:
    def __init__(self, paramvec, energygrad, method, value, converged, message):
        self.paramvec = paramvec
        self.energygrad = energygrad
        self.method = method
        self.value = value
        self.converged = converged
        self.message = message

    def __str__(self):
        dest = "==== Minimizer Result ====\n"
        dest += f"converged: {self.converged}\n"
        dest += f"Value: {self.value}\n"
        dest += f"Method: {self.method}\n"
        dest += f"Parameters: {self.paramvec}\n"
        dest += f"Message: {self.message}\n"
        dest += "==========================\n"
        return dest

class MinimizerConfig():

    def __init__(self):
        self.max_iter: int = 100
        self.min_grad: float = 1e-5
        self.alpha: float = 1e-2
        self._method: str = "CG"
        self.use_saved_cache: bool = True

    @property
    def method(self) -> str:
        return self._method

    @method.setter
    def method(self, val):
        self._method = val.upper()

class Minimizer():
    STOP_AFTER_CURRENT_ITERATION = False # this is a flag to catch interrupts to end minimization
    supported_methods = ["CG", "BFGS", "L-BFGS-B", "POWELL", "NELDER-MEAD", "TNC"]

    def __init__(self, cfg, evaluator):
        self.cfg = cfg
        # We use the polymorphism of python classes.
        # Below, we will have to be careful to only call valid functions
        self.evaluator = evaluator
        self._method: str = "CG"
        self.last_paramvec = None
        self.last_result = None
        self.min_result = None

        # Cache for the energy values and gradients
        self.cache = Cache()
        if self.cfg.use_saved_cache:
            self.cache.load_cache_file(self.cache.cache_file)

    def minimize(self):
        if self.cfg.method == "CUSTOM":
            return self.minimize_custom()
        elif self.cfg.method in self.supported_methods:
            return self.minimize_scipy()
        else:
            logger.error(f"Unkown minimization method '{self.cfg.method}'. Aborting...")
            return None

    def minimize_custom(self):
        paramvec = self.evaluator.system_cfg.paramvec

        for ind in range(self.cfg.max_iter):
            if self.last_paramvec is None or not np.allclose(self.last_paramvec,paramvec):
                # We copy here to get a new set of variables. We will change paramvec below and do not want to change last_paramvec
                self.last_paramvec = np.copy(paramvec)
                result = self.evaluator.simulate()

            energy = result.get_obs_mean("energy")
            grad_paramvec = result.get_obs_mean("energy_grad")
            
            max_grad_paramvec = np.max(np.abs(grad_paramvec))
            self.last_result = result

            # Update logs
            print_callback(ind, self)

            # Check if the maximum of the gradient is smaller than min_grad
            if max_grad_paramvec < abs(self.cfg.min_grad):
                message = f"Reached convergence: max grad paramvec < {self.cfg.min_grad}"
                logger.info(message)
                self.min_result = MinimizerResult(paramvec, self.cfg.method, energy, grad_paramvec, True, message)
                return self.min_result

            if self.STOP_AFTER_CURRENT_ITERATION:
                message = f"Recieved interrupt signal from user. Ending minimization."
                logger.info(message)
                self.min_result = MinimizerResult(paramvec, self.cfg.method, energy, grad_paramvec, True, message)
                return self.min_result

            #Adapt the parametervec according to the gradient
            # TODO: Implement stochastic reconfiguration

            # We have to use the internal name of the paramvec if we write to it since it is a property and not just an array
            self.evaluator.system_cfg.paramvec -= self.cfg.alpha * grad_paramvec

        message = "Reached maximum number of iterations without convergence."
        logger.warn(message)
        self.min_result = MinimizerResult(paramvec, self.cfg.method, energy, grad_paramvec, False, message)
        return self.min_result


    def minimize_scipy(self):
        
        # Energy wrapper
        def energy_wrapper(paramvec):
            # Check if value is stored in cache (e.g. from previous minimization)
            energy = self.cache.load_from_local_cache(paramvec, 'energy')
            if energy is not None:
                return energy

            if self.last_paramvec is None or not np.allclose(self.last_paramvec, paramvec):
                # We only set the parametervec and start the simulation if the parametervec is new
                self.last_paramvec = paramvec
                self.evaluator.system_cfg.paramvec = np.reshape(paramvec,(-1, self.evaluator.system_cfg._nparams))
                self.last_result = self.evaluator.simulate()
            
            energy = self.last_result.get_obs_mean('energy')
            self.cache.add_to_cache(paramvec, 'energy', energy)

            return energy
        
        # Jacobian wrapper
        def gradient_wrapper(paramvec):
            """Wrapper for the gradient of the total energy

            Args:
                paramvec (np.ndarray): parameters, arranged as a 1D array

            Returns:
                gradients (np.ndarray): gradients of the total energy, arranged as a 1D array
            """

            # Check if value is stored in cache (e.g. from previous minimization)
            parametergrad = self.cache.load_from_local_cache(paramvec, 'energy_grad')
            if parametergrad is not None:
                logger.debug('Found cached value for energy_grad')
                return parametergrad

            if self.last_paramvec is None or not np.allclose(self.last_paramvec, paramvec):
                # We only set the parametervec and start the simulation if the parametervec is new
                self.last_paramvec = paramvec
                #self.evaluator.mc_cfg.minimizer_mode = True # make sure to calculate derivatives
                self.evaluator.system_cfg.paramvec = np.reshape(paramvec,(-1, self.evaluator.system_cfg._nparams))
                self.last_result = self.evaluator.simulate()
            
            parametergrad = self.last_result.get_obs_mean('energy_grad')
            parametergrad = parametergrad.reshape((-1))
            self.cache.add_to_cache(paramvec, 'energy_grad', parametergrad)

            return parametergrad

        # Use the random initialization from the system.initialize as first guess.
        # We might want to change this later.
        paramvec = np.reshape(self.evaluator.system_cfg.paramvec, (-1))
        min_result = minimize(energy_wrapper,
                              paramvec,
                              method=self.cfg.method,
                              jac=gradient_wrapper,
                              callback=lambda x: print_callback(x, self),
                              options={"maxiter": self.cfg.max_iter})
        paramvec = min_result.x
        energygrad = min_result.jac
        energy = min_result.fun
        converged = min_result.success
        message = f"message: {min_result.message} Total iters: {min_result.nit}, function evals: {min_result.nfev}, jac evals: {min_result.njev}"

        dest = MinimizerResult(paramvec, energygrad, self.cfg.method, energy, converged, message)
        self.min_result = dest
        return dest

    def save(self, output_dir = "."):
        if self.min_result is not None:
            sys_cfg = self.evaluator.system_cfg

            #FIXME: Adapt the filenames here
            fname_mc_summary = f"summary_min_L_{sys_cfg.lattice.nx:02d}-{sys_cfg.lattice.ny:02d}_gel_{sys_cfg.g_el:.4f}_gmag_{sys_cfg.g_mag:.4f}_gint_{sys_cfg.g_int:.4f}_gmass_{sys_cfg.g_mass:.4f}_ncopy_{sys_cfg.ncopy:02d}_nlayer_{sys_cfg.nlayer:02d}.pkl"
            fname_result_min = f"result_min_L_{sys_cfg.lattice.nx:02d}-{sys_cfg.lattice.ny:02d}_gel_{sys_cfg.g_el:.4f}_gmag_{sys_cfg.g_mag:.4f}_gint_{sys_cfg.g_int:.4f}_gmass_{sys_cfg.g_mass:.4f}_ncopy_{sys_cfg.ncopy:02d}_nlayer_{sys_cfg.nlayer:02d}.pkl"

            self.last_result.save_summary(os.path.join(output_dir, fname_mc_summary))
            with open(os.path.join(output_dir, fname_result_min), "wb") as outfile:
                pickle.dump(self.min_result, outfile)


def print_callback(x, minimizer):
    
    res = minimizer.last_result
    paramvec = minimizer.evaluator.system_cfg.paramvec

    # If caching is on, the first time we call this function, the last_result will be None
    if res is None:
        logger.info("Callback message due to caching. The last_result is None.")
        return
    
    if minimizer.evaluator.type == 'exact':
        acceptance_prob = np.nan # acceptance_prob is undefined for exact contraction
    else:
        acceptance_prob = res.get_obs_mean("acceptance_prob")
    
    energy = res.get_obs_mean("energy")
    number_per_site = res.get_obs_mean("number_per_site")
    grad_paramvec = res.get_obs_mean("energy_grad")

    mass_energy = res.get_obs_mean("mass_energy")
    int_energy = res.get_obs_mean("int_energy")
    el_energy = res.get_obs_mean("el_energy")
    mag_energy = res.get_obs_mean("mag_energy")
    max_grad_paramvec = np.max(np.abs(grad_paramvec))

    if minimizer.cfg.method == 'CUSTOM':
        # We only have access to the iteration number if we are handling the minimization (via the CUSTOM method)
        logger.info(f"Iter: {x:03d}, Energy: {energy:.9f}, Occupation: {number_per_site:.6f}, Max grad paramvec: {max_grad_paramvec:.6f}, acceptance prob: {acceptance_prob:.5f}")
    else: 
        logger.info(f"Energy: {energy:.9f}, Occupation: {number_per_site:.6f}, Max grad paramvec: {max_grad_paramvec:.6f}, acceptance prob: {acceptance_prob:.6f}")

    logger.debug(f"el: {el_energy:.6f}, mag: {mag_energy:.6f}, mass: {mass_energy:.6f}, int: {int_energy:.6f}")
    logger.debug(f"Parametervec: {paramvec}")

    # If we're at the lowest energy seen so far, log the parameters
    #if current_iter == 0 or energy < lowest_energy:
    #    lowest_energy = energy
    #    #logger.info(f"New best energy. Parametervec: {paramvec}")   

    # If python recieves a signal to stop computation gracefully, we catch it here.
    # There have been recent developments within scipy's handling of these callbacks.
    # See (eg): https://github.com/scipy/scipy/issues/9412 
    #           https://github.com/scipy/scipy/issues/7306#issuecomment-301183706 
    # If/when this is implemented for scipy minimize, the handling of this should only be done here (not in the CUSTOM methods above)
    #if STOP_AFTER_CURRENT_ITERATION:
    #    return True  
    #    raise StopIteration

