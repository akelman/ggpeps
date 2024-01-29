import os
import pickle
import numpy as np

import ggpeps
from ggpeps import utils
from ggpeps import logger

####################### Caching #######################

class Cache:
    def __init__(self, cache_file: str = 'cache.pkl'):
        self.cache_version = 0.1
        self.cache_file: str = cache_file
        self.cache_data: dict = {'cache_version': self.cache_version,
                                 'git_hash': utils.get_git_hash(),
                                 'rng_state_internal_repr': None,
                                 'minimizer': None,
                                 'evaluator_manager': None,
                                 'system': None,
                                 'energy': {}, 
                                 'energy_grad': {}} 

    def paramvec2key(self, paramvec: np.ndarray):
        return paramvec.data.tobytes()

    def key2paramvec(self, key: bytes):
        return np.frombuffer(key)

    def save_cache_file(self):
        with open(self.cache_file, "wb") as outfile:
            pickle.dump(self.cache_data, outfile)

    def add_obj_to_cache(self, obj_name: str, obj_val):
        if obj_name not in self.cache_data.keys():
            logger.warn(f"Cache does not support {obj_name}. Not adding to cache.")
        else:
            self.cache_data[obj_name] = obj_val
        return

    def load_obj_from_local_cache(self, obj_name: str):
        return self.cache_data[obj_name]
    
    def add_obs_to_cache(self, paramvec: np.ndarray, obs: str, val: float, save_to_file: bool = True):
        key = self.paramvec2key(paramvec)
        obs_cache = self.cache_data[obs]
        obs_cache[key] = val

        obs_cache_len = len(obs_cache)
        if obs_cache_len > 1000 and obs_cache_len % 50: # 1000 is an arbitrary threshold
            logger.warn(f"Cache for obs {obs} is large: {obs_cache_len} items.")

        # Save to pickle file
        if save_to_file:
            self.save_cache_file()

    def load_obs_from_local_cache(self, paramvec: np.ndarray, obs: str):
        if obs not in ['energy', 'energy_grad']:
            raise ValueError(f"Unknown observable {obs} is not in cache.")
        obs_cache = self.cache_data[obs]
        for key in obs_cache.keys():
            if np.allclose(self.key2paramvec(key), paramvec):
                return obs_cache[key]
        if self.cache_data['evaluator_manager'] is not None:
            eval_manager = self.cache_data['evaluator_manager']
            new_params = np.reshape(np.copy(paramvec), (-1, 20))
            if np.allclose(eval_manager.system_cfg.paramvec, new_params):
                ggpeps.global_vars['minimizer'].evaluator_manager = eval_manager
                '''
                new_params = np.reshape(paramvec, (-1)) 
                #eval_manager.mc_cfg.set_rng_state_internal_repr(self.cache_data['rng_state_internal_repr'])
                eval_manager.evaluator.cfg.set_rng_state_internal_repr(self.cache_data['rng_state_internal_repr'])
                
                evaluator = self.cache_data['evaluator_manager'].resume_simulation()
                self.add_obs_to_cache(paramvec, obs, evaluator.get_obs_mean(obs))

                # save the rng state so that it can be used for the next evaluation
                self.cache_data['rng_state_internal_repr'] = eval_manager.evaluator.cfg.get_rng_state_internal_repr()
                self.cache_data['evaluator_manager'] = None # reset
                return evaluator.get_obs_mean(obs)
                '''
        return None

    def load_cache_file(self, cache_file: str):
        # TODO: once we include other objects in the cache,
        #       this function should check that cached objects have the same configs
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as infile:
                cache_data = pickle.load(infile)
                if cache_data['cache_version'] == self.cache_version:
                    self.cache_data = cache_data
                    logger.info(f"Loaded cache file {cache_file}")
                else:
                    logger.warn(f"Cache file {cache_file} has version {cache_data['cache_version']} \
                                but the current code uses version {self.cache_version}. Ignoring cached data.")
                    # TODO: we can probably recover some of the data
        return self.cache_data
