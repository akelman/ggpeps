import os
import pickle
import logging

import numpy as np

import ggpeps
from ggpeps import utils

logger = logging.getLogger(ggpeps.LOGGER_NAME)

####################### Caching #######################

class Cache:
    def __init__(self, mode: str, cache_file: str = 'cache.pkl'):
        self.cache_version = 0.1
        self.cache_file: str = cache_file
        self.cache_data: dict = {'cache_version': self.cache_version,
                                 'git_hash': utils.get_git_hash(),
                                 'mode': mode,
                                 'evaluator_manager': None,
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
        
        # if cached value is not found, but an eval manager is present, 
        # update the minimizer to use that eval manager
        # ISSUE: note that this overwrites the current eval manager, including when the config (e.g. meas_steps) has changed
        # TODO: this is a hack, we should have a better way to handle this
        if self.cache_data['evaluator_manager'] is not None:
            eval_manager = self.cache_data['evaluator_manager']
            if eval_manager.nrunner == ggpeps.global_vars['minimizer'].evaluator_manager.nrunner:
                if np.allclose(eval_manager.system_cfg.paramvec, np.reshape(paramvec, (-1, 20)) ):
                    ggpeps.global_vars['minimizer'].evaluator_manager = eval_manager
        return None

    def load_cache_file(self, cache_file: str) -> bool:
        # TODO: once we include other objects in the cache,
        #       this function should check that cached objects have the same configs
        #       (unless a change is deliberate...)
        success = False
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as infile:
                cache_data = pickle.load(infile)
                if cache_data['cache_version'] == self.cache_version:
                    self.cache_data = cache_data
                    success = True
                    logger.info(f"Loaded cache file {cache_file}")
                else:
                    message = f"Cache version or mode mismatch: " \
                            + f"file {cache_file} has version {cache_data['cache_version']} " \
                            + f"but the current code uses version {self.cache_version}; " \
                            + f"file {cache_file} has mode {cache_data['mode']} " \
                            + f"but the current run uses mode {ggpeps.global_vars['args'].mode}. " \
                            + f"Ignoring cached data."
                    logger.warn(message)
                    # TODO: we can probably recover some of the data
        return success

def remove_eval_manager_from_cache(cache_files):
    """Remove the evaluator_manager from the cache file.

    Args:
        cache_file (str): path to cache file
    """
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            cache = Cache('', cache_file)
            cache.cache_data['evaluator_manager'] = None
            cache.save_cache_file()
    return

def main(args):
    remove_eval_manager_from_cache(args.files)
    return


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--files", nargs="+", help="path to cache files")
    args = parser.parse_args()

    main(args)