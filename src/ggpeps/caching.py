import os
import pickle
import logging

import numpy as np

import ggpeps

logger = logging.getLogger(ggpeps.LOGGER_NAME)

####################### Caching #######################


class Cache:
    def __init__(self, disable_cache: bool = False):
        self.cache_version = 0.2
        self.disable_cache: bool = disable_cache

        self.cache_data: dict = {
            "cache_version": self.cache_version,
            "ggpeps_version": ggpeps.__version__,
            "evaluator_manager": None,
            "energy": {},
            "energy_grad": {},
        }

    def paramvec2key(self, paramvec: np.ndarray):
        """Convert paramvect to bytes for use as a key in the cache."""
        return paramvec.data.tobytes()

    def key2paramvec(self, key: bytes) -> np.ndarray:
        """Convert bytes key to paramvec."""
        return np.frombuffer(key)

    def save_cache_file(self, dest_filepath: str) -> None:
        """Save the current cache to file."""

        if self.disable_cache:
            return

        # if os.path.exists(dest_filepath):
        #    logger.warning(f"Overwriting cache file {dest_filepath}")
        with open(dest_filepath, "wb") as outfile:
            pickle.dump(self.cache_data, outfile)
        return

    def add_obj_to_cache(self, obj_name: str, obj_val) -> None:
        if self.disable_cache:
            return
        if obj_name not in self.cache_data.keys():
            logger.warning(f"Cache does not support {obj_name}. Not adding to cache.")
        else:
            self.cache_data[obj_name] = obj_val
        return

    def load_obj_from_local_cache(self, obj_name: str):
        return self.cache_data[obj_name]

    def add_obs_to_cache(
        self, paramvec: np.ndarray, obs: str, val: float, save_to_file: bool = False
    ) -> None:
        if self.disable_cache:
            return

        key = self.paramvec2key(paramvec)
        obs_cache = self.cache_data[obs]
        obs_cache[key] = val

        obs_cache_len = len(obs_cache)
        if (
            obs_cache_len >= 1000 and not obs_cache_len % 500
        ):  # 1000 is an arbitrary threshold
            logger.warning(f"Cache for obs {obs} is large: {obs_cache_len} items.")

        # Save to pickle file
        if save_to_file:
            self.save_cache_file(ggpeps.global_vars["args"].save_cache_dest)

    def load_obs_from_local_cache(self, paramvec: np.ndarray, obs: str):
        if self.disable_cache:
            return
        if obs not in ["energy", "energy_grad"]:
            raise ValueError(f"Unknown observable {obs} is not in cache.")
        obs_cache = self.cache_data[obs]
        for key in obs_cache.keys():
            if np.allclose(self.key2paramvec(key), paramvec):
                return obs_cache[key]

        return

    def load_cache_file(self, cache_file: str) -> bool:
        # TODO: once we include other objects in the cache,
        #       this function should check that cached objects have the same configs
        #       (unless a change is deliberate...)
        success = False
        if self.disable_cache:
            return success
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as infile:
                cache_data = pickle.load(infile)
                if cache_data["cache_version"] == self.cache_version:
                    self.cache_data = cache_data
                    success = True
                    logger.info(f"Loaded cache file {cache_file}")
                else:
                    message = (
                        f"Cache version mismatch: "
                        + f"file {cache_file} has version {cache_data['cache_version']} "
                        + f"but the current code uses version {self.cache_version}. "
                        + f"Ignoring cached data."
                    )
                    logger.warning(message)
                    # TODO: we can probably recover some of the data
        return success


def remove_eval_manager_from_cache(cache_files: list[str]) -> None:
    """Remove the evaluator_manager from the cache files.

    Args:
        cache_files (list): paths to cache files
    """
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            cache = Cache("")
            cache.load_cache_file(cache_file)
            if "evaluator_manager" in cache.cache_data.keys():
                cache.cache_data["evaluator_manager"] = None
                cache.save_cache_file(cache_file)
    return


def main(args):
    remove_eval_manager_from_cache(args.files)
    return


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--files", nargs="+", help="path to cache files")
    args = parser.parse_args()

    main(args)
