import numpy as np
import time
import matplotlib.pyplot as plt
import logging
import pickle
import pandas as pd
import gzip
import utils
import copy
import ray
from measurement import Measurement

################################### Multiprocessing layer #######################


@ray.remote
def run_mc(runner_id, mc_cfg, system_cls, system_cfg):
    system = system_cls(system_cfg)
    system.initialize()
    mc = MonteCarloEstimator(mc_cfg, system)
    mc.simulate()
    return mc


class MonteCarloManager:
    def __init__(self, mc_cfg, system_cls, system_cfg, nrunner):
        self.nrunner = nrunner
        self.mc_cfg = mc_cfg
        self.system_cfg = system_cfg
        self.system_cls = system_cls

    def simulate(self):
        """Start the simulation of the runners"""
        resultvec = []
        if self.nrunner > 0:
            system_cfg_id = ray.put(self.system_cfg)
            for i in range(self.nrunner):
                cfg = copy.deepcopy(self.mc_cfg)
                cfg.seed = self.mc_cfg.seed+i
                cfg.meas_steps = self.mc_cfg.meas_steps//self.nrunner
                resultvec.append(run_mc.remote(
                    i, cfg, self.system_cls, system_cfg_id))
            resultvec = ray.get(resultvec)
            return self.collect(resultvec)
        else:
            system = self.system_cls(self.system_cfg)
            system.initialize()
            mc = MonteCarloEstimator(self.mc_cfg, system)
            mc.simulate()
            return mc

    def collect(self, resultvec):
        """Unify the results of the different Monte Carlo runners"""
        system = self.system_cls(self.system_cfg)
        dest = MonteCarloEstimator(self.mc_cfg, system)
        if len(resultvec) > 1:
            dest.obsdict = utils.mergeDict(
                resultvec[0].obsdict, resultvec[1].obsdict)
            for mc_runner in resultvec[2:]:
                dest.obsdict = utils.mergeDict(dest.obsdict, mc_runner.obsdict)
        else:
            dest = resultvec[0]
        return dest

################################### Monte Carlo runner and config ###############


class MonteCarloEstimatorConfig:
    """Monte Carlo Configuration"""

    def __init__(self):
       self.warmup_steps = None
       self._seed = None
       self.meas_steps = None
       self.binsize = 1

    @property
    def seed(self):
        if self._seed is None:
            self._seed = int(time.time())
        return self._seed

    @seed.setter
    def seed(self, seedval):
        self._seed = seedval

    def __str__(self):
        dest = ""
        dest += "Seed: {}\n".format(self.seed)
        dest += "Warmup steps: {}\n".format(self.warmup_steps)
        dest += "Measurement steps: {}\n".format(self.meas_steps)
        return dest


class MonteCarloEstimator:
    def __init__(self, cfg, system):
        self.cfg = cfg
        self.system = system
        self.obsdict = {}
        self.init_measurements()
        self.step = 0

    def init_measurements(self):
        """Add empty measurement vectors to the measurement dictionary"""
        binsize = self.cfg.binsize
        self.obsdict["acceptance_prob"] = Measurement(
            "Acceptance Probablity", binsize)
        self.obsdict["energy"] = Measurement("Energy", binsize)

    def measure(self):
        """Measure the corresponding observables in the dictionary"""
        self.obsdict["energy"].append(self.system.energy)

    def warmup(self):
        """Warm up phase without measurement"""
        while self.step < self.cfg.warmup_steps:
            if self.step % 1000 == 0:
                logging.debug("Warmup: {}".format(self.step))
            self.update()
            self.step += 1

    def run(self):
        """Meaurement phase phase (with measurement)"""
        while self.step < self.cfg.warmup_steps+self.cfg.meas_steps:
            if self.step % 1000 == 0:
                logging.debug("Run: {}".format(self.step))
            self.update()
            self.measure()
            self.step += 1

    def update(self):
        #TODO: Implement Update
        pass

    def simulate(self):
        self.warmup()
        self.run()

    #### Data management functions ####

    def get_obs_mean(self, obsname):
        if obsname in self.obsdict.keys():
            meas = self.obsdict[obsname]
            if meas is not None and len(meas) > 0:
                return meas.mean()
        return None

    def get_obs_mean_err(self, obsname):
        if obsname in self.obsdict.keys():
            meas = self.obsdict[obsname]
            if meas is not None and len(meas) > 0:
                return meas.mean_err()
        return None

    def get_obs_std(self, obsname):
        if obsname in self.obsdict.keys():
            meas = self.obsdict[obsname]
            if meas is not None and len(meas) > 0:
                return meas.std()
        return None

    def get_obs_var(self, obsname):
        if obsname in self.obsdict.keys():
            meas = self.obsdict[obsname]
            if meas is not None and len(meas) > 0:
                return meas.var()
        return None

    def save_summary(self, fname_summary):
        df_summary = self.summary()
        df_summary.to_pickle(fname_summary)

    def save_full(self, fname_full):
        data_full = {
            "version": utils.get_git_hash(),
            "rng_state": np.random.get_state(),
            "mc": self
        }
        with gzip.open(fname_full, "wb") as outfile:
            pickle.dump(data_full, outfile)

    def save(self):
        fname_full = "data_L_{:02d}_gel_{:.3f}_gm_{:.3f}_gmag_{:.3f}_wsteps_{:07d}_msteps_{:07d}.pkl.gz".format(
            self.system.cfg.lattice.nx, self.system.cfg.g_el, self.system.cfg.g_gm, self.system.cfg.g_mag, self.cfg.warmup_steps, self.cfg.meas_steps)
        fname_summary = "summary_L_{:02d}_gel_{:.3f}_gm_{:.3f}_gmag_{:.3f}_wsteps_{:07d}_msteps_{:07d}.pkl".format(
            self.system.cfg.lattice.nx, self.system.cfg.g_el, self.system.cfg.g_gm, self.system.cfg.g_mag, self.cfg.warmup_steps, self.cfg.meas_steps)
        self.save_full(fname_full)
        self.save_summary(fname_summary)

    ####### post-processing functions after the simulation ########################

    # TODO: Calculate Gradient

    #### Output (plots or on the commandline) ####

    def print_stats(self):
        for key in self.obsdict.keys():
            val = self.obsdict[key]
            if val is not None and len(val) > 0:
                print("<{}>".format(key), self.obsdict[key].mean())

    def summary(self):
        dest = {"name": [], "g_el": [], "g_gm": [], "g_mag": [], "warmup_steps": [
        ], "meas_steps": [], "seed": [], "mean": [], "err": []}
        for key in self.obsdict.keys():
            dest['name'].append(key)
            dest['g_el'].append(self.system.cfg.g_el)
            dest['g_gm'].append(self.system.cfg.g_gm)
            dest['g_mag'].append(self.system.cfg.g_mag)
            dest['seed'].append(self.cfg.seed)
            dest['warmup_steps'].append(self.cfg.warmup_steps)
            dest['meas_steps'].append(self.cfg.meas_steps)
            dest["mean"].append(self.get_obs_mean(key))
            dest["err"].append(self.get_obs_mean_err(key))
        df = pd.DataFrame(dest)
        return df
