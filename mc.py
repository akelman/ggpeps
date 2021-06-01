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
import lattice
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

        #This might change in the future if we implement different updates
        self.update=self.update_single_site


    def init_measurements(self):
        """Add empty measurement vectors to the measurement dictionary"""
        binsize = self.cfg.binsize
        self.obsdict["acceptance_prob"] = Measurement(
            "Acceptance Probablity", binsize)
        self.obsdict["energy"] = Measurement("Energy", binsize)
        self.obsdict["mag_energy"] = Measurement("Magnetic Energy", binsize)
        self.obsdict["el_energy"] = Measurement("Electric Energy", binsize)
        self.obsdict["wilson_00_11"] = Measurement("Wilson (0,0) 1x1", binsize)
        self.obsdict["polyakov_00_x"] = Measurement("Polyakov (0,0) x", binsize)
        #self.obsdict["cov_ferm"] = Measurement("Covariance Matrix fermions", binsize)

    def measure(self):
        """Measure the corresponding observables in the dictionary"""
        polyakov_loop = self.system.cfg.lattice.generate_polyakov_loop(
            (0, 0), lattice.Direction.X)
        wilson_loop = self.system.cfg.lattice.generate_wilson_loop(
            (0, 0), (1,1))

        self.obsdict["energy"].append(self.system.energy)
        self.obsdict["wilson_00_11"].append(np.real(self.system.compute_path(wilson_loop)))
        self.obsdict["polyakov_00_x"].append(np.real(self.system.compute_path(polyakov_loop)))
        #self.obsdict["cov_ferm"].append(self.system.compute_ferm_cov())
        self.obsdict["mag_energy"].append(self.system.mag_energy)
        self.obsdict["el_energy"].append(self.system.el_energy)

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

    def update_single_site(self):
        # Pick a site to update
        lattice=self.system.cfg.lattice
        nlinks=lattice.nlinks
        link_ind=np.random.randint(0,nlinks)
        # Uniformly pick a gauge to replace
        theta=self.system.gaugemgr.get_random_gauge_value()
        # Store the old values
        weight_old=self.system.weight
        weight_new=self.system.calculate_weight_attempt(link_ind,theta)
        if np.exp(weight_new - weight_old) > np.random.rand():
            # Accept
            self.obsdict["acceptance_prob"].append(1)
            self.system.update_gauge_ind(link_ind,theta)
        else:
            # Reject
            self.obsdict["acceptance_prob"].append(0)

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
        syscfg=self.system.cfg
        meas_steps=self.cfg.meas_steps
        warmup_steps=self.cfg.warmup_steps
        t=syscfg.paramdict["t"]
        y=syscfg.paramdict["y"]
        z=syscfg.paramdict["z"]
        fname_full = "data_L_{:02d}_gel_{:.3f}_gm_{:.3f}_gmag_{:.3f}_t_{:.3f}_y_{:.3f}_z_{:.3f}_wsteps_{:07d}_msteps_{:07d}.pkl.gz".format(
            syscfg.lattice.nx, syscfg.g_el, syscfg.g_gm, syscfg.g_mag, t, y, z, warmup_steps, meas_steps)
        fname_summary = "summary_L_{:02d}_gel_{:.3f}_gm_{:.3f}_gmag_{:.3f}_t_{:.3f}_y_{:.3f}_z_{:.3f}_wsteps_{:07d}_msteps_{:07d}.pkl".format(
            syscfg.lattice.nx, syscfg.g_el, syscfg.g_gm, syscfg.g_mag, t, y, z, warmup_steps, meas_steps)
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
        dest = {"name": [], "t": [], "y": [], "z": [], "g_el": [], "g_gm": [], "g_mag": [], "warmup_steps": [
        ], "meas_steps": [], "seed": [], "mean": [], "err": []}
        for key in self.obsdict.keys():
            dest['name'].append(key)
            dest['g_el'].append(self.system.cfg.g_el)
            dest['g_gm'].append(self.system.cfg.g_gm)
            dest['g_mag'].append(self.system.cfg.g_mag)
            dest['t'].append(self.system.cfg.paramdict["t"])
            dest['y'].append(self.system.cfg.paramdict["y"])
            dest['z'].append(self.system.cfg.paramdict["z"])
            dest['seed'].append(self.cfg.seed)
            dest['warmup_steps'].append(self.cfg.warmup_steps)
            dest['meas_steps'].append(self.cfg.meas_steps)
            dest["mean"].append(self.get_obs_mean(key))
            dest["err"].append(self.get_obs_mean_err(key))
        df = pd.DataFrame(dest)
        return df
