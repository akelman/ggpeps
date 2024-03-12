import os
import ray
import copy
import gzip
import pickle
import logging

import numpy as np
import pandas as pd

import ggpeps
import ggpeps.utils as utils
import ggpeps.lattice as lattice

from ggpeps.evaluator import Evaluator
from ggpeps.measurement import Measurement

logger = logging.getLogger('ggpeps')

#################### Monte Carlo Estimator Config ###################

class MonteCarloEvaluatorConfig:
    """Monte Carlo Configuration

    This class manages the parameters of the MC simulation. 
    It is more convenient than passing an extensive number of parameters to the constructor.
    """
    def __init__(self):
        self.warmup_steps = None
        self._seed = None
        self._rng_state = None
        self.meas_steps = None
        self.binsize: int = 1
        self.minimizer_mode: bool = False
        self.update_size_per_step: int = 1 # this can be set anywhere from 1 to nlinks (inclusive)

        # Logging frequency
        self.warmup_log_freq: int = 5000 # log every X steps
        self.run_log_freq: int = 20000

    @property
    def seed(self):
        if self._seed is None:
            self._seed = np.random.randint(np.iinfo(np.int32).max)
            self._rng_state = np.random.RandomState(self._seed)
        return self._seed

    @seed.setter
    def seed(self, seedval):
        self._seed = seedval
        self._rng_state = np.random.RandomState(seedval)

    @property
    def rng_state(self):
        if self._rng_state is None:
            self._seed = np.random.randint(np.iinfo(np.int32).max)
            self._rng_state = np.random.RandomState(self._seed)
        return self._rng_state

    @rng_state.setter
    def rng_state(self, state):
        logger.error("MonteCarloEstimatorConfig: Do not set the state directly. Use a seed instead.")
        self.rng_state = None
        self.seed = None

    def get_rng_state_internal_repr(self):
        return self._rng_state.get_state()
    
    def set_rng_state_internal_repr(self, state_repr):
        self._rng_state.set_state(state_repr)
        return

    def __str__(self):
        dest = ""
        dest += f"Seed: {self.seed}\n"
        dest += f"Warmup steps: {self.warmup_steps}\n"
        dest += f"Measurement steps: {self.meas_steps}\n"
        dest += f"Update size: {self.update_size_per_step}\n"
        return dest


################################### Multiprocessing layer #######################
    
@ray.remote
def run_mc(runner_id, mc_cfg, system_cls, system_cfg):
    # TODO: get logger working within ray
    system = system_cls(copy.deepcopy(system_cfg))
    system.initialize()
    mc = MonteCarloEvaluator(mc_cfg, system)
    mc.evaluate()
    return mc


################################### Monte Carlo runner ###############

class MonteCarloEvaluator(Evaluator):
    """Class to take care of the MC simulation on a single runner
    """
    def __init__(self, evaluator_cfg: MonteCarloEvaluatorConfig, system):
        self.cfg = evaluator_cfg
        self.system = system
        self.obsdict: dict = {}
        self.step: int = 0
        self.evaluator_type = 'mc'
        self.init_measurements()

        # Choose how to update in each MC step
        # (This might change in the future if we implement different updates)
        if evaluator_cfg.update_size_per_step == self.system.cfg.lattice.nlinks:
            self.update = self.update_all_sites_single_site
        else:
            #self.update = self.update_single_site
            self.update = self.update_N_sites

    def init_measurements(self):
        """Add empty measurement vectors to the measurement dictionary"""
        binsize = self.cfg.binsize

        self.obsdict["acceptance_prob"] = Measurement("Acceptance Probablity", binsize)
        self.obsdict["energy"] = Measurement("Energy", binsize)
        self.obsdict["mag_energy"] = Measurement("Magnetic Energy", binsize)
        self.obsdict["el_energy"] = Measurement("Electric Energy", binsize)
        self.obsdict["int_energy"] = Measurement("Interaction Energy", binsize)
        self.obsdict["mass_energy"] = Measurement("Mass Energy", binsize)
        self.obsdict["mag_energy_op"] = Measurement("Magnetic Energy Operator (bare)", binsize)
        self.obsdict["el_energy_op"] = Measurement("Electric Energy Operator (bare)", binsize)
        self.obsdict["int_energy_op"] = Measurement("Interaction Energy Operator (bare)", binsize)
        self.obsdict["mass_energy_op"] = Measurement("Mass Energy Operator (bare)", binsize)
        self.obsdict["polyakov_00_x"] = Measurement("Polyakov (0,0) x", binsize)
        self.obsdict["norm"] = Measurement("Norm", binsize)
        self.obsdict["number_per_site"] = Measurement("Number per site", binsize)

        if self.cfg.minimizer_mode:
            self.obsdict["el_energy_op_grad"] = Measurement("Electric Energy Operator Gradient", binsize)
            self.obsdict["int_energy_op_grad"] = Measurement("Interaction Energy Operator Gradient", binsize)
            self.obsdict["mass_energy_op_grad"] = Measurement("Mass Energy Operator Gradient", binsize)
            self.obsdict["grad_norm"] = Measurement("Gradient of Norm/Norm", binsize)
            self.obsdict["energy_grad"] = Measurement("Gradient of Total Energy", binsize)
        #self.obsdict["cov_ferm"] = Measurement("Covariance Matrix fermions", binsize)

        # Wilson loops (of various sizes)
        sizes = self.system.cfg.lattice.generate_allowed_loop_dimensions()
        for size in sizes: 
            loop_name = f"wilson_loop_0-0_{size[0]}x{size[1]}"
            self.obsdict[loop_name] = Measurement(loop_name, binsize)
        

    def measure(self):
        """Measure the corresponding observables in the dictionary"""
        polyakov_loop = self.system.cfg.lattice.generate_polyakov_loop((0, 0), lattice.Direction.X)
        
        self.obsdict["polyakov_00_x"].append(np.real(self.system.compute_path(polyakov_loop)))
        #self.obsdict["cov_ferm"].append(self.system.compute_ferm_cov())
        self.obsdict["mag_energy_op"].append(self.system.mag_energy_op)
        self.obsdict["el_energy_op"].append(self.system.el_energy_op)
        self.obsdict["int_energy_op"].append(self.system.int_energy_op)
        self.obsdict["mass_energy_op"].append(self.system.mass_energy_op)

        # These values could be calculated in a post-processing step
        self.obsdict["energy"].append(self.system.energy)
        self.obsdict["el_energy"].append(self.system.el_energy)
        self.obsdict["mag_energy"].append(self.system.mag_energy)
        self.obsdict["int_energy"].append(self.system.int_energy)
        self.obsdict["mass_energy"].append(self.system.mass_energy)
        self.obsdict["norm"].append(self.system.calculate_lognorm(all_factors=True))
        self.obsdict["number_per_site"].append(self.system.number_per_site)

        if self.cfg.minimizer_mode:
            self.obsdict["el_energy_op_grad"].append(self.system.el_energy_op_grad_vec)
            self.obsdict["int_energy_op_grad"].append(self.system.int_energy_op_grad_vec)
            self.obsdict["mass_energy_op_grad"].append(self.system.mass_energy_op_grad_vec)
            self.obsdict["grad_norm"].append(self.system.compute_grad_norm_vec())
            self.obsdict["energy_grad"].append(self.energy_gradient_mc())
        
        # Wilson loops
        sizes = self.system.cfg.lattice.generate_allowed_loop_dimensions()
        loops = self.system.cfg.lattice.generate_all_wilson_loops((0,0), sizes)
        for k in range(len(sizes)):
            loop_name = f"wilson_loop_0-0_{sizes[k][0]}x{sizes[k][1]}"
            self.obsdict[loop_name].append(np.real(self.system.compute_path(loops[k])))

    def energy_gradient_mc(self):
        # Compute the energy gradient from the MC results
        meas_grad_over_norm = self.obsdict["grad_norm"]

        # Gradient of the magnetic energy
        meas_mag_energy_op = self.obsdict["mag_energy_op"]
        prod_mag_energy_grad = meas_mag_energy_op * meas_grad_over_norm
        mag_energy_op_grad = prod_mag_energy_grad.mean() - meas_mag_energy_op.mean() * meas_grad_over_norm.mean()
        # Add the constants back into the expression of the magnetic energy
        mag_energy_grad = - 2 * self.system.cfg.g_mag * mag_energy_op_grad

        # Gradient of the electric energy
        meas_el_energy_op = self.obsdict["el_energy_op"]
        meas_el_energy_op_grad = self.obsdict["el_energy_op_grad"]
        prod_el_energy_grad = meas_el_energy_op * meas_grad_over_norm
        el_energy_op_grad = prod_el_energy_grad.mean() - meas_el_energy_op.mean()*meas_grad_over_norm.mean() + meas_el_energy_op_grad.mean()
        # Add the constants back into the expression of the electric energy
        el_energy_grad = - 2 * self.system.cfg.g_el * el_energy_op_grad

        # Gradient of the interaction energy
        meas_int_energy_op = self.obsdict["int_energy_op"]
        meas_int_energy_op_grad = self.obsdict["int_energy_op_grad"]
        prod_int_energy_grad = meas_int_energy_op * meas_grad_over_norm
        int_energy_op_grad = prod_int_energy_grad.mean() - meas_int_energy_op.mean()*meas_grad_over_norm.mean() + meas_int_energy_op_grad.mean()
        # Add the constants back into the expression of the interaction energy
        int_energy_grad = self.system.cfg.g_int * int_energy_op_grad

        # Gradient of the mass energy
        meas_mass_energy_op = self.obsdict["mass_energy_op"]
        meas_mass_energy_op_grad = self.obsdict["mass_energy_op_grad"]
        prod_mass_energy_grad = meas_mass_energy_op * meas_grad_over_norm
        mass_energy_op_grad = prod_mass_energy_grad.mean() - meas_mass_energy_op.mean()*meas_grad_over_norm.mean() + meas_mass_energy_op_grad.mean()
        # Add the constants back into the expression of the mass energy
        mass_energy_grad = self.system.cfg.g_mass * mass_energy_op_grad

        return mag_energy_grad + el_energy_grad + int_energy_grad + mass_energy_grad

    def warmup(self):
        """Warm up phase without measurement"""
        logger.debug("Starting MC warmup")
        while self.step < self.cfg.warmup_steps:
            if self.step % self.cfg.warmup_log_freq == 0:
                logger.debug(f"Warmup: {self.step}")
            self.update()
            self.step += 1
        logger.debug("Finished MC warmup")

    def run(self):
        """Meaurement phase"""
        logger.debug("Starting MC measurement")
        while self.step < self.cfg.warmup_steps + self.cfg.meas_steps:
            if self.step % self.cfg.run_log_freq == 0:
                logger.debug(f"Run: {self.step}")
            self.update()
            self.measure()
            self.step += 1
        logger.debug("Finished MC measurement")

    def update_single_site(self):
        """Update for the MC simulation.
        This updates randomly chooses a single site and updates it.
        The update is local. The new gauge field value is drawn uniformly from the distribution of possible gauge fields (according to the gauge group).
        """
        # Pick a site to update
        lattice = self.system.cfg.lattice
        nlinks = lattice.nlinks
        link_ind = self.cfg.rng_state.randint(0, nlinks)
        # Uniformly pick a gauge value
        theta = self.system.gaugemgr.get_random_gauge_value(self.cfg.rng_state)
        # Store the old values
        weight_old = self.system.weight
        weight_new = self.system.calculate_weight_attempt(link_ind, theta)
        if np.exp(weight_new - weight_old) > self.cfg.rng_state.rand():
            # Accept
            self.obsdict["acceptance_prob"].append(1)
            self.system.update_gauge_ind(link_ind, theta)
        else:
            # Reject
            self.obsdict["acceptance_prob"].append(0)

    def update_all_sites_single_site(self):
        """Update for the MC simulation.
        This updates iterates over all lattice sites and updates every site once.
        The update is local. 
        The new gauge field value is drawn uniformly from the distribution of possible gauge fields (according to the gauge group).
        """
        # Pick a site to update
        lattice = self.system.cfg.lattice
        nlinks = lattice.nlinks
        for i in range(nlinks):
            # Uniformly pick a gauge to replace
            theta = self.system.gaugemgr.get_random_gauge_value(self.cfg.rng_state)
            # Store the old values
            weight_old = self.system.weight
            weight_new = self.system.calculate_weight_attempt(i, theta)
            if np.exp(weight_new - weight_old) > self.cfg.rng_state.rand():
                # Accept
                self.obsdict["acceptance_prob"].append(1)
                self.system.update_gauge_ind(i, theta)
            else:
                # Reject
                self.obsdict["acceptance_prob"].append(0)
        
    def update_N_sites(self):
        """Update for the MC simulation.
        This updates iterates over N lattice sites and updates every site once.
        The update is local.
        The new gauge field value is drawn uniformly from the distribution of possible gauge fields (according to the gauge group).
        """
        nlinks = self.system.cfg.lattice.nlinks
        links_inds = self.cfg.rng_state.choice([k for k in range(nlinks)], self.cfg.update_size_per_step, replace=False)
        for link_ind in links_inds:
            # Uniformly pick a gauge to replace
            theta = self.system.gaugemgr.get_random_gauge_value(self.cfg.rng_state)
            # Store the old values
            weight_old = self.system.weight
            weight_new = self.system.calculate_weight_attempt(link_ind, theta)
            if np.exp(weight_new - weight_old) > self.cfg.rng_state.rand():
                # Accept
                self.obsdict["acceptance_prob"].append(1)
                self.system.update_gauge_ind(link_ind, theta)
            else:
                # Reject
                self.obsdict["acceptance_prob"].append(0)

    def evaluate(self):
        """Main routine to start a Monte Carlo simulation.
        """
        self.warmup()
        self.run()

    #### Data management functions ####

    def get_obs_mean(self, obsname: str):
        """Returns the mean value of an observable

        Args:
            obsname (str): Name of observable

        Returns:
            float: Mean value of the observable
        """
        if obsname in self.obsdict.keys():
            meas = self.obsdict[obsname]
            if meas is not None and len(meas) > 0:
                return meas.mean()
        return None

    def get_obs_mean_err(self, obsname: str):
        """Returns the error on the mean of an observable

        Args:
            obsname (str): Name of observable

        Returns:
            float: Error on mean of observable
        """
        if obsname in self.obsdict.keys():
            meas = self.obsdict[obsname]
            if meas is not None and len(meas) > 0:
                return meas.mean_err()
        return None

    def get_obs_std(self, obsname: str):
        """Returns the standard deviation of an observable

        Args:
            obsname (str): Name of observable

        Returns:
            float: Standard deviation of an observable
        """
        if obsname in self.obsdict.keys():
            meas = self.obsdict[obsname]
            if meas is not None and len(meas) > 0:
                return meas.std()
        return None

    def get_obs_var(self, obsname: str):
        """Returns the variance of an observable

        Args:
            obsname (str): Name of observable

        Returns:
            float: Variance of the observable
        """
        if obsname in self.obsdict.keys():
            meas = self.obsdict[obsname]
            if meas is not None and len(meas) > 0:
                return meas.var()
        return None

    def save_summary(self, fname_summary: str):
        """Save the summary to disk

        Args:
            fname_summary (str): Filename of the summary
        """
        df_summary = self.summary()
        df_summary.to_pickle(fname_summary)

    def save_full(self, fname_full: str):
        """Save the full MonteCarloEstimator

        Args:
            fname_full (str): Filename of the full MonteCarloEstimator
        """
        data_full = {
            "version": utils.get_git_hash(),
            "rng_state": self.cfg.rng_state.get_state(),
            "mc": self
        }
        with gzip.open(fname_full, "wb") as outfile:
            pickle.dump(data_full, outfile)

    def save(self, output_dir = "."):
        """Convenience function to combine saving the MonteCarloEstimator and the summary of the observables
        """
        syscfg = self.system.cfg
        meas_steps = self.cfg.meas_steps
        warmup_steps = self.cfg.warmup_steps

        fname_full = f"data_mc_L_{syscfg.lattice.nx:02d}-{syscfg.lattice.ny:02d}_gel_{syscfg.g_el:.3f}_gmag_{syscfg.g_mag:.3f}_gint_{syscfg.g_int:.3f}_nlayer_{syscfg.nlayer:02d}_wsteps_{warmup_steps:07d}_msteps_{meas_steps:07d}.pkl.gz"
        fname_summary = f"summary_mc_L_{syscfg.lattice.nx:02d}-{syscfg.lattice.ny:02d}_gel_{syscfg.g_el:.3f}_gmag_{syscfg.g_mag:.3f}_gint_{syscfg.g_int:.3f}_nlayer_{syscfg.nlayer:02d}_wsteps_{warmup_steps:07d}_msteps_{meas_steps:07d}.pkl"
        
        self.save_full(os.path.join(output_dir, fname_full))
        self.save_summary(os.path.join(output_dir, fname_summary))

    #### Output (plots or on the commandline) ####

    def print_stats(self):
        """Print a quick summary of the observables
        """
        for key in self.obsdict.keys():
            val = self.obsdict[key]
            if val is not None and len(val) > 0:
                logger.info(f"<{key}>: {self.obsdict[key].mean()}")

    def summary(self):
        """Generate a summary of the simulation in the form of a pandas dataframe

        Returns:
            pd.DataFrame: Pandas dataframe with a summary of all results
        """
        dest = {
            "name": [],
            "nx":[],
            "ny":[],
            "paramvec":[],
            "ncopy":[],
            "nlayer":[],
            "g_el": [],
            "g_mag": [],
            "g_int": [],
            "g_mass": [],
            "warmup_steps": [],
            "meas_steps": [],
            "seed": [],
            "mean": [],
            "err": []
        }
        for key in self.obsdict.keys():
            dest['name'].append(key)
            dest['nx'].append(self.system.cfg.lattice.nx)
            dest['ny'].append(self.system.cfg.lattice.ny)
            dest['g_el'].append(self.system.cfg.g_el)
            dest['g_int'].append(self.system.cfg.g_int)
            dest['g_mag'].append(self.system.cfg.g_mag)
            dest['g_mass'].append(self.system.cfg.g_mass)
            dest['paramvec'].append(self.system.cfg.paramvec)
            dest['ncopy'].append(self.system.cfg.ncopy)
            dest['nlayer'].append(self.system.cfg.nlayer)
            dest['seed'].append(self.cfg.seed)
            dest['warmup_steps'].append(self.cfg.warmup_steps)
            dest['meas_steps'].append(self.cfg.meas_steps)
            dest["mean"].append(self.get_obs_mean(key))
            dest["err"].append(self.get_obs_mean_err(key))
        df = pd.DataFrame(dest)
        return df
