import os
import gzip
import pickle
import logging

import pandas as pd
import numpy as np
import copy

import ggpeps
import ggpeps.utils as utils
import ggpeps.lattice as lattice

from ggpeps.evaluator import Evaluator
from ggpeps.measurement import Measurement

logger = logging.getLogger(ggpeps.LOGGER_NAME)


########### Non-equilibrium variational Monte Carlo Estimator Config ###########


class NEVMC_EvaluatorConfig:
    """Non-equilibrium variational Monte Carlo (NEVMC) Configuration

    This class manages the parameters of the NEVMC simulation.
    It is more convenient than passing an extensive number of parameters to the constructor.
    """

    def __init__(
        self,
        warmup_steps: int = 10000,
        meas_steps: int = 10000,
        binsize: int = 1,
        compute_grads: bool = False,
        update_size_per_step: int = 1,
        warmup_log_freq: int = 5000,
        run_log_freq: int = 20000,
    ) -> None:

        self.warmup_steps = warmup_steps
        self.meas_steps = meas_steps
        self.binsize = binsize
        self.compute_grads = compute_grads
        self.update_size_per_step = update_size_per_step  # this can be set anywhere from 1 to nlinks (inclusive)

        self._seed = None
        self._rng_state = None

        self.first_warmup = False
        self.scanning = False

        # Logging frequency
        self.warmup_log_freq: int = warmup_log_freq  # log every X steps
        self.run_log_freq: int = run_log_freq

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


############### Non-equilibrium variational Monte Carlo runner ###############


class NEVMC_Evaluator(Evaluator):
    """Class to take care of the NEVMC simulation on a single runner"""

    evaluator_type = "nevmc"

    def __init__(
        self,
        evaluator_cfg: NEVMC_EvaluatorConfig,
        system,
        store_gauge,
        store_weights,
        store_work,
    ):
        super().__init__(evaluator_cfg, system)
        self.store_gauge = store_gauge
        self.store_weights = store_weights
        self.store_work = store_work

        self.step: int = 0
        self.init_measurements()

        self.NEVMC_update = self.NEVMC_update_N_sites
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
        self.obsdict["chem_energy"] = Measurement("Chemical Energy", binsize)
        self.obsdict["mag_energy_op"] = Measurement("Magnetic Energy Operator (bare)", binsize)
        self.obsdict["el_energy_op"] = Measurement("Electric Energy Operator (bare)", binsize)
        self.obsdict["int_energy_op"] = Measurement("Interaction Energy Operator (bare)", binsize)
        self.obsdict["mass_energy_op"] = Measurement("Mass Energy Operator (bare)", binsize)
        self.obsdict["polyakov_00_x"] = Measurement("Polyakov (0,0) x", binsize)
        self.obsdict["norm"] = Measurement("Norm", binsize)
        if self.system.cfg.num_fermionic_layer > 0:  # We only compute occupations if there are fermionic layers
            self.obsdict["all_occupations"] = Measurement("All Occupations (after PH)", binsize)
            self.obsdict["average_occupation"] = Measurement("Average Occupation", binsize)

        # Wilson loops (of various sizes)
        sizes = self.system.cfg.lattice.generate_allowed_loop_dimensions()
        for size in sizes:
            loop_name = f"wilson_loop_0-0_{size[0]}x{size[1]}"
            self.obsdict[loop_name] = Measurement(loop_name, binsize)

        # Meson strings
        max_string = 1 + max(self.system.cfg.lattice.nx, self.system.cfg.lattice.ny) // 2
        for k in range(1, max_string):
            self.obsdict[f"square_string_0-0_{k}x{k}"] = Measurement(f"square_string_0-0_{k}x{k}", binsize)

        if self.cfg.compute_grads:
            ### beg NEVMC ###
            self.obsdict["work"] = Measurement("Work", binsize)
            ### end NEVMC ###
            self.obsdict["el_energy_op_grad"] = Measurement("Electric Energy Operator Gradient", binsize)
            self.obsdict["int_energy_op_grad"] = Measurement("Interaction Energy Operator Gradient", binsize)
            self.obsdict["mass_energy_op_grad"] = Measurement("Mass Energy Operator Gradient", binsize)
            self.obsdict["chem_energy_op_grad"] = Measurement("Chemical Energy Operator Gradient", binsize)
            self.obsdict["grad_norm"] = Measurement("Gradient of Norm/Norm", binsize)
            self.obsdict["energy_grad"] = Measurement("Gradient of Total Energy", binsize)

    def measure(self):
        """Measure the corresponding observables in the dictionary"""
        polyakov_loop = self.system.cfg.lattice.generate_polyakov_loop((0, 0), lattice.Direction.X)

        self.obsdict["polyakov_00_x"].append(np.real(self.system.compute_path(polyakov_loop)))
        # self.obsdict["cov_ferm"].append(self.system.ferm_covmat_vec)
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
        self.obsdict["chem_energy"].append(self.system.chem_energy)
        self.obsdict["norm"].append(np.sum(self.system.calculate_lognormvec(all_factors=True)))
        if self.system.cfg.num_fermionic_layer > 0:  # We only compute occupations if there are fermionic layers
            self.obsdict["average_occupation"].append(self.system.average_occupation())
            self.obsdict["all_occupations"].append(self.system.occupations_before_ph)

        # Wilson loops
        sizes = self.system.cfg.lattice.generate_allowed_loop_dimensions()
        loops = self.system.cfg.lattice.generate_all_wilson_loops((0, 0), sizes)
        for k in range(len(sizes)):
            loop_name = f"wilson_loop_0-0_{sizes[k][0]}x{sizes[k][1]}"
            self.obsdict[loop_name].append(np.real(self.system.compute_path(loops[k])))

        # Meson strings
        max_string = 1 + max(self.system.cfg.lattice.nx, self.system.cfg.lattice.ny) // 2
        strings = [self.system.cfg.lattice.generate_L_string((0, 0), (k, k)) for k in range(1, max_string)]
        for k in range(1, max_string):
            string_name = f"square_string_0-0_{k}x{k}"
            self.obsdict[string_name].append(self.system.meson_string(strings[k - 1]))

        if self.cfg.compute_grads:
            self.obsdict["el_energy_op_grad"].append(self.system.el_energy_op_grad_vec)
            self.obsdict["int_energy_op_grad"].append(self.system.int_energy_op_grad_vec)
            self.obsdict["mass_energy_op_grad"].append(self.system.mass_energy_op_grad_vec)
            self.obsdict["chem_energy_op_grad"].append(self.system.chem_energy_op_grad_vec)
            self.obsdict["grad_norm"].append(self.system.grad_over_norm_vec)

        return

    def energy_gradient_mc(self):
        # Compute the energy gradient from the MC results
        meas_grad_over_norm = self.obsdict["grad_norm"]

        # Gradient of the magnetic energy
        meas_mag_energy_op = self.obsdict["mag_energy_op"]
        prod_mag_energy_grad = meas_mag_energy_op * meas_grad_over_norm
        mag_energy_op_grad = prod_mag_energy_grad.mean() - meas_mag_energy_op.mean() * meas_grad_over_norm.mean()
        # Add the constants back into the expression of the magnetic energy
        mag_energy_grad = -2 * self.system.cfg.g_mag * mag_energy_op_grad

        # Gradient of the electric energy
        meas_el_energy_op = self.obsdict["el_energy_op"]
        meas_el_energy_op_grad = self.obsdict["el_energy_op_grad"]
        prod_el_energy_grad = meas_el_energy_op * meas_grad_over_norm
        el_energy_op_grad = (
            prod_el_energy_grad.mean()
            - meas_el_energy_op.mean() * meas_grad_over_norm.mean()
            + meas_el_energy_op_grad.mean()
        )
        # Add the constants back into the expression of the electric energy
        el_energy_grad = -2 * self.system.cfg.g_el * el_energy_op_grad

        # Gradient of the interaction energy
        meas_int_energy_op = self.obsdict["int_energy_op"]
        meas_int_energy_op_grad = self.obsdict["int_energy_op_grad"]
        prod_int_energy_grad = meas_int_energy_op * meas_grad_over_norm
        int_energy_op_grad = (
            prod_int_energy_grad.mean()
            - meas_int_energy_op.mean() * meas_grad_over_norm.mean()
            + meas_int_energy_op_grad.mean()
        )
        # Add the constants back into the expression of the interaction energy
        int_energy_grad = self.system.cfg.g_int * int_energy_op_grad

        # Gradient of the mass energy
        meas_mass_energy_op = self.obsdict["mass_energy_op"]
        meas_mass_energy_op_grad = self.obsdict["mass_energy_op_grad"]
        prod_mass_energy_grad = meas_mass_energy_op * meas_grad_over_norm
        mass_energy_op_grad = (
            prod_mass_energy_grad.mean()
            - meas_mass_energy_op.mean() * meas_grad_over_norm.mean()
            + meas_mass_energy_op_grad.mean()
        )
        # Add the constants back into the expression of the mass energy
        mass_energy_grad = self.system.cfg.g_mass * mass_energy_op_grad

        return mag_energy_grad + el_energy_grad + int_energy_grad + mass_energy_grad

    def warmup(self):
        """Warm up phase without measurement"""
        logger.debug("Starting NEVMC warmup")
        while self.step < self.cfg.warmup_steps:
            if self.step % self.cfg.warmup_log_freq == 0:
                logger.debug(f"Warmup: {self.step}")
            self.update()
            self.step += 1
        logger.debug("Finished NEVMC warmup")

    #################################################################################################################
    ### beg NEVMC ###

    def evaluate(self):

        if self.cfg.first_warmup:
            self.warmup()
            # warmup uses the standard update function
            self.run0(self.cfg.warmup_steps)
            self.store_work = [0] * len(self.store_weights)

        else:
            if self.cfg.scanning:
                for i in range(len(self.store_weights)):
                    self.scan_cfgs(i)
            else:
                self.run()

    def run0(self, warmsteps):
        logger.debug("Starting first NEVMC measurement")
        while self.step < warmsteps + self.cfg.meas_steps:
            if self.step % self.cfg.run_log_freq == 0:
                logger.debug(f"Run: {self.step}")
            self.NEVMC_update()

            self.measure()
            self.step += 1

        total_grad = self.energy_gradient_mc()
        self.obsdict["energy_grad"].extend([total_grad] * len(self.obsdict["energy"]))

        logger.debug("Finished first NEVMC measurement")

    def run(self):
        """Meaurement phase"""
        logger.debug("Starting NEVMC measurement")
        idx = 0
        while self.step < self.cfg.meas_steps:
            if self.step % self.cfg.run_log_freq == 0:
                logger.debug(f"Run: {self.step}")
            self.NEVMC_update(idx=idx, NEQ=True)

            self.measure_nograd()
            self.step += 1
            idx += 1

        logger.debug("Finished NEVMC measurement")

    def NEVMC_update_N_sites(self, idx=0, NEQ=False):
        links_inds = self.cfg.rng_state.choice(
            self.system.cfg.lattice.comp_tree,
            self.cfg.update_size_per_step,
            replace=False,
        )

        if NEQ:
            curr_gauge = self.store_gauge[idx]

            self.system.update_gauge_full_system(curr_gauge)
            tmp = copy.deepcopy(self.system.weight)

        for link_ind in links_inds:
            # Uniformly pick a gauge to replace
            theta = self.system.cfg.gaugemgr.get_random_gauge_value(self.cfg.rng_state)
            # Store the old values
            if NEQ:
                weight_old = tmp
            else:
                weight_old = self.system.weight
            weight_new = self.system.calculate_weight_attempt(link_ind, theta)

            if np.exp(weight_new - weight_old) > self.cfg.rng_state.rand():
                # Accept
                self.obsdict["acceptance_prob"].append(1)
                self.system.update_gauge_ind(link_ind, theta)

                if NEQ:
                    self.store_weights[idx] = weight_new
                    self.store_gauge[idx] = copy.deepcopy(self.system._gaugefieldvec)
                else:
                    self.store_weights.append(weight_new)
                    self.store_gauge.append(copy.deepcopy(self.system._gaugefieldvec))

            else:
                # Reject
                self.obsdict["acceptance_prob"].append(0)

                if NEQ:
                    self.store_weights[idx] = weight_old
                else:
                    self.store_weights.append(weight_old)
                    self.store_gauge.append(copy.deepcopy(self.system._gaugefieldvec))

    def scan_cfgs(self, idx):
        curr_gauge = self.store_gauge[idx]
        weight = self.store_weights[idx]

        self.system.update_gauge_full_system(curr_gauge)
        new_weight = copy.deepcopy(self.system.weight)

        self.store_work[idx] += -new_weight + weight

        self.obsdict["work"].append(self.store_work[idx])
        self.measure_grad()

    def NEVMC_energy_gradient_mc(self, expW):
        # Compute the energy gradient from the MC results
        meas_grad_over_norm = self.obsdict["grad_norm"]
        expW_grad_over_norm = meas_grad_over_norm * expW

        # Gradient of the magnetic energy
        meas_mag_energy_op = self.obsdict["mag_energy_op"]
        prod_mag_energy_grad = meas_mag_energy_op * meas_grad_over_norm

        # Reweighted
        prod_mag_energy_grad = prod_mag_energy_grad * expW
        meas_mag_energy_op = meas_mag_energy_op * expW

        mag_energy_op_grad = prod_mag_energy_grad.mean() - meas_mag_energy_op.mean() * expW_grad_over_norm.mean()
        # Add the constants back into the expression of the magnetic energy
        mag_energy_grad = -2 * self.system.cfg.g_mag * mag_energy_op_grad

        # Gradient of the electric energy
        meas_el_energy_op = self.obsdict["el_energy_op"]
        meas_el_energy_op_grad = self.obsdict["el_energy_op_grad"]
        prod_el_energy_grad = meas_el_energy_op * meas_grad_over_norm

        # Reweighted
        prod_el_energy_grad = prod_el_energy_grad * expW
        meas_el_energy_op = meas_el_energy_op * expW
        meas_el_energy_op_grad = meas_el_energy_op_grad * expW

        el_energy_op_grad = (
            prod_el_energy_grad.mean()
            - meas_el_energy_op.mean() * expW_grad_over_norm.mean()
            + meas_el_energy_op_grad.mean()
        )
        # Add the constants back into the expression of the electric energy
        el_energy_grad = -2 * self.system.cfg.g_el * el_energy_op_grad

        # Gradient of the interaction energy
        meas_int_energy_op = self.obsdict["int_energy_op"]
        meas_int_energy_op_grad = self.obsdict["int_energy_op_grad"]
        prod_int_energy_grad = meas_int_energy_op * meas_grad_over_norm

        # Reweighted
        prod_int_energy_grad = prod_int_energy_grad * expW
        meas_int_energy_op = meas_int_energy_op * expW
        meas_int_energy_op_grad = meas_int_energy_op_grad * expW

        int_energy_op_grad = (
            prod_int_energy_grad.mean()
            - meas_int_energy_op.mean() * expW_grad_over_norm.mean()
            + meas_int_energy_op_grad.mean()
        )
        # Add the constants back into the expression of the interaction energy
        int_energy_grad = self.system.cfg.g_int * int_energy_op_grad

        # Gradient of the mass energy
        meas_mass_energy_op = self.obsdict["mass_energy_op"]
        meas_mass_energy_op_grad = self.obsdict["mass_energy_op_grad"]
        prod_mass_energy_grad = meas_mass_energy_op * meas_grad_over_norm

        # Reweighted
        prod_mass_energy_grad = prod_mass_energy_grad * expW
        meas_mass_energy_op = meas_mass_energy_op * expW
        meas_mass_energy_op_grad = meas_mass_energy_op_grad * expW

        mass_energy_op_grad = (
            prod_mass_energy_grad.mean()
            - meas_mass_energy_op.mean() * expW_grad_over_norm.mean()
            + meas_mass_energy_op_grad.mean()
        )
        # Add the constants back into the expression of the mass energy
        mass_energy_grad = self.system.cfg.g_mass * mass_energy_op_grad

        return mag_energy_grad + el_energy_grad + int_energy_grad + mass_energy_grad

    def measure_nograd(self):
        """Measure the corresponding observables in the dictionary"""
        polyakov_loop = self.system.cfg.lattice.generate_polyakov_loop((0, 0), lattice.Direction.X)

        self.obsdict["polyakov_00_x"].append(np.real(self.system.compute_path(polyakov_loop)))
        # self.obsdict["cov_ferm"].append(self.system.ferm_covmat_vec)
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
        self.obsdict["chem_energy"].append(self.system.chem_energy)
        self.obsdict["norm"].append(np.sum(self.system.calculate_lognormvec(all_factors=True)))
        if self.system.cfg.num_fermionic_layer > 0:  # We only compute occupations if there are fermionic layers
            self.obsdict["average_occupation"].append(self.system.average_occupation())
            self.obsdict["all_occupations"].append(self.system.occupations_before_ph)

        # Wilson loops
        sizes = self.system.cfg.lattice.generate_allowed_loop_dimensions()
        loops = self.system.cfg.lattice.generate_all_wilson_loops((0, 0), sizes)
        for k in range(len(sizes)):
            loop_name = f"wilson_loop_0-0_{sizes[k][0]}x{sizes[k][1]}"
            self.obsdict[loop_name].append(np.real(self.system.compute_path(loops[k])))

        # Meson strings
        max_string = 1 + max(self.system.cfg.lattice.nx, self.system.cfg.lattice.ny) // 2
        strings = [self.system.cfg.lattice.generate_L_string((0, 0), (k, k)) for k in range(1, max_string)]
        for k in range(1, max_string):
            string_name = f"square_string_0-0_{k}x{k}"
            self.obsdict[string_name].append(self.system.meson_string(strings[k - 1]))

        return

    def measure_grad(self):
        polyakov_loop = self.system.cfg.lattice.generate_polyakov_loop((0, 0), lattice.Direction.X)

        self.obsdict["polyakov_00_x"].append(np.real(self.system.compute_path(polyakov_loop)))
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
        self.obsdict["chem_energy"].append(self.system.chem_energy)
        self.obsdict["norm"].append(np.sum(self.system.calculate_lognormvec(all_factors=True)))
        if self.system.cfg.num_fermionic_layer > 0:  # We only compute occupations if there are fermionic layers
            self.obsdict["average_occupation"].append(self.system.average_occupation())
            self.obsdict["all_occupations"].append(self.system.occupations_before_ph)

        #############################
        self.obsdict["el_energy_op_grad"].append(self.system.el_energy_op_grad_vec)
        self.obsdict["int_energy_op_grad"].append(self.system.int_energy_op_grad_vec)
        self.obsdict["mass_energy_op_grad"].append(self.system.mass_energy_op_grad_vec)
        self.obsdict["chem_energy_op_grad"].append(self.system.chem_energy_op_grad_vec)
        self.obsdict["grad_norm"].append(self.system.grad_over_norm_vec)
        return

    ### end NEVMC ###
    #################################################################################################################

    def update_single_site(self):
        """Update for the MC simulation.
        This updates randomly chooses a single site and updates it.
        The update is local. The new gauge field value is drawn uniformly from the distribution of possible gauge
        fields (according to the gauge group).

        TODO: add gauge fixing here
        """
        # Pick a site to update
        link_ind = self.cfg.rng_state.choice(
            self.system.cfg.lattice.comp_tree, replace=False
        )  # we choose a link from those that are not fixed by gauge fixing
        # Uniformly pick a gauge value
        theta = self.system.cfg.gaugemgr.get_random_gauge_value(self.cfg.rng_state)
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
        The new gauge field value is drawn uniformly from the distribution of possible gauge fields
        (according to the gauge group).
        """
        # Pick a site to update
        lattice = self.system.cfg.lattice
        comp_tree = lattice.comp_tree  # non gauge fixed links
        for i in comp_tree:
            # Uniformly pick a gauge to replace
            theta = self.system.cfg.gaugemgr.get_random_gauge_value(self.cfg.rng_state)
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
        The new gauge field value is drawn uniformly from the distribution of possible gauge fields
        (according to the gauge group).
        """
        links_inds = self.cfg.rng_state.choice(
            self.system.cfg.lattice.comp_tree,
            self.cfg.update_size_per_step,
            replace=False,
        )

        for link_ind in links_inds:
            # Uniformly pick a gauge to replace
            theta = self.system.cfg.gaugemgr.get_random_gauge_value(self.cfg.rng_state)
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

    def save_full(self, fname_full: str):
        """Save the full MonteCarloEstimator

        Args:
            fname_full (str): Filename of the full MonteCarloEstimator
        """
        data_full = {
            "version": utils.get_git_hash(),
            "rng_state": self.cfg.rng_state.get_state(),
            "mc": self,
        }
        with gzip.open(fname_full, "wb") as outfile:
            pickle.dump(data_full, outfile)

    def save(self, output_dir="."):
        """Convenience function to combine saving the MonteCarloEstimator and the summary of the observables"""
        sys_cfg = self.system.cfg
        meas_steps = self.cfg.meas_steps
        warmup_steps = self.cfg.warmup_steps

        chem_str = ",".join([f"{val:.3f}" for val in sys_cfg.g_chem])
        couplings_str = (
            f"gel_{sys_cfg.g_el:.3f}_gmag_{sys_cfg.g_mag:.3f}_gint_{sys_cfg.g_int:.3f}"
            f"_gmass_{sys_cfg.g_mass:.3f}_gchem_{chem_str}"
        )

        fname_full = (
            f"data_nevmc_L_{sys_cfg.lattice.nx:02d}-{sys_cfg.lattice.ny:02d}_{couplings_str}"
            f"_nlayer_{sys_cfg.nlayer:02d}_wsteps_{warmup_steps:07d}_msteps_{meas_steps:07d}.pkl.gz"
        )
        fname_summary = (
            f"summary_nevmc_L_{sys_cfg.lattice.nx:02d}-{sys_cfg.lattice.ny:02d}_{couplings_str}"
            f"_nlayer_{sys_cfg.nlayer:02d}_wsteps_{warmup_steps:07d}_msteps_{meas_steps:07d}.pkl"
        )
        self.save_full(os.path.join(output_dir, fname_full))
        self.save_summary(os.path.join(output_dir, fname_summary))

    #### Output (plots or on the commandline) ####

    def print_stats(self):
        """Print a quick summary of the observables"""
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
            "nx": [],
            "ny": [],
            "paramvec": [],
            "ncopy": [],
            "nlayer": [],
            "g_el": [],
            "g_mag": [],
            "g_int": [],
            "g_mass": [],
            "mean": [],
            "warmup_steps": [],
            "meas_steps": [],
            "update_size": [],
            "seed": [],
            "err": [],
        }
        for key in self.obsdict.keys():
            dest["name"].append(key)
            dest["nx"].append(self.system.cfg.lattice.nx)
            dest["ny"].append(self.system.cfg.lattice.ny)
            dest["g_el"].append(self.system.cfg.g_el)
            dest["g_int"].append(self.system.cfg.g_int)
            dest["g_mag"].append(self.system.cfg.g_mag)
            dest["g_mass"].append(self.system.cfg.g_mass)
            dest["paramvec"].append(self.system.cfg.paramvec)
            dest["ncopy"].append(self.system.cfg.ncopy)
            dest["nlayer"].append(self.system.cfg.nlayer)
            dest["seed"].append(self.cfg.seed)
            dest["warmup_steps"].append(self.cfg.warmup_steps)
            dest["meas_steps"].append(self.cfg.meas_steps)
            dest["update_size"].append(self.cfg.update_size_per_step)
            dest["mean"].append(self.get_obs_mean(key))
            dest["err"].append(self.get_obs_mean_err(key))
        df = pd.DataFrame(dest)
        return df
