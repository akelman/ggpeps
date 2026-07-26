from ctypes import sizeof
from typing import Union, Optional

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
from ggpeps.system.system_base import System2DBase

logger = logging.getLogger(ggpeps.LOGGER_NAME)

#################### Monte Carlo Estimator Config ###################


class MonteCarloEvaluatorConfig:
    """Monte Carlo Configuration

    This class manages the parameters of the MC simulation.
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
        observables_mode: str = "all",
    ) -> None:

        self.warmup_steps = warmup_steps
        self.meas_steps = meas_steps
        self.binsize = binsize
        self.compute_grads = compute_grads
        self.update_size_per_step = update_size_per_step  # this can be set anywhere from 1 to nlinks (inclusive)
        # "all": measure every observable. "energy": only what the energy (and, with compute_grads,
        # the energy gradient) needs -- for minimization, where the other observables are unused.
        if observables_mode not in ("all", "energy"):
            raise ValueError(f"observables_mode must be 'all' or 'energy', got {observables_mode!r}")
        self.observables_mode = observables_mode

        # Logging frequency
        self.warmup_log_freq: int = warmup_log_freq
        self.run_log_freq: int = run_log_freq

        # Randomness
        self._seed: Optional[int] = None
        self._rng_state: Optional[np.random.RandomState] = None

    @property
    def seed(self) -> int:
        if self._seed is None:
            self._seed = np.random.randint(np.iinfo(np.int32).max)
            self._rng_state = np.random.RandomState(self._seed)
        return self._seed

    @seed.setter
    def seed(self, seedval: int) -> None:
        self._seed = seedval
        self._rng_state = np.random.RandomState(seedval)

    @property
    def rng_state(self) -> np.random.RandomState:
        if self._rng_state is None:
            self._seed = np.random.randint(np.iinfo(np.int32).max)
            self._rng_state = np.random.RandomState(self._seed)
        return self._rng_state

    @rng_state.setter
    def rng_state(self, state: np.random.RandomState) -> None:
        logger.error(
            "MonteCarloEstimatorConfig: Do not set the state directly. Use a seed instead."
            "Request to set the state directly was ignored."
        )

    def get_rng_state_internal_repr(self) -> dict:
        """Get the state of the RNG.

        Returns:
            dict
        """
        rng_state = self.rng_state  # this will initialize the RNG if it is not set
        return rng_state.get_state()

    def set_rng_state_internal_repr(self, state_repr: dict) -> None:
        """Set the state of the RNG.

        Args:
            state_repr (dict)
        """
        self.rng_state.set_state(state_repr)
        return

    def __str__(self) -> str:
        dest = f"Seed: {self.seed}\n"
        dest += f"Warmup steps: {self.warmup_steps}\n"
        dest += f"Measurement steps: {self.meas_steps}\n"
        dest += f"Update size: {self.update_size_per_step}\n"
        return dest


############### Monte Carlo runner ###############


class MonteCarloEvaluator(Evaluator):
    """Class to take care of the MC simulation on a single runner"""

    evaluator_type = "mc"

    def __init__(self, evaluator_cfg: MonteCarloEvaluatorConfig, system: System2DBase):
        self.obsdict: dict[str, Measurement]  # specify the type used in this class
        super().__init__(evaluator_cfg, system)

        self.step: int = 0
        self._initialize_measurement_geometry()
        self.init_measurements()

        # Choose how to update in each MC step
        # (This might change in the future if we implement different updates)
        self.update = self.update_N_sites

    def _initialize_measurement_geometry(self) -> None:
        """Precompute static loop/string definitions used by `measure`.

        These only depend on the lattice geometry and are unchanged during MC updates.
        """
        lat = self.system.cfg.lattice
        self._polyakov_loop = lat.generate_polyakov_loop((0, 0), lattice.Direction.X)

        self._wilson_sizes = lat.generate_allowed_loop_dimensions()
        self._wilson_loops = lat.generate_all_wilson_loops((0, 0), self._wilson_sizes)
        self._wilson_names = [f"wilson_loop_0-0_{size[0]}x{size[1]}" for size in self._wilson_sizes]

        max_string = 1 + max(lat.nx, lat.ny) // 2
        self._meson_strings = [lat.generate_L_string((0, 0), (k, k)) for k in range(1, max_string)]
        self._meson_names = [f"square_string_0-0_{k}x{k}" for k in range(1, max_string)]

    def init_measurements(self) -> None:
        """Add empty measurement vectors to the measurement dictionary.

        With ``observables_mode == "energy"`` only the observables consumed by the energy and by
        ``energy_gradient_mc`` are created (and later measured): the bare ``*_energy_op`` operators,
        ``energy``, ``chem_energy`` and the gradients. Everything else (loops, strings, lognorm,
        occupations, the derived per-term energies) is skipped -- it is not needed for minimization.
        """
        binsize = self.cfg.binsize
        energy_only = self.cfg.observables_mode == "energy"
        self.obsdict = {}  # reset

        self.obsdict["acceptance_prob"] = Measurement("Acceptance Probablity", binsize)
        self.obsdict["energy"] = Measurement("Energy", binsize)
        self.obsdict["chem_energy"] = Measurement("Chemical Energy", binsize)
        self.obsdict["mag_energy_op"] = Measurement("Magnetic Energy Operator (bare)", binsize)
        self.obsdict["el_energy_op"] = Measurement("Electric Energy Operator (bare)", binsize)
        self.obsdict["int_energy_op"] = Measurement("Interaction Energy Operator (bare)", binsize)
        self.obsdict["mass_energy_op"] = Measurement("Mass Energy Operator (bare)", binsize)

        if not energy_only:
            self.obsdict["mag_energy"] = Measurement("Magnetic Energy", binsize)
            self.obsdict["el_energy"] = Measurement("Electric Energy", binsize)
            self.obsdict["int_energy"] = Measurement("Interaction Energy", binsize)
            self.obsdict["mass_energy"] = Measurement("Mass Energy", binsize)
            self.obsdict["polyakov_00_x"] = Measurement("Polyakov (0,0) x", binsize)
            self.obsdict["lognorm"] = Measurement("LogNorm", binsize)
            if self.system.cfg.num_fermionic_layer > 0:
                self.obsdict["all_occupations"] = Measurement("All Occupations (after PH)", binsize)
                self.obsdict["average_occupation"] = Measurement("Average Occupation", binsize)
                self.obsdict["variance_occupation"] = Measurement("Variance Occupation", binsize)

            # Wilson loops (of various sizes)
            for loop_name in self._wilson_names:
                self.obsdict[loop_name] = Measurement(loop_name, binsize)

            # Meson strings
            for string_name in self._meson_names:
                self.obsdict[string_name] = Measurement(string_name, binsize)

        # Gradients
        if self.cfg.compute_grads:
            self.obsdict["el_energy_op_grad"] = Measurement("Electric Energy Operator Gradient", binsize)
            self.obsdict["int_energy_op_grad"] = Measurement("Interaction Energy Operator Gradient", binsize)
            self.obsdict["mass_energy_op_grad"] = Measurement("Mass Energy Operator Gradient", binsize)
            self.obsdict["chem_energy_op_grad"] = Measurement("Chemical Energy Operator Gradient", binsize)
            self.obsdict["grad_norm"] = Measurement("Gradient of Norm/Norm", binsize)
            self.obsdict["energy_grad"] = Measurement("Gradient of Total Energy", binsize)

    def measure(self) -> None:
        """Measure the corresponding observables in the dictionary"""
        # The bare operators, the energy and chem_energy are always measured: the energy is the
        # minimization objective and energy_gradient_mc needs the operators for its covariances.
        self.obsdict["mag_energy_op"].append(np.asarray(self.system.mag_energy_op))
        self.obsdict["el_energy_op"].append(np.asarray(self.system.el_energy_op))
        self.obsdict["int_energy_op"].append(np.asarray(self.system.int_energy_op))
        self.obsdict["mass_energy_op"].append(np.asarray(self.system.mass_energy_op))
        self.obsdict["energy"].append(float(self.system.energy))
        self.obsdict["chem_energy"].append(float(self.system.chem_energy))

        if self.cfg.observables_mode != "energy":
            self.obsdict["polyakov_00_x"].append(np.real(self.system.compute_path(self._polyakov_loop)))
            # self.obsdict["cov_ferm"].append(self.system.ferm_covmat_vec)

            # Most of these values could be calculated in a post-processing step
            self.obsdict["el_energy"].append(float(self.system.el_energy))
            self.obsdict["mag_energy"].append(float(self.system.mag_energy))
            self.obsdict["int_energy"].append(float(self.system.int_energy))
            self.obsdict["mass_energy"].append(float(self.system.mass_energy))
            self.obsdict["lognorm"].append(float(self.system.calculate_lognorm_inc(all_factors=True)))
            if self.system.cfg.num_fermionic_layer > 0:  # We only compute occupations for fermionic layers
                self.obsdict["all_occupations"].append(np.asarray(self.system.occupations_before_ph))
                self.obsdict["average_occupation"].append(np.asarray(self.system.average_occupation()))

            # Wilson loops
            # TODO: save sizes/loops/strings in a more efficient way, so that they are not recomputed each step
            for loop_name, loop_path in zip(self._wilson_names, self._wilson_loops):
                self.obsdict[loop_name].append(np.real(self.system.compute_path(loop_path)))

            # Meson strings
            for string_name, string_path in zip(self._meson_names, self._meson_strings):
                self.obsdict[string_name].append(np.asarray(self.system.meson_string(string_path)))

        if self.cfg.compute_grads:
            self.obsdict["el_energy_op_grad"].append(np.asarray(self.system.el_energy_op_grad_vec))
            self.obsdict["int_energy_op_grad"].append(np.asarray(self.system.int_energy_op_grad_vec))
            self.obsdict["mass_energy_op_grad"].append(np.asarray(self.system.mass_energy_op_grad_vec))
            self.obsdict["chem_energy_op_grad"].append(np.asarray(self.system.chem_energy_op_grad_vec))
            self.obsdict["grad_norm"].append(np.asarray(self.system.grad_over_norm_vec))

        return

    def energy_gradient_mc(self) -> np.ndarray:
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
        el_energy_grad = -self.system.cfg.gaugemgr.el_mult_factor * self.system.cfg.g_el * el_energy_op_grad

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

        # Gradient of the chemical potential
        meas_chem_energy = self.obsdict["chem_energy"]
        meas_chem_energy_op_grad = copy.deepcopy(self.obsdict["chem_energy_op_grad"])
        for lay in range(self.system.cfg.num_pg_layer, self.system.cfg.nlayer):
            # the gradients must be scaled by the chemical potential
            ind = lay - self.system.cfg.num_pg_layer
            meas_chem_energy_op_grad.datavec[lay] *= self.system.cfg.g_chem[ind]
        prod_chem_energy_grad = meas_chem_energy * meas_grad_over_norm
        chem_energy_grad = (
            prod_chem_energy_grad.mean()
            - meas_chem_energy.mean() * meas_grad_over_norm.mean()
            + meas_chem_energy_op_grad.mean()
        )

        # Total gradient
        grad = mag_energy_grad + el_energy_grad + int_energy_grad + mass_energy_grad + chem_energy_grad
        return grad

    def warmup(self) -> None:
        """Warm up phase without measurement"""

        logger.debug("Starting MC warmup")
        # Warmup does no measurements, so skip maintaining the measurement-only open-link ("mod") trackers
        # on every accepted step; recompute them once from scratch before the measurement phase begins.
        self.system.defer_mod_trackers = True
        while self.step < self.cfg.warmup_steps:
            if self.step % self.cfg.warmup_log_freq == 0:
                logger.debug(f"Warmup: {self.step}")
            self.update()
            self.step += 1
        self.system.defer_mod_trackers = False
        self.system.recompute_mod_trackers()
        logger.debug("Finished MC warmup")

    def run(self) -> None:
        """Meaurement phase"""

        logger.debug("Starting MC measurement")
        while self.step < self.cfg.warmup_steps + self.cfg.meas_steps:
            if self.step % self.cfg.run_log_freq == 0:
                acceptance_ratio = np.mean(self.obsdict["acceptance_prob"].datavec[-self.cfg.run_log_freq : :])
                logger.debug(
                    f"Run: {self.step}. Acceptance ratio of last {self.cfg.run_log_freq} steps is {acceptance_ratio}"
                )
            self.update()
            self.measure()
            self.step += 1

        # Update observables which depend on expectation values
        if self.system.cfg.num_fermionic_layer > 0 and self.cfg.observables_mode != "energy":
            # We only compute occupations if there are fermionic layers
            # and the required observables are being measured (not in "energy"-only mode)
            # TODO: this could be done much more efficiently with arrays
            # TODO: this variance observable has not been properly tested
            avg = self.obsdict["average_occupation"].mean()
            vals = [np.asarray((val - avg) ** 2) for val in self.obsdict["average_occupation"].datavec]
            self.obsdict["variance_occupation"].extend(vals)
        if self.cfg.compute_grads:
            # Update gradients which depend on expectation values
            # For interface reasons, we insert meas_steps copies of this gradient
            total_grad = self.energy_gradient_mc()
            self.obsdict["energy_grad"].extend([total_grad] * len(self.obsdict["energy"]))

        logger.debug("Finished MC measurement")
        return

    def update_N_sites(self) -> None:
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

    def evaluate(self) -> None:
        """Main routine to run a Monte Carlo simulation."""
        self.warmup()
        self.run()

    #### Data management functions ####

    def get_obs_mean(self, obsname: str) -> Union[None, float, np.ndarray]:
        """Returns the mean value of an observable

        Args:
            obsname (str): Name of observable

        Returns:
            float: Mean value of the observable
        """
        if obsname not in self.obsdict.keys():
            raise ValueError(f"Observable {obsname} not found in the measurement dictionary.")

        meas = self.obsdict[obsname]
        if meas is not None and len(meas) > 0:
            return meas.mean()
        return None

    def get_obs_mean_err(self, obsname: str) -> Union[None, float, np.ndarray]:
        """Returns the error on the mean of an observable

        Args:
            obsname (str): Name of observable

        Returns:
            float: Error on mean of observable
        """
        if obsname not in self.obsdict.keys():
            raise ValueError(f"Observable {obsname} not found in the measurement dictionary.")

        meas = self.obsdict[obsname]
        if obsname == "energy_grad":
            nlayer, unitcell_size, nparams = self.system.cfg.param_shape()
            dest = np.zeros((nlayer, unitcell_size, nparams))
            energy_obsvec = np.asarray(self.obsdict["energy"].get_timeseries())
            el_energy_grad = np.asarray(self.obsdict["el_energy_op_grad"].get_timeseries())
            g_el = self.system.cfg.g_el
            el_energy_grad = -2 * g_el * el_energy_grad

            mass_energy_grad = np.asarray(self.obsdict["mass_energy_op_grad"].get_timeseries())
            g_mass = self.system.cfg.g_mass
            mass_energy_grad = g_mass * mass_energy_grad
            int_energy_grad = np.asarray(self.obsdict["int_energy_op_grad"].get_timeseries())
            g_int = self.system.cfg.g_int
            int_energy_grad = g_int * int_energy_grad

            chem_energy_grad = np.copy(np.asarray(self.obsdict["chem_energy_op_grad"].get_timeseries()))
            for lay in range(self.system.cfg.num_pg_layer, self.system.cfg.nlayer):
                # the gradients must be scaled by the chemical potential
                ind = lay - self.system.cfg.num_pg_layer
                chem_energy_grad[lay] *= self.system.cfg.g_chem[ind]

            energy_grad_obsvec = el_energy_grad + mass_energy_grad + int_energy_grad + chem_energy_grad
            grad_norm_obsvec = np.asarray(self.obsdict["grad_norm"].get_timeseries())

            zeroed_params = self.system.cfg.get_zeroed_params()
            for layer in range(nlayer):
                for unit_cell in range(unitcell_size):
                    for grad_ind in range(nparams):
                        if (layer, unit_cell, grad_ind) in zeroed_params:
                            # If this is the a forced zeroed component, the error is 0.0
                            dest[layer, unit_cell, grad_ind] = 0.0
                        else:
                            energy_grad_component = energy_grad_obsvec[:, layer, unit_cell, grad_ind]
                            grad_norm_component = grad_norm_obsvec[:, layer, unit_cell, grad_ind]
                            dest[layer, unit_cell, grad_ind] = utils.compute_grad_err(
                                energy_obsvec, energy_grad_component, grad_norm_component
                            )
            return dest

        if meas is not None and len(meas) > 0:
            return meas.mean_err()
        return None

    def get_obs_std(self, obsname: str) -> Union[None, float, np.ndarray]:
        """Returns the standard deviation of an observable

        Args:
            obsname (str): Name of observable

        Returns:
            float: Standard deviation of an observable
        """
        if obsname not in self.obsdict.keys():
            raise ValueError(f"Observable {obsname} not found in the measurement dictionary.")

        meas = self.obsdict[obsname]
        if meas is not None and len(meas) > 0:
            return meas.std()
        return None

    def get_obs_var(self, obsname: str) -> Union[None, float, np.ndarray]:
        """Returns the variance of an observable

        Args:
            obsname (str): Name of observable

        Returns:
            float: Variance of the observable
        """
        if obsname not in self.obsdict.keys():
            raise ValueError(f"Observable {obsname} not found in the measurement dictionary.")

        meas = self.obsdict[obsname]
        if meas is not None and len(meas) > 0:
            return meas.var()
        return None

    def save_full(self, fname_full: str) -> None:
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

    def save(self, output_dir: str = ".") -> None:
        """Convenience function to combine saving the MonteCarloEstimator and the
        summary of the observables"""

        sys_cfg = self.system.cfg
        meas_steps = self.cfg.meas_steps
        warmup_steps = self.cfg.warmup_steps

        chem_str = ",".join([f"{val:.3f}" for val in sys_cfg.g_chem])
        couplings_str = (
            f"gel_{sys_cfg.g_el:.3f}_gmag_{sys_cfg.g_mag:.3f}_gint_{sys_cfg.g_int:.3f}"
            f"_gmass_{sys_cfg.g_mass:.3f}_gchem_{chem_str}"
        )

        fname_full = (
            f"data_mc_L_{sys_cfg.lattice.nx:02d}-{sys_cfg.lattice.ny:02d}_{couplings_str}"
            f"_nlayer_{sys_cfg.nlayer:02d}_wsteps_{warmup_steps:07d}_msteps_{meas_steps:07d}.pkl.gz"
        )
        fname_summary = (
            f"summary_mc_L_{sys_cfg.lattice.nx:02d}-{sys_cfg.lattice.ny:02d}_{couplings_str}"
            f"_nlayer_{sys_cfg.nlayer:02d}_wsteps_{warmup_steps:07d}_msteps_{meas_steps:07d}.pkl"
        )

        # Especially for large lattices, the system can have a very large memory footprint,
        # so we remove the large data which is not needed
        self.system.initialize()

        self.save_full(os.path.join(output_dir, fname_full))
        self.save_summary(os.path.join(output_dir, fname_summary))

    def summary(self) -> pd.DataFrame:
        """Create panda dataframe file that summarizes the evaluation."""
        dest: dict = {
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
            "g_chem": [],
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
            dest["g_chem"].append(self.system.cfg.g_chem)
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
