import os
import logging
import itertools as it
from typing import Union
from collections.abc import Iterator

import numpy as np
import pandas as pd

import ggpeps
import ggpeps.lattice as lattice
from ggpeps.evaluator import Evaluator
from ggpeps.system.system_base import System2DBase

logger = logging.getLogger(ggpeps.LOGGER_NAME)


class ExactEvaluatorConfig:
    """ExactEvaluator Configuration

    This class manages the parameters of the exact simulation.
    It is more convenient than passing an extensive number of parameters to the constructor.
    """

    def __init__(self) -> None:
        self.compute_grads: bool = True


class ExactEvaluator(Evaluator):
    """An ExactEvaluator exactly evaluates the expectation value of an observable by
    iterating over all possible states of the gauge field."""

    evaluator_type: str = "exact"

    def __init__(self, evaluator_cfg: ExactEvaluatorConfig, system: System2DBase) -> None:
        super().__init__(evaluator_cfg, system)

    def compute_expval(self, obs: np.ndarray, normvec: np.ndarray) -> Union[float, np.ndarray]:
        """Compute the expectation value of an observable.

        Args:
            obs (np.ndarray): Measurement values of an observables for different gauge
                              field configurations. The last dimension is the gauge field configurations.
            normvec (np.ndarray): Values of the norm of <Psi(G)|Psi(G)> for different gauge field configurations.

        Returns:
            float or array: the expectation value of the observable
        """
        normalization = np.sum(normvec)
        # sum over the last dimension (the gauge field configurations)
        expval = np.sum(obs * normvec, axis=obs.ndim - 1)
        return expval / normalization

    def get_obs_mean(self, obs: str) -> Union[float, np.ndarray]:
        """Return expectation value of an observable.
        Use this function to match the interface of the MonteCarloEvaluator."""
        return self.obsdict[obs]

    def evaluate(self) -> None:
        """Main evaluation function of ExactEvaluator.
        This function computes the exact expectation values <Psi|O|Psi>/<Psi|Psi> for
        a range of observables defined in the function.

        Returns:
            dict: Dictionary of the results
        """
        if not self.obsdict:  # obsdict is empty
            # Build an iterable object with all field configurations for all the links
            configvec = self.generate_config_vec()

            polyakov_loop = self.system.cfg.lattice.generate_polyakov_loop((0, 0), lattice.Direction.X)

            # Wilson loop & meson string preliminaries
            sizes = self.system.cfg.lattice.generate_allowed_loop_dimensions()
            loops = self.system.cfg.lattice.generate_all_wilson_loops((0, 0), sizes)
            max_string = 1 + max(self.system.cfg.lattice.nx, self.system.cfg.lattice.ny) // 2
            strings = [self.system.cfg.lattice.generate_L_string((0, 0), (k, k)) for k in range(1, max_string)]

            data: dict = {
                "energy": [],
                "norm": [],
                "mag_energy": [],
                "el_energy": [],
                "mass_energy": [],
                "int_energy": [],
                "chem_energy": [],
                "mag_energy_op": [],
                "el_energy_op": [],
                "mass_energy_op": [],
                "int_energy_op": [],
                "el_energy_op_grad": [],
                "mass_energy_op_grad": [],
                "int_energy_op_grad": [],
                "chem_energy_op_grad": [],
                "grad_norm": [],
                "polyakov_00_x": [],
            }
            if self.system.cfg.num_fermionic_layer > 0:
                # Only compute occupations if there are fermionic layers
                data["all_occupations"] = []
                data["average_occupation"] = []

            # Wilson loops
            for k in range(len(sizes)):
                loop_name = f"wilson_loop_0-0_{sizes[k][0]}x{sizes[k][1]}"
                data[loop_name] = []
            # Meson strings - for now, we compute only "square" meson strings
            for k in range(1, max_string):
                data[f"square_string_0-0_{k}x{k}"] = []

            for ind, config in enumerate(configvec):
                self.system.update_gauge_full_system(config)
                # logger.debug(f"Configuration: {config}")

                data["energy"].append(self.system.energy)
                data["mag_energy"].append(self.system.mag_energy)
                data["el_energy"].append(self.system.el_energy)
                data["mass_energy"].append(self.system.mass_energy)
                data["int_energy"].append(self.system.int_energy)
                data["chem_energy"].append(self.system.chem_energy)
                data["mag_energy_op"].append(self.system.mag_energy_op)
                data["el_energy_op"].append(self.system.el_energy_op)
                data["mass_energy_op"].append(self.system.mass_energy_op)
                data["int_energy_op"].append(self.system.int_energy_op)

                data["norm"].append(self.system.calculate_lognorm(all_factors=True))
                data["polyakov_00_x"].append(np.real(self.system.compute_path(polyakov_loop)))

                # Wilson loops
                for k in range(len(sizes)):
                    loop_name = f"wilson_loop_0-0_{sizes[k][0]}x{sizes[k][1]}"
                    data[loop_name].append(np.real(self.system.compute_path(loops[k])))

                # Meson strings
                for k in range(1, max_string):
                    data[f"square_string_0-0_{k}x{k}"].append(self.system.meson_string(strings[k - 1]))

                # Occupations
                if self.system.cfg.num_fermionic_layer > 0:  # If there are fermionic layers
                    data["average_occupation"].append(self.system.average_occupation())
                    data["all_occupations"].append(self.system.all_occupations)

                if self.cfg.compute_grads:
                    data["el_energy_op_grad"].append(self.system.el_energy_op_grad_vec)
                    data["mass_energy_op_grad"].append(self.system.mass_energy_op_grad_vec)
                    data["int_energy_op_grad"].append(self.system.int_energy_op_grad_vec)
                    data["chem_energy_op_grad"].append(self.system.chem_energy_op_grad_vec)
                    data["grad_norm"].append(self.system.grad_over_norm_vec)

            # Convert all lists or jax arrays to numpy arrays
            data = {key: np.asarray(data[key]) for key in data}

            # We need to change from log values to regular values here
            normvec = np.exp(data["norm"])

            # Expectation values
            dest: dict[str, Union[float, np.ndarray]] = {}
            dest["energy"] = self.compute_expval(data["energy"], normvec)
            dest["mag_energy"] = self.compute_expval(data["mag_energy"], normvec)
            dest["el_energy"] = self.compute_expval(data["el_energy"], normvec)
            dest["mass_energy"] = self.compute_expval(data["mass_energy"], normvec)
            dest["int_energy"] = self.compute_expval(data["int_energy"], normvec)
            dest["chem_energy"] = self.compute_expval(data["chem_energy"], normvec)
            dest["mass_energy_op"] = self.compute_expval(data["mass_energy_op"], normvec)
            dest["polyakov_00_x"] = self.compute_expval(data["polyakov_00_x"], normvec)
            if self.system.cfg.num_fermionic_layer > 0:  # If there are fermionic layers
                dest["average_occupation"] = self.compute_expval(
                    np.transpose(data["average_occupation"], [1, 0]), normvec
                )

                avg_occ = np.asarray(dest["average_occupation"])[:, np.newaxis]

                dest["variance_occupation"] = self.compute_expval(
                    (np.transpose(data["average_occupation"], [1, 0]) - avg_occ) ** 2,
                    normvec,
                )
                dest["all_occupations"] = self.compute_expval(
                    np.transpose(data["all_occupations"], [1, 2, 0]), normvec
                )

            # Wilson loops
            for k in range(len(sizes)):
                loop_name = f"wilson_loop_0-0_{sizes[k][0]}x{sizes[k][1]}"
                dest[loop_name] = self.compute_expval(data[loop_name], normvec)

            # Meson strings
            for k in range(1, max_string):
                string_name = f"square_string_0-0_{k}x{k}"
                dest[string_name] = self.compute_expval(data[string_name], normvec)

                # Fredenhagen-Marcu (FM) parameter
                dest[f"FM_{k}x{k}"] = dest[string_name] / np.sqrt(np.abs(dest[f"wilson_loop_0-0_{k}x{k}"]))

            # The norm that we turn in the end is the actual norm, not the lognorm!
            dest["norm"] = np.sum(normvec)

            # Compute the gradients
            if self.cfg.compute_grads:
                # Transpose to enable broadcasting
                grad_norm_transposed = np.transpose(data["grad_norm"], [1, 2, 3, 0])

                dest["grad_norm"] = self.compute_expval(grad_norm_transposed, normvec)

                # Magnetic gradient
                prod_mag_op_norm = data["mag_energy_op"] * grad_norm_transposed
                expval_prod_mag = self.compute_expval(prod_mag_op_norm, normvec)
                prod_expval_mag = self.compute_expval(data["mag_energy_op"], normvec) * dest["grad_norm"]
                mag_op_grad = expval_prod_mag - prod_expval_mag
                mag_energy_grad = (
                    -2 * self.system.cfg.g_mag * mag_op_grad
                )  # the factor of two comes from the Hamiltonian
                dest["mag_energy_grad"] = mag_energy_grad

                # Electric gradient
                prod_el_op_norm = data["el_energy_op"] * grad_norm_transposed
                expval_prod_el = self.compute_expval(prod_el_op_norm, normvec)
                prod_expval_el = self.compute_expval(data["el_energy_op"], normvec) * dest["grad_norm"]
                el_op_grad = (
                    expval_prod_el
                    - prod_expval_el
                    + self.compute_expval(np.transpose(data["el_energy_op_grad"], [1, 2, 3, 0]), normvec)
                )
                el_energy_grad = -self.system.cfg.g_el * el_op_grad
                dest["el_energy_grad"] = el_energy_grad

                # Mass gradient
                prod_mass_op_norm = data["mass_energy_op"] * grad_norm_transposed
                expval_prod_mass = self.compute_expval(prod_mass_op_norm, normvec)
                prod_expval_mass = self.compute_expval(data["mass_energy_op"], normvec) * dest["grad_norm"]
                mass_energy_grad = (
                    expval_prod_mass
                    - prod_expval_mass
                    + self.compute_expval(np.transpose(data["mass_energy_op_grad"], [1, 2, 3, 0]), normvec)
                )
                mass_energy_grad *= self.system.cfg.g_mass
                dest["mass_energy_grad"] = mass_energy_grad

                # Interaction gradient
                prod_int_op_norm = data["int_energy_op"] * grad_norm_transposed
                expval_prod_int = self.compute_expval(prod_int_op_norm, normvec)
                prod_expval_int = self.compute_expval(data["int_energy_op"], normvec) * dest["grad_norm"]
                int_energy_grad = (
                    expval_prod_int
                    - prod_expval_int
                    + self.compute_expval(np.transpose(data["int_energy_op_grad"], [1, 2, 3, 0]), normvec)
                )
                int_energy_grad *= self.system.cfg.g_int
                dest["int_energy_grad"] = int_energy_grad

                # Chemical potential gradient
                prod_chem_op_norm = data["chem_energy"] * grad_norm_transposed
                expval_prod_chem = self.compute_expval(prod_chem_op_norm, normvec)
                prod_expval_chem = self.compute_expval(data["chem_energy"], normvec) * dest["grad_norm"]
                scaled_chem_grad = np.transpose(data["chem_energy_op_grad"], [1, 2, 3, 0])
                for lay in range(self.system.cfg.num_pg_layer, self.system.cfg.nlayer):
                    # the gradients must be scaled by the chemical potential
                    ind = lay - self.system.cfg.num_pg_layer
                    scaled_chem_grad[lay, :, :, :] *= self.system.cfg.g_chem[ind]
                chem_energy_grad = expval_prod_chem - prod_expval_chem + self.compute_expval(scaled_chem_grad, normvec)
                dest["chem_energy_grad"] = chem_energy_grad

                # Add for the full gradient, subject to conditions on parameterization
                total_grad = mag_energy_grad + el_energy_grad + mass_energy_grad + int_energy_grad + chem_energy_grad

                assert isinstance(total_grad, np.ndarray)
                self.system.cfg.enforce_parameter_conditions(total_grad)
                dest["energy_grad"] = total_grad

            # Save data
            self.obsdict = dest
        return

    def generate_config_vec(self) -> Iterator[list[np.ndarray]]:
        """Generate gauge field configurations for all links."""

        poss_gauges = self.system.cfg.gaugemgr.get_possible_gauge_values()
        nlinks = self.system.cfg.lattice.nlinks
        non_fixed_links_ind = self.system.cfg.lattice.comp_tree  # All possible indices for the non-fixed positions
        neutral_gauge = self.system.cfg.gaugemgr.get_neutral_gauge_value()

        # Generate all possible gauge field combinations
        combinations = it.product(poss_gauges, repeat=len(non_fixed_links_ind))

        for combo in combinations:
            # Initialize configvec as a list filled with the neutral gauge value
            configvec = [neutral_gauge] * nlinks
            for i, pos in enumerate(non_fixed_links_ind):
                configvec[pos] = combo[i]
            yield configvec

    def generate_config_vec_no_gf(self) -> Iterator:
        """Generate gauge field configurations for all links when there is no gauge fixing.
        The function above, generate_config_vec(), is fully general, but this one may be more
        efficient when there is no gauge fixing."""

        poss_gauges = self.system.cfg.gaugemgr.get_possible_gauge_values()
        nlinks = self.system.cfg.lattice.nlinks
        configvec = it.product(
            poss_gauges, repeat=nlinks
        )  # an iterable object with all possible field configurations for the entire lattice.
        return configvec

    def summary(self) -> pd.DataFrame:
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
        }
        for key in self.obsdict.keys():
            dest["name"].append(key)
            dest["nx"].append(self.system.cfg.lattice.nx)
            dest["ny"].append(self.system.cfg.lattice.ny)
            dest["g_el"].append(self.system.cfg.g_el)
            dest["g_mag"].append(self.system.cfg.g_mag)
            dest["g_int"].append(self.system.cfg.g_int)
            dest["g_mass"].append(self.system.cfg.g_mass)
            dest["g_chem"].append(self.system.cfg.g_chem)
            dest["paramvec"].append(self.system.cfg.paramvec)
            dest["ncopy"].append(self.system.cfg.ncopy)
            dest["nlayer"].append(self.system.cfg.nlayer)
            dest["mean"].append(self.obsdict[key])
        df = pd.DataFrame(dest)
        return df

    def save(self, output_dir: str = ".") -> None:
        """Convenience function to generate a filename and save the summary in one step"""
        sys_cfg = self.system.cfg

        chem_str = ",".join([f"{val:.3f}" for val in sys_cfg.g_chem])
        couplings_str = (
            f"gel_{sys_cfg.g_el:.3f}_gmag_{sys_cfg.g_mag:.3f}_gint_{sys_cfg.g_int:.3f}"
            f"_gmass_{sys_cfg.g_mass:.3f}_gchem_{chem_str}"
        )

        fname_summary = f"summary_exact_L_{sys_cfg.lattice.nx:02d}-{sys_cfg.lattice.ny:02d}_{couplings_str}.pkl"
        self.save_summary(os.path.join(output_dir, fname_summary))
