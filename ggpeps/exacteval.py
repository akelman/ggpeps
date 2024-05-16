import os
import itertools as it

import numpy as np
import pandas as pd

import ggpeps.lattice as lattice

class ExactEvaluatorManager:
    """Wrapper Class for the ExactEvaluator.
    This matches the structure of the MonteCarloManager and makes the two classes freely interchangable.
    """
    def __init__(self, system_cls, system_cfg):
        self.system_cfg = system_cfg
        self.system_cls = system_cls

    def simulate(self):
        """Start the simulation of the runners.
        The implementation currently only supports a single runner."""
        system = self.system_cls(self.system_cfg)
        system.initialize()
        exact_eval = ExactEvaluator(system)
        exact_eval.evaluate()
        return exact_eval


class ExactEvaluator():
    """An ExactEvaluator exactly evaluates the expectation value of an observable by iterating over all possible states of the gauge field.
    """
    def __init__(self, system) -> None:
        self.system = system
        self.obsdict = None

    def compute_expval(self, obs, normvec):
        """Compute the expectation value of an observable.

        Args:
            obs (np.array): Measurement values of an observables for different gauge field configurations
            normvec (np.array): Values of the norm of <Psi(G)|Psi(G)> for different gauge field configurations

        Returns:
            _type_: _description_
        """
        normalization = np.sum(normvec)
        if len(obs.shape) > 1:
            # We have to treat the gradients differently as they are multi-dimensional observables
            prod = obs * normvec
            expval = np.transpose(np.sum(prod, axis=2))
        else:
            expval = np.sum(obs*normvec)
        return expval/normalization


    def evaluate(self):
        """Main evaluation function of ExactEvaluator.
        This function computes the exact expectation values <Psi|O|Psi>/<Psi|Psi> for a range of observables defined in the function.

        Returns:
            dict: Dictionary of the results
        """
        if self.obsdict is None:
            poss_gauges = self.system.gaugemgr.get_possible_gauge_values()
            nlinks = self.system.cfg.lattice.nlinks
            configvec = it.product(poss_gauges, repeat=nlinks) # an iterable object with all possible field configurations for the entire lattice

            polyakov_loop = self.system.cfg.lattice.generate_polyakov_loop(
                (0, 0), lattice.Direction.X)
            wilson_loop = self.system.cfg.lattice.generate_wilson_loop((0, 0),
                                                                    (1, 1))

            # Wilson loops
            sizes = self.system.cfg.lattice.generate_allowed_loop_dimensions()
            loops = self.system.cfg.lattice.generate_all_wilson_loops((0,0), sizes)

            data = {
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
                "number_per_site": []
            }
            # Wilson loops
            for k in range(len(sizes)):
                loop_name = f"wilson_loop_0-0_{sizes[k][0]}x{sizes[k][1]}"
                data[loop_name] = []

            for config in configvec:
                self.system.update_gauge_full_system(config)
                #logging.debug(f"Configuration: {config}")
                
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
                
                data["el_energy_op_grad"].append(self.system.el_energy_op_grad_vec)
                data["mass_energy_op_grad"].append(self.system.mass_energy_op_grad_vec) 
                data["int_energy_op_grad"].append(self.system.int_energy_op_grad_vec) 
                data["chem_energy_op_grad"].append(self.system.chem_energy_op_grad_vec) 
                
                data["norm"].append(self.system.calculate_lognorm(all_factors=True))
                data["grad_norm"].append(self.system.compute_grad_norm_vec())
                data["polyakov_00_x"].append(np.real(self.system.compute_path(polyakov_loop)))

                data["number_per_site"].append(np.real(self.system.number_per_site))

                # Wilson loops
                for k in range(len(sizes)):
                    loop_name = f"wilson_loop_0-0_{sizes[k][0]}x{sizes[k][1]}"
                    data[loop_name].append(np.real(self.system.compute_path(loops[k])))

            # Expectation values
            dest = {}
            # Convert all lists to arrays
            data = {key: np.asarray(data[key]) for key in data}

            # We need to change from log values to regular values here
            normvec = np.exp(data["norm"])

            # Transpose to enable broadcasting
            grad_norm_transposed = np.transpose(data["grad_norm"],[2,1,0])

            dest["energy"] = self.compute_expval(data["energy"], normvec)
            dest["mag_energy"] = self.compute_expval(data["mag_energy"], normvec)
            dest["el_energy"] = self.compute_expval(data["el_energy"], normvec)
            dest["mass_energy"] = self.compute_expval(data["mass_energy"], normvec)
            dest["int_energy"] = self.compute_expval(data["int_energy"], normvec)
            dest["polyakov_00_x"] = self.compute_expval(data["polyakov_00_x"], normvec)
            dest["number_per_site"] = self.compute_expval(data["number_per_site"], normvec)
            dest["grad_norm"] = self.compute_expval(grad_norm_transposed, normvec)

            # Wilson loops
            for k in range(len(sizes)):
                loop_name = f"wilson_loop_0-0_{sizes[k][0]}x{sizes[k][1]}"
                dest[loop_name] = self.compute_expval(data[loop_name], normvec)

            #The norm that we turn in the end is the actual norm, not the lognorm!
            dest["norm"] = np.sum(normvec)

            # Compute the gradients

            # Magnetic gradient
            prod_mag_op_norm = data["mag_energy_op"] * grad_norm_transposed
            expval_prod_mag = self.compute_expval(prod_mag_op_norm, normvec)
            prod_expval_mag = self.compute_expval(data["mag_energy_op"], normvec) * dest["grad_norm"]
            mag_op_grad = expval_prod_mag - prod_expval_mag
            mag_energy_grad = -2 * self.system.cfg.g_mag * mag_op_grad # the factor of two comes from the Hamiltonian
            dest["mag_energy_grad"] = mag_energy_grad

            # Electric gradient
            prod_el_op_norm = data["el_energy_op"] * grad_norm_transposed
            expval_prod_el = self.compute_expval(prod_el_op_norm, normvec)
            prod_expval_el = self.compute_expval(data["el_energy_op"], normvec) * dest["grad_norm"]
            el_op_grad = expval_prod_el - prod_expval_el + self.compute_expval(np.transpose(data["el_energy_op_grad"],[2,1,0]), normvec)
            el_energy_grad = -2 * self.system.cfg.g_el * el_op_grad # the factor of two comes from the Hamiltonian
            dest["el_energy_grad"] = el_energy_grad

            # Mass gradient
            prod_mass_op_norm = data["mass_energy_op"] * grad_norm_transposed
            expval_prod_mass = self.compute_expval(prod_mass_op_norm, normvec)
            prod_expval_mass = self.compute_expval(data["mass_energy_op"], normvec) * dest["grad_norm"]
            mass_energy_grad = expval_prod_mass - prod_expval_mass + self.compute_expval(np.transpose(data["mass_energy_op_grad"], [2,1,0]), normvec)
            mass_energy_grad *= self.system.cfg.g_mass
            dest["mass_energy_grad"] = mass_energy_grad

            # Interaction gradient
            prod_int_op_norm = data["int_energy_op"] * grad_norm_transposed
            expval_prod_int = self.compute_expval(prod_int_op_norm, normvec)
            prod_expval_int = self.compute_expval(data["int_energy_op"], normvec) * dest["grad_norm"]
            int_energy_grad = expval_prod_int - prod_expval_int + self.compute_expval(np.transpose(data["int_energy_op_grad"], [2,1,0]), normvec)
            int_energy_grad *= self.system.cfg.g_int
            dest["int_energy_grad"] = int_energy_grad

            # Chemical potential gradient
            prod_chem_op_norm = data["chem_energy"] * grad_norm_transposed
            expval_prod_chem = self.compute_expval(prod_chem_op_norm, normvec)
            prod_expval_chem = self.compute_expval(data["chem_energy"], normvec) * dest["grad_norm"]
            chem_energy_grad = expval_prod_chem - prod_expval_chem + self.compute_expval(np.transpose(data["chem_energy_op_grad"], [2,1,0]), normvec)
            dest["chem_energy_grad"] = chem_energy_grad

            # Add for the full gradient, subject to conditions on parameterization
            total_grad = mag_energy_grad + el_energy_grad + mass_energy_grad + int_energy_grad + chem_energy_grad
            self.system.cfg.enforce_parameter_conditions(total_grad)
            dest["energy_grad"] = total_grad
            self.obsdict = dest

        return self.obsdict

    def summary(self):
        """Summarize the results of the exact contraction in a dataframe.

        Returns:
            pd.DataFrame: Result of the contraction
        """
        dest = {
            "name": [],
            "nx": [],
            "ny": [],
            "paramvec":[],
            "ncopy":[],
            "nlayer":[],
            "g_el": [],
            "g_mag": [],
            "g_int": [],
            "g_mass": [],
            "mean": []
        }
        for key in self.obsdict.keys():
            dest['name'].append(key)
            dest['nx'].append(self.system.cfg.lattice.nx)
            dest['ny'].append(self.system.cfg.lattice.ny)
            dest['g_el'].append(self.system.cfg.g_el)
            dest['g_mag'].append(self.system.cfg.g_mag)
            dest['g_int'].append(self.system.cfg.g_int)
            dest['g_mass'].append(self.system.cfg.g_mass)
            dest['paramvec'].append(self.system.cfg.paramvec)
            dest['ncopy'].append(self.system.cfg.ncopy)
            dest['nlayer'].append(self.system.cfg.nlayer)
            dest["mean"].append(self.obsdict[key])
        df = pd.DataFrame(dest)
        return df

    def save(self, output_dir="."):
        """Convenience function to generate a filename and save the summary in one step
        """
        syscfg = self.system.cfg
        tvec = syscfg.paramvec[:,0]
        yvec = syscfg.paramvec[:,1]
        zvec = syscfg.paramvec[:,2]
        tstr = "-".join([str(t) for t in tvec])
        ystr = "-".join([str(y) for y in yvec])
        zstr = "-".join([str(z) for z in zvec])

        fname_summary = f"summary_exact_L_{syscfg.lattice.nx:02d}-{syscfg.lattice.ny:02d}_gel_{syscfg.g_el:.3f}_gmag_{syscfg.g_mag:.3f}_gint_{syscfg.g_int:.3f}_t_{tstr}_y_{ystr}_z_{zstr}.pkl"
        self.save_summary(os.path.join(output_dir,fname_summary))

    def save_summary(self, fname_summary: str):
        """Save the summary of the computation to a given filename

        Args:
            fname_summary (str): Output filename for the summary
        """
        df_summary = self.summary()
        df_summary.to_pickle(fname_summary)