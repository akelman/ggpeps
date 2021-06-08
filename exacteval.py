import numpy as np
import itertools as it
import lattice

class ExactEvaluator():
    def __init__(self, system) -> None:
        self.system = system

    def compute_expval(self, obs, norm):
        normalization=np.sum(norm)
        if len(obs.shape)>1:
            # We have to treat the gradients differently as they are multi-dimensional observables
            expval=np.sum(obs*norm,axis=1)
        else:
            expval=np.sum(obs*norm)
        return expval/normalization


    def evaluate(self):
        poss_gauges = self.system.gaugemgr.get_possible_gauge_values()
        nlinks = self.system.cfg.lattice.nlinks
        configvec = it.product(poss_gauges, repeat=nlinks)

        polyakov_loop = self.system.cfg.lattice.generate_polyakov_loop(
            (0, 0), lattice.Direction.X)
        wilson_loop = self.system.cfg.lattice.generate_wilson_loop((0, 0),
                                                                   (1, 1))

        data = {
            "energy": [],
            "norm": [],
            "mag_energy": [],
            "el_energy": [],
            "mag_energy_op": [],
            "el_energy_op": [],
            "grad_norm": [],
            "wilson_00_11": [],
            "polyakov_00_x": []
        }
        for config in configvec:
            self.system.update_gauge_full_system(config)
            data["energy"].append(self.system.energy)
            data["mag_energy"].append(self.system.mag_energy)
            data["el_energy"].append(self.system.el_energy)
            data["mag_energy_op"].append(self.system.mag_energy_op)
            data["el_energy_op"].append(self.system.el_energy_op)
            data["norm"].append(self.system.calculate_lognorm(all_factors=True))
            data["grad_norm"].append(self.system.compute_grad_norm())
            data["wilson_00_11"].append(np.real(self.system.compute_path(wilson_loop)))
            data["polyakov_00_x"].append(np.real(self.system.compute_path(polyakov_loop)))

        # Expectation values
        dest = {}
        # Convert all lists to arrays
        data = {key: np.asarray(data[key]) for key in data}

        # We need to change from log values to regular values here
        normvec = np.exp(data["norm"])

        dest["energy"] = self.compute_expval(data["energy"], normvec)
        dest["mag_energy"] = self.compute_expval(data["mag_energy"], normvec)
        dest["el_energy"] = self.compute_expval(data["el_energy"], normvec)
        dest["norm"] = np.sum(normvec)

        # Compute the gradients

        # Magnetic gradient
        prod_mag_op_norm = data["mag_energy_op"] * np.transpose(data["grad_norm"])
        expval_prod_mag = self.compute_expval(prod_mag_op_norm, normvec)
        prod_expval_mag = self.compute_expval(data["mag_energy_op"],normvec) * self.compute_expval(np.transpose(data["grad_norm"]),normvec)
        mag_op_grad = expval_prod_mag - prod_expval_mag
        mag_energy_grad = -self.system.cfg.g_mag * mag_op_grad
        dest["mag_energy_grad"] = mag_energy_grad

        #TODO: Compute Electric gradient
        el_energy_grad = np.zeros(3)
        dest["el_energy_grad"] = el_energy_grad
        dest["energy_grad"] = mag_energy_grad + el_energy_grad

        return dest