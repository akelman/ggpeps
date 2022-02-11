import numpy as np
import itertools as it
import lattice
import pandas as pd

class ExactEvaluatorManager:
    def __init__(self, system_cls, system_cfg):
        self.system_cfg = system_cfg
        self.system_cls = system_cls

    def simulate(self):
        """Start the simulation of the runners"""
        system = self.system_cls(self.system_cfg)
        system.initialize()
        exact_eval=ExactEvaluator(system)
        exact_eval.evaluate()
        return exact_eval


class ExactEvaluator():
    def __init__(self, system) -> None:
        self.system = system
        self.obsdict = None

    def compute_expval(self, obs, normvec):
        normalization=np.sum(normvec)
        if len(obs.shape)>1:
            # We have to treat the gradients differently as they are multi-dimensional observables
            prod = obs * normvec
            expval=np.transpose(np.sum(prod,axis=2))
        else:
            expval=np.sum(obs*normvec)
        return expval/normalization


    def evaluate(self):
        if self.obsdict is None:
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
                "el_energy_op_grad": [],
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
                data["el_energy_op_grad"].append(self.system.el_energy_op_grad_vec)
                data["norm"].append(self.system.calculate_lognorm(all_factors=True))
                data["grad_norm"].append(self.system.compute_grad_norm_vec())
                data["wilson_00_11"].append(np.real(self.system.compute_path(wilson_loop)))
                data["polyakov_00_x"].append(np.real(self.system.compute_path(polyakov_loop)))

            # Expectation values
            dest = {}
            # Convert all lists to arrays
            data = {key: np.asarray(data[key]) for key in data}

            # We need to change from log values to regular values here
            normvec = np.exp(data["norm"])

            # Transpose to enable broadcasting
            grad_norm_transposed=np.transpose(data["grad_norm"],[2,1,0])

            dest["energy"] = self.compute_expval(data["energy"], normvec)
            dest["mag_energy"] = self.compute_expval(data["mag_energy"], normvec)
            dest["el_energy"] = self.compute_expval(data["el_energy"], normvec)
            dest["wilson_00_11"] = self.compute_expval(data["wilson_00_11"], normvec)
            dest["polyakov_00_x"] = self.compute_expval(data["polyakov_00_x"], normvec)
            dest["grad_norm"] = self.compute_expval(grad_norm_transposed, normvec)

            #The norm that we turn in the end is the actual norm, not the lognorm!
            dest["norm"] = np.sum(normvec)

            # Compute the gradients

            # Magnetic gradient
            prod_mag_op_norm = data["mag_energy_op"] * grad_norm_transposed
            expval_prod_mag = self.compute_expval(prod_mag_op_norm, normvec)
            prod_expval_mag = self.compute_expval(data["mag_energy_op"],normvec) * dest["grad_norm"]
            mag_op_grad = expval_prod_mag - prod_expval_mag
            mag_energy_grad = -2*self.system.cfg.g2_mag * mag_op_grad
            dest["mag_energy_grad"] = mag_energy_grad

            # Electric gradient
            prod_el_op_norm = data["el_energy_op"] * grad_norm_transposed
            expval_prod_el = self.compute_expval(prod_el_op_norm, normvec)
            prod_expval_el = self.compute_expval(data["el_energy_op"],normvec) * dest["grad_norm"]
            el_op_grad = expval_prod_el - prod_expval_el + self.compute_expval(np.transpose(data["el_energy_op_grad"],[2,1,0]),normvec)
            el_energy_grad = - 2 *self.system.cfg.g2_el * el_op_grad
            dest["el_energy_grad"] = el_energy_grad

            # Add for the full electric gradient
            dest["energy_grad"] = mag_energy_grad + el_energy_grad
            self.obsdict=dest

        return self.obsdict

    def summary(self):
        dest = {
            "name": [],
            "nx": [],
            "ny": [],
            "paramvec":[],
            "ncopy":[],
            "nlayer":[],
            "g2_el": [],
            "g_gm": [],
            "g2_mag": [],
            "mean": []
        }
        for key in self.obsdict.keys():
            dest['name'].append(key)
            dest['nx'].append(self.system.cfg.lattice.nx)
            dest['ny'].append(self.system.cfg.lattice.ny)
            dest['g2_el'].append(self.system.cfg.g2_el)
            dest['g_gm'].append(self.system.cfg.g_gm)
            dest['g2_mag'].append(self.system.cfg.g2_mag)
            dest['paramvec'].append(self.system.cfg.paramvec)
            dest['ncopy'].append(self.system.cfg.ncopy)
            dest['nlayer'].append(self.system.cfg.nlayer)
            dest["mean"].append(self.obsdict[key])
        df = pd.DataFrame(dest)
        return df

    def save(self):
        syscfg = self.system.cfg
        tvec = syscfg.paramvec[:,0]
        yvec = syscfg.paramvec[:,1]
        zvec = syscfg.paramvec[:,2]
        tstr="-".join([str(t) for t in tvec])
        ystr="-".join([str(y) for y in yvec])
        zstr="-".join([str(z) for z in zvec])
        fname_summary = "summary_exact_L_{:02d}-{:02d}_gel_{:.3f}_gm_{:.3f}_gmag_{:.3f}_t_{}_y_{}_z_{}.pkl".format(
            syscfg.lattice.nx,syscfg.lattice.ny, syscfg.g2_el, syscfg.g_gm, syscfg.g2_mag, tstr, ystr, zstr)
        self.save_summary(fname_summary)

    def save_summary(self, fname_summary):
        df_summary = self.summary()
        df_summary.to_pickle(fname_summary)