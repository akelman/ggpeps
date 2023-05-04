import numpy as np
import os
import logging
import pickle
from scipy.optimize import minimize

class MinimizerResult:
    def __init__(self, paramvec, energygrad, method, value, converged):
        self.paramvec = paramvec
        self.energygrad = energygrad
        self.method = method
        self.value = value
        self.converged = converged

    def __str__(self):
        dest="==== Minimizer Result ====\n"
        dest+="converged: {}\n".format(self.converged)
        dest+="Value: {}\n".format(self.value)
        dest+="Method: {}\n".format(self.method)
        dest+="Parameters: {}\n".format(self.paramvec)
        dest+="==========================\n"
        return dest

class MinimizerConfig():

    def __init__(self):
        self.max_iter = 100
        self.min_grad = 1e-5
        self.alpha = 1e-2
        self._method = "CG"

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self,val):
        self._method=val.upper()

class Minimizer():
    supported_methods=["CG","BFGS","L-BFGS-B"]

    def __init__(self, cfg, evaluator, use_exact=False):
        self.cfg = cfg
        self.use_exact = use_exact
        # We use the polymorphism of python classes.
        # Below, we will have to be careful to only call valid functions
        self.evaluator = evaluator
        self._method = "CG"
        self.last_paramvec = None
        self.last_result = None
        self.min_result = None


    def minimize(self):
        if self.cfg.method=="CUSTOM":
            return self.minimize_custom()
        elif self.cfg.method in self.supported_methods:
            return self.minimize_scipy()
        else:
            logging.error("Unkown minimization method '{}'. Aborting...".format(self.cfg.method))
            return None

    def minimize_custom(self):
        paramvec=self.evaluator.system_cfg.paramvec

        for ind in range(self.cfg.max_iter):
            if self.last_paramvec is None or not np.allclose(self.last_paramvec,paramvec):
                # We copy here to get a new set of variables. We will paramvec below and do not want to change last_paramvec
                self.last_paramvec=np.copy(paramvec)
                result=self.evaluator.simulate()

            if self.use_exact:
                energy = result.obsdict["energy"]
                grad_paramvec = result.obsdict["energy_grad"]
                acceptance_prob = np.nan
            else:
                energy = result.get_obs_mean("energy")
                grad_paramvec = self.energy_gradient_mc(result)
                acceptance_prob = result.get_obs_mean("acceptance_prob")
            max_grad_paramvec = np.max(np.abs(grad_paramvec))
            logging.debug("Parametervec: {}".format(paramvec))
            logging.info("Iter: {:03d}, Energy: {:.5f}, Max grad paramvec: {:.5f} acceptance prob: {:.5f} ".format(ind,energy,max_grad_paramvec,acceptance_prob))
            self.last_result = result

            #Check if the maximum of the gradient is smaller than min_grad
            if max_grad_paramvec < abs(self.cfg.min_grad):
                self.min_result = MinimizerResult(paramvec,self.cfg.method,energy,grad_paramvec,True)
                return self.min_result

            #Adapt the parametervec according to the gradient
            # TODO: Implement stochastic reconfiguration

            # We have to use the internal name of the paramvec if we write to it since it is a property and not just an array
            self.evaluator.system_cfg.paramvec -= self.cfg.alpha * grad_paramvec

        logging.warn("Reached maximum number of iterations without convergence")
        self.min_result = MinimizerResult(paramvec,self.cfg.method,energy,grad_paramvec,False)
        return self.min_result


    def minimize_scipy(self):
        def energy_wrapper(paramvec):
            #Energy wrapper
            #print("Entered energy wrapper")
            if self.last_paramvec is None or not np.allclose(self.last_paramvec, paramvec):
                # We only set the parametervec and start the simulation if the parametervec is new
                self.last_paramvec = paramvec
                self.evaluator.system_cfg.paramvec = np.reshape(paramvec,(-1,self.evaluator.system_cfg._nparams))
                self.last_result = self.evaluator.simulate()
            if self.use_exact:
                energy=self.last_result.obsdict["energy"]
            else:
                energy=self.last_result.get_obs_mean("energy")
            #print("Compute Energy; param: {}, energy: {}".format(parametervec, energy))
            return energy

        def gradient_wrapper(paramvec):
            #Jacobian wrapper
            #print("Entered gradient wrapper")
            if self.last_paramvec is None or not np.allclose(self.last_paramvec,paramvec):
                # We only set the parametervec and start the simulation if the parametervec is new
                self.last_paramvec = paramvec
                self.evaluator.system_cfg.paramvec = np.reshape(paramvec,(-1,self.evaluator.system_cfg._nparams))
                self.last_result = self.evaluator.simulate()
            if self.use_exact:
                parametergrad=self.last_result.obsdict["energy_grad"]
            else:
                parametergrad=self.energy_gradient_mc(self.last_result)
            #print("Compute Gradient:", parametervec)
            return parametergrad.reshape((-1))

        # Use the random initialization from the system.initialize as first guess.
        # We might want to change this later.
        paramvec = np.reshape(self.evaluator.system_cfg.paramvec,(-1))
        min_result = minimize(energy_wrapper,
                              paramvec,
                              method=self.cfg.method,
                              jac=gradient_wrapper,
                              callback=lambda x: print_callback(x, self),
                              options={"maxiter": self.cfg.max_iter})
        paramvec = min_result.x
        energygrad = min_result.jac
        energy = min_result.fun
        converged = min_result.success

        dest = MinimizerResult(paramvec, energygrad, self.cfg.method, energy,
                               converged)
        self.min_result = dest
        return dest

    def energy_gradient_mc(self,mc):
        # Compute the energy gradient from the MC results
        meas_grad_over_norm = mc.obsdict["grad_norm"]

        # Gradient of the magnetic energy
        meas_mag_energy_op = mc.obsdict["mag_energy_op"]
        prod_mag_energy_grad = meas_mag_energy_op * meas_grad_over_norm
        mag_energy_op_grad = prod_mag_energy_grad.mean() - meas_mag_energy_op.mean() * meas_grad_over_norm.mean()
        # Add the constants back into the expression of the magnetic energy
        mag_energy_grad = - 2 * mc.system.cfg.g_mag * mag_energy_op_grad

        # Gradient of the electric energy
        meas_el_energy_op = mc.obsdict["el_energy_op"]
        meas_el_energy_op_grad = mc.obsdict["el_energy_op_grad"]
        prod_el_energy_grad = meas_el_energy_op * meas_grad_over_norm
        el_energy_op_grad = prod_el_energy_grad.mean() - meas_el_energy_op.mean()*meas_grad_over_norm.mean() + meas_el_energy_op_grad.mean()
        # Add the constants back into the expression of the electric energy
        el_energy_grad = - 2 * mc.system.cfg.g_el * el_energy_op_grad

        return mag_energy_grad + el_energy_grad


    def save(self, output_dir = "."):
        if self.min_result is not None:
            sys_cfg=self.evaluator.system_cfg

            #FIXME: Adapt the filenames here
            fname_mc_summary = "summary_min_L_{:02d}-{:02d}_gel_{:.4f}_gmag_{:.4f}_gint_{:.4f}_ncopy_{:02d}_nlayer_{:02d}.pkl".format(
                sys_cfg.lattice.nx, sys_cfg.lattice.ny, sys_cfg.g_el, sys_cfg.g_mag, sys_cfg.g_int, sys_cfg.ncopy, sys_cfg.nlayer)
            fname_result_min = "result_min_L_{:02d}-{:02d}_gel_{:.4f}_gmag_{:.4f}_gint_{:.4f}_ncopy_{:02d}_nlayer_{:02d}.pkl".format(
                sys_cfg.lattice.nx, sys_cfg.lattice.ny, sys_cfg.g_el,sys_cfg.g_mag,
                sys_cfg.g_int, sys_cfg.ncopy, sys_cfg.nlayer)

            self.last_result.save_summary(os.path.join(output_dir,fname_mc_summary))
            with open(os.path.join(output_dir, fname_result_min),"wb") as outfile:
                pickle.dump(self.min_result,outfile)

def print_callback(x,minimizer):
    res=minimizer.last_result
    if minimizer.use_exact:
        acceptance_prob=np.nan
        energy=res.obsdict["energy"]
        grad_paramvec=res.obsdict["energy_grad"]
    else:
        acceptance_prob=res.get_obs_mean("acceptance_prob")
        energy=res.get_obs_mean("energy")
        grad_paramvec=minimizer.energy_gradient_mc(res)
    max_grad_paramvec=np.max(np.abs(grad_paramvec))
    logging.info("Energy: {:.8f}, Max grad paramvec: {:.8f}, acceptance prob: {:.5f} ".format(energy,max_grad_paramvec,acceptance_prob))
    
    mass_energy = res.obsdict["mass_energy"]
    int_energy = res.obsdict["int_energy"]
    el_energy = res.obsdict["el_energy"]
    mag_energy = res.obsdict["mag_energy"]
    logging.debug("el: {:.8f}, mag: {:.8f}, mass: {:.8f}, int: {:.8f}".format(el_energy, mag_energy, mass_energy, int_energy))