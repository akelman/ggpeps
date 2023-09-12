import os
import pickle
import logging

import numpy as np
from scipy.optimize import minimize


class MinimizerResult:
    def __init__(self, paramvec, energygrad, method, value, converged, message):
        self.paramvec = paramvec
        self.energygrad = energygrad
        self.method = method
        self.value = value
        self.converged = converged
        self.message = message

    def __str__(self):
        dest = "==== Minimizer Result ====\n"
        dest += f"converged: {self.converged}\n"
        dest += f"Value: {self.value}\n"
        dest += f"Method: {self.method}\n"
        dest += f"Parameters: {self.paramvec}\n"
        dest += f"Message: {self.message}\n"
        dest += "==========================\n"
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
    def method(self, val):
        self._method = val.upper()

class Minimizer():
    STOP_AFTER_CURRENT_ITERATION = False # this is a flag to catch interrupts to end minimization
    supported_methods = ["CG", "BFGS", "L-BFGS-B", "POWELL", "NELDER-MEAD", "TNC"]

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
        if self.cfg.method == "CUSTOM":
            return self.minimize_custom()
        elif self.cfg.method in self.supported_methods:
            return self.minimize_scipy()
        else:
            logging.error(f"Unkown minimization method '{self.cfg.method}'. Aborting...")
            return None

    def minimize_custom(self):
        paramvec = self.evaluator.system_cfg.paramvec

        for ind in range(self.cfg.max_iter):
            if self.last_paramvec is None or not np.allclose(self.last_paramvec,paramvec):
                # We copy here to get a new set of variables. We will change paramvec below and do not want to change last_paramvec
                self.last_paramvec = np.copy(paramvec)
                result = self.evaluator.simulate()

            if self.use_exact:
                energy = result.obsdict["energy"]
                grad_paramvec = result.obsdict["energy_grad"]
            else:
                energy = result.get_obs_mean("energy")
                grad_paramvec = self.energy_gradient_mc(result)
            
            max_grad_paramvec = np.max(np.abs(grad_paramvec))
            self.last_result = result

            # Update logs
            print_callback(ind, self)

            # Check if the maximum of the gradient is smaller than min_grad
            if max_grad_paramvec < abs(self.cfg.min_grad):
                message = f"Reached convergence: max grad paramvec < {self.cfg.min_grad}"
                logging.info(message)
                self.min_result = MinimizerResult(paramvec, self.cfg.method, energy, grad_paramvec, True, message)
                return self.min_result

            if self.STOP_AFTER_CURRENT_ITERATION:
                message = f"Recieved interrupt signal from user. Ending minimization."
                logging.info(message)
                self.min_result = MinimizerResult(paramvec, self.cfg.method, energy, grad_paramvec, True, message)
                return self.min_result

            #Adapt the parametervec according to the gradient
            # TODO: Implement stochastic reconfiguration

            # We have to use the internal name of the paramvec if we write to it since it is a property and not just an array
            self.evaluator.system_cfg.paramvec -= self.cfg.alpha * grad_paramvec

        message = "Reached maximum number of iterations without convergence."
        logging.warn(message)
        self.min_result = MinimizerResult(paramvec, self.cfg.method, energy, grad_paramvec, False, message)
        return self.min_result


    def minimize_scipy(self):
        
        # Energy wrapper
        def energy_wrapper(paramvec):
            if self.last_paramvec is None or not np.allclose(self.last_paramvec, paramvec):
                # We only set the parametervec and start the simulation if the parametervec is new
                self.last_paramvec = paramvec
                self.evaluator.system_cfg.paramvec = np.reshape(paramvec,(-1, self.evaluator.system_cfg._nparams))
                self.last_result = self.evaluator.simulate()
            
            if self.use_exact:
                energy = self.last_result.obsdict["energy"]
            else:
                energy = self.last_result.get_obs_mean("energy")
            return energy
        
        # Jacobian wrapper
        def gradient_wrapper(paramvec):
            if self.last_paramvec is None or not np.allclose(self.last_paramvec, paramvec):
                # We only set the parametervec and start the simulation if the parametervec is new
                self.last_paramvec = paramvec
                #self.evaluator.mc_cfg.minimizer_mode = True # make sure to calculate derivatives
                self.evaluator.system_cfg.paramvec = np.reshape(paramvec,(-1, self.evaluator.system_cfg._nparams))
                self.last_result = self.evaluator.simulate()
            
            if self.use_exact:
                parametergrad = self.last_result.obsdict["energy_grad"]
            else:
                parametergrad = self.energy_gradient_mc(self.last_result)
            return parametergrad.reshape((-1))

        # Use the random initialization from the system.initialize as first guess.
        # We might want to change this later.
        paramvec = np.reshape(self.evaluator.system_cfg.paramvec, (-1))
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
        message = f"message: {min_result.message} Total iters: {min_result.nit}, function evals: {min_result.nfev}, jac evals: {min_result.njev}"

        dest = MinimizerResult(paramvec, energygrad, self.cfg.method, energy, converged, message)
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

        # Gradient of the interaction energy
        meas_int_energy_op = mc.obsdict["int_energy_op"]
        meas_int_energy_op_grad = mc.obsdict["int_energy_op_grad"]
        prod_int_energy_grad = meas_int_energy_op * meas_grad_over_norm
        int_energy_op_grad = prod_int_energy_grad.mean() - meas_int_energy_op.mean()*meas_grad_over_norm.mean() + meas_int_energy_op_grad.mean()
        # Add the constants back into the expression of the interaction energy
        int_energy_grad = mc.system.cfg.g_int * int_energy_op_grad

        # Gradient of the mass energy
        meas_mass_energy_op = mc.obsdict["mass_energy_op"]
        meas_mass_energy_op_grad = mc.obsdict["mass_energy_op_grad"]
        prod_mass_energy_grad = meas_mass_energy_op * meas_grad_over_norm
        mass_energy_op_grad = prod_mass_energy_grad.mean() - meas_mass_energy_op.mean()*meas_grad_over_norm.mean() + meas_mass_energy_op_grad.mean()
        # Add the constants back into the expression of the mass energy
        mass_energy_grad = mc.system.cfg.g_mass * mass_energy_op_grad

        return mag_energy_grad + el_energy_grad + int_energy_grad + mass_energy_grad


    def save(self, output_dir = "."):
        if self.min_result is not None:
            sys_cfg = self.evaluator.system_cfg

            #FIXME: Adapt the filenames here
            fname_mc_summary = f"summary_min_L_{sys_cfg.lattice.nx:02d}-{sys_cfg.lattice.ny:02d}_gel_{sys_cfg.g_el:.4f}_gmag_{sys_cfg.g_mag:.4f}_gint_{sys_cfg.g_int:.4f}_gmass_{sys_cfg.g_mass:.4f}_ncopy_{sys_cfg.ncopy:02d}_nlayer_{sys_cfg.nlayer:02d}.pkl"
            fname_result_min = f"result_min_L_{sys_cfg.lattice.nx:02d}-{sys_cfg.lattice.ny:02d}_gel_{sys_cfg.g_el:.4f}_gmag_{sys_cfg.g_mag:.4f}_gint_{sys_cfg.g_int:.4f}_gmass_{sys_cfg.g_mass:.4f}_ncopy_{sys_cfg.ncopy:02d}_nlayer_{sys_cfg.nlayer:02d}.pkl"

            self.last_result.save_summary(os.path.join(output_dir, fname_mc_summary))
            with open(os.path.join(output_dir, fname_result_min), "wb") as outfile:
                pickle.dump(self.min_result, outfile)


def print_callback(x, minimizer):
    
    res = minimizer.last_result
    paramvec = minimizer.evaluator.system_cfg.paramvec
    
    if minimizer.use_exact:
        acceptance_prob = np.nan
        energy = res.obsdict["energy"]
        number_per_site = res.obsdict["number_per_site"]
        grad_paramvec = res.obsdict["energy_grad"]

        mass_energy = res.obsdict["mass_energy"]
        int_energy = res.obsdict["int_energy"]
        el_energy = res.obsdict["el_energy"]
        mag_energy = res.obsdict["mag_energy"]
    else:
        acceptance_prob = res.get_obs_mean("acceptance_prob")
        energy = res.get_obs_mean("energy")
        number_per_site = res.get_obs_mean("number_per_site")
        grad_paramvec = minimizer.energy_gradient_mc(res)

        mass_energy = res.get_obs_mean("mass_energy")
        int_energy = res.get_obs_mean("int_energy")
        el_energy = res.get_obs_mean("el_energy")
        mag_energy = res.get_obs_mean("mag_energy")
    max_grad_paramvec = np.max(np.abs(grad_paramvec))

    if minimizer.cfg.method == 'CUSTOM':
        # We only have access to the iteration number if we are handling the minimization (via the CUSTOM method)
        logging.info(f"Iter: {x:03d}, Energy: {energy:.9f}, Occupation: {number_per_site:.6f}, Max grad paramvec: {max_grad_paramvec:.6f}, acceptance prob: {acceptance_prob:.5f}")
    else: 
        logging.info(f"Energy: {energy:.9f}, Occupation: {number_per_site:.6f}, Max grad paramvec: {max_grad_paramvec:.6f}, acceptance prob: {acceptance_prob:.6f}")

    logging.debug(f"el: {el_energy:.6f}, mag: {mag_energy:.6f}, mass: {mass_energy:.6f}, int: {int_energy:.6f}")
    logging.debug(f"Parametervec: {paramvec}")

    # If we're at the lowest energy seen so far, log the parameters
    #if current_iter == 0 or energy < lowest_energy:
    #    lowest_energy = energy
    #    #logging.info(f"New best energy. Parametervec: {paramvec}")   

    # If python recieves a signal to stop computation gracefully, we catch it here.
    # There have been recent developments within scipy's handling of these callbacks.
    # See (eg): https://github.com/scipy/scipy/issues/9412 
    #           https://github.com/scipy/scipy/issues/7306#issuecomment-301183706 
    # If/when this is implemented for scipy minimize, the handling of this should only be done here (not in the CUSTOM methods above)
    #if STOP_AFTER_CURRENT_ITERATION:
    #    return True  
    #    raise StopIteration

