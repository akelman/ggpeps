import numpy as np
import scipy
import logging
import pickle
from scipy.optimize import minimize

class MinimizerResult:
    def __init__(self,parametervec,energygrad,method,value,converged):
        self.parametervec=parametervec
        self.energygrad=energygrad
        self.method=method
        self.value=value
        self.converged=converged

    def __str__(self):
        dest="==== Minimizer Result ====\n"
        dest+="converged: {}\n".format(self.converged)
        dest+="Value: {}\n".format(self.value)
        dest+="Method: {}\n".format(self.method)
        dest+="==========================\n"
        return dest

class Minimizer():
    supported_methods=["CG","BFGS","L-BFGS-B"]

    def __init__(self,mc):
        self.mc_mgr=mc
        self.method="CG"
        self.last_paramvec=None
        self.last_mcresult=None
        self.min_result=None

        # Parameters for custom minimization
        self.max_it=100
        self.min_grad=1e-5
        self.alpha=0.1

    def minimize(self):
        if self.method=="custom":
            return self.minimize_custom()
        elif self.method in self.supported_methods:
            return self.minimize_scipy()
        else:
            logging.error("Unkown minimization method '{}'. Aborting...".format(self.method))
            return None

    def minimize_custom(self):
        paramvec=self.mc_mgr.system_cfg.paramvec

        for ind in range(self.max_it):
            if self.last_paramvec is None or not np.allclose(self.last_paramvec,paramvec):
                # We copy here to get a new set of variables. We will paramvec below and do not want to change last_paramvec
                self.last_paramvec=np.copy(paramvec)
                mc_result=self.mc_mgr.simulate()

            energy = mc_result.get_obs_mean("energy")
            grad_paramvec = self.energy_gradient(mc_result)
            max_grad_paramvec = max(abs(grad_paramvec))
            acceptance_prob = mc_result.get_obs_mean("acceptance_prob")
            logging.info("It: {:03d}, Energy: {:.5f}, Max grad paramvec: {:.5f} acceptance prob: {:.5f} ".format(ind,energy,max_grad_paramvec,acceptance_prob))

            #Check if the maximum of the gradient is smaller than min_grad
            if max_grad_paramvec < abs(self.min_grad):
                self.min_result = MinimizerResult(paramvec,self.method,energy,grad_paramvec,True)
                return self.min_result

            #Adapt the parametervec according to the gradient
            # TODO: Implement stochastic reconfiguration
            paramvec-=self.alpha*grad_paramvec

        logging.warn("Reached maximum number of iterations without convergence")
        self.min_result = MinimizerResult(paramvec,self.method,energy,grad_paramvec,False)
        return self.min_result


    def minimize_scipy(self):
        def energy_wrapper(parametervec):
            #Energy wrapper
            if self.last_paramvec is None or not np.allclose(self.last_paramvec,parametervec):
                #We only set the parametervec and start the simulation if the parametervec is new
                self.last_paramvec=parametervec
                self.mc_mgr.system_cfg.parametervec=parametervec
                self.last_mcresult=self.mc_mgr.simulate()
            energy=self.last_mcresult.get_obs_mean("energy")
            return energy

        def gradient_wrapper(parametervec):
            #Jacobian wrapper
            if self.last_paramvec is None or not np.allclose(self.last_paramvec,parametervec):
                #We only set the parametervec and start the simulation if the parametervec is new
                self.last_paramvec=parametervec
                self.mc_mgr.system_cfg.parametervec=parametervec
                self.last_mcresult=self.mc_mgr.simulate()
            parametergrad=self.energy_gradient(self.last_mcresult)
            return parametergrad

        # Use the random initialization from the system.initialize as first guess.
        # We might want to change this later.
        paramvec=self.mc_mgr.system_cfg.paramvec
        min_result= minimize(energy_wrapper, paramvec,
                                         method=self.method,
                                         jac=gradient_wrapper,
                                         callback=lambda x: print_callback(x, self))
        parametervec = min_result.x
        energygrad = min_result.jac
        energy = min_result.fun
        converged = min_result.success

        dest = MinimizerResult(parametervec, energygrad, self.method, energy,
                               converged)
        self.min_result = dest
        return dest

    def energy_gradient(self,mc):
        # Compute the energy gradient from the MC results
        meas_grad_over_norm = mc.obsdict["grad_norm"]

        # Gradient of the magnetic energy
        meas_mag_energy_op = mc.obsdict["mag_energy_op"]
        prod_mag_energy_grad = meas_mag_energy_op * meas_grad_over_norm
        mag_energy_op_grad = prod_mag_energy_grad.mean() - meas_mag_energy_op.mean() * meas_grad_over_norm.mean()
        # Add the constants back into the expression of the magnetic energy
        mag_energy_grad = - mc.system.cfg.g_mag * mag_energy_op_grad

        # Gradient of the electric energy
        meas_el_energy_op = mc.obsdict["el_energy_op"]
        meas_el_energy_op_grad = mc.obsdict["el_energy_op_grad"]
        prod_el_energy_grad = meas_el_energy_op * meas_grad_over_norm
        el_energy_op_grad = prod_el_energy_grad.mean() - meas_el_energy_op.mean()*meas_grad_over_norm.mean() + meas_el_energy_op_grad.mean()
        # Add the constants back into the expression of the magnetic energy
        el_energy_grad = - mc.system.cfg.g_el * el_energy_op_grad

        return mag_energy_grad + el_energy_grad


    def save(self):
        if self.min_result is not None:
            #fname_mc_summary="summary_min_L_{:02d}_wsteps_{:07d}_msteps_{:07d}.pkl".format(self.mc_mgr.system_cfg.lattice.nx,self.mc_mgr.mc_cfg.warmup_steps,self.mc_mgr.mc_cfg.meas_steps)
            sys_cfg=self.mc_mgr.system_cfg
            fname_result_min = "result_min_L_{:02d}_gel_{:.4f}_gm_{:.4f}.pkl".format(
                sys_cfg.lattice.nx, sys_cfg.g_el, sys_cfg.g_gm)
            with open(fname_result_min,"wb") as outfile:
                pickle.dump(self.min_result,outfile)

def print_callback(x,minimizer):
    mc=minimizer.last_mcresult
    acceptance_prob=mc.get_obs_mean("acceptance_prob")
    energy=mc.get_obs_mean("energy")
    grad_parametervec=minimizer.energy_gradient(mc)
    max_grad_paramvec=max(abs(grad_parametervec))
    logging.info("Energy: {:.5f}, Max grad paramvec: {:.5f} acceptance prob: {:.5f} ".format(energy,max_grad_paramvec,acceptance_prob))
