from typing import Optional

import os
import pickle
import logging

import numpy as np
from scipy.optimize import minimize

import ggpeps
from ggpeps.caching import Cache
from ggpeps.evaluator_manager import EvaluatorManager
from ggpeps.evaluator import Evaluator

logger = logging.getLogger(ggpeps.LOGGER_NAME)

####################### Minimizer #######################


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


class MinimizerConfig:

    def __init__(self):
        self.max_iter: int = 100
        self.tol: float = 1e-5  # convergence tol (e.g. stop when grad falls below tol)
        self.alpha: float = 1e-2
        self._method: str = "CG"

    @property
    def method(self) -> str:
        return self._method

    @method.setter
    def method(self, val: str):
        self._method = val.upper()


class Minimizer:
    grad_methods = ["CG", "BFGS", "L-BFGS-B", "TNC", "CUSTOM"]
    no_grad_methods = ["POWELL", "NELDER-MEAD"]
    supported_methods = grad_methods + no_grad_methods

    def __init__(self, cfg: MinimizerConfig, evaluator_manager: EvaluatorManager):
        self.cfg: MinimizerConfig = cfg
        # We use the polymorphism of python classes.
        # Below, we will have to be careful to only call valid functions
        self.evaluator_manager: EvaluatorManager = evaluator_manager
        self.last_paramvec: Optional[np.ndarray] = None
        self.last_result: Optional[Evaluator] = None
        self.min_result: Optional[MinimizerResult] = None

        # Cache for the energy values and gradients
        self.cache: Cache = ggpeps.global_vars["cache"]

    def minimize(self):
        if self.cfg.method == "CUSTOM":
            return self.minimize_custom()
        elif self.cfg.method in self.supported_methods:
            return self.minimize_scipy()
        else:
            logger.error(f"Unkown minimization method '{self.cfg.method}'. Aborting...")
            return None

    def minimize_custom(self):
        paramvec = self.evaluator_manager.system_cfg.paramvec

        for ind in range(self.cfg.max_iter):
            if self.last_paramvec is None or not np.allclose(
                self.last_paramvec, paramvec
            ):
                # We copy here to get a new set of variables. We will change paramvec below and do not want to change last_paramvec
                self.last_paramvec = np.copy(paramvec)
                result = self.evaluator_manager.simulate()

            energy = result.get_obs_mean("energy")
            grad_paramvec = result.get_obs_mean("energy_grad")

            max_grad_paramvec = np.max(np.abs(grad_paramvec))
            self.last_result = result

            # Update logs
            print_callback(ind, self)

            # Check if the maximum of the gradient is smaller than convergence tolerance
            if max_grad_paramvec < abs(self.cfg.tol):
                message = f"Reached convergence: max grad paramvec < {self.cfg.tol}"
                logger.info(message)
                self.min_result = MinimizerResult(
                    paramvec, self.cfg.method, energy, grad_paramvec, True, message
                )
                return self.min_result

            # Adapt the parametervec according to the gradient
            # TODO: Implement stochastic reconfiguration

            # We have to use the internal name of the paramvec if we write to it since it is a property and not just an array
            self.evaluator_manager.system_cfg.paramvec -= self.cfg.alpha * grad_paramvec

        message = "Reached maximum number of iterations without convergence."
        logger.warning(message)
        self.min_result = MinimizerResult(
            paramvec, self.cfg.method, energy, grad_paramvec, False, message
        )
        return self.min_result

    def minimize_scipy(self):

        # Energy wrapper
        def energy_wrapper(flattened_paramvec):
            # Check if value is stored in cache (e.g. from previous minimization)
            energy = self.cache.load_obs_from_local_cache(flattened_paramvec, "energy")
            if energy is not None:
                logger.debug(f"Found cached value for energy: {energy}")
                return energy

            if self.last_paramvec is None or not np.allclose(
                self.last_paramvec, flattened_paramvec
            ):
                # We only set the parametervec and start the simulation if the parametervec is new
                self.last_paramvec = flattened_paramvec
                self.evaluator_manager.system_cfg.paramvec = np.reshape(
                    flattened_paramvec,
                    self.evaluator_manager.system_cfg.param_shape(),
                )
                self.last_result = self.evaluator_manager.simulate()

            # Save to cache -
            #   it is important to save energy and gradients (even though the last_paramvec stores both)
            #   so that if the computation is interrupted (which loses the last_paramvec),
            #   we can still use the cached values
            energy = self.last_result.get_obs_mean("energy")
            self.cache.add_obs_to_cache(flattened_paramvec, "energy", energy)
            if self.evaluator_manager.cfg.compute_grads:
                parametergrad = self.last_result.get_obs_mean("energy_grad")
                self.cache.add_obs_to_cache(
                    flattened_paramvec, "energy_grad", parametergrad
                )
            logger.debug(f"Calculated energy: {energy}")

            return energy

        # Jacobian wrapper
        def gradient_wrapper(flattened_paramvec):
            """Wrapper for the gradient of the total energy

            Args:
                paramvec (np.ndarray): parameters, arranged as a 1D array

            Returns:
                gradients (np.ndarray): gradients of the total energy, arranged as a 1D array
            """

            # Check if value is stored in cache (e.g. from previous minimization)
            parametergrad = self.cache.load_obs_from_local_cache(
                flattened_paramvec, "energy_grad"
            )
            if parametergrad is not None:
                # logger.debug('Found cached value for energy_grad')
                return parametergrad.reshape((-1))

            if self.last_paramvec is None or not np.allclose(
                self.last_paramvec, flattened_paramvec
            ):
                # We only set the parametervec and start the simulation if the parametervec is new
                self.last_paramvec = flattened_paramvec
                # self.evaluator.mc_cfg.compute_grads = True # make sure to calculate derivatives
                self.evaluator_manager.system_cfg.paramvec = np.reshape(
                    flattened_paramvec,
                    self.evaluator_manager.system_cfg.param_shape(),
                )
                self.last_result = self.evaluator_manager.simulate()

            # Save to cache
            energy = self.last_result.get_obs_mean("energy")
            self.cache.add_obs_to_cache(flattened_paramvec, "energy", energy)
            parametergrad = self.last_result.get_obs_mean("energy_grad")
            self.cache.add_obs_to_cache(
                flattened_paramvec, "energy_grad", parametergrad
            )

            return parametergrad.reshape((-1))

        # Manage settings for different minimization algorithms
        options_dict = {}
        if self.cfg.method != "TNC":
            # TNC does not support a maximum number of iterations
            options_dict["maxiter"] = self.cfg.max_iter

        # Use the random initialization from the system.initialize as first guess.
        # We might want to change this later.
        flattened_paramvec = np.reshape(
            self.evaluator_manager.system_cfg.paramvec, (-1)
        )
        min_result = minimize(
            energy_wrapper,
            flattened_paramvec,
            method=self.cfg.method,
            jac=gradient_wrapper,
            tol=self.cfg.tol,
            callback=lambda x: print_callback(x, self),
            options=options_dict,
        )
        flattened_paramvec = min_result.x
        if self.cfg.method in self.no_grad_methods:
            # these methods do not use the gradient
            flattened_energygrad = None
            num_jac_evals = 0
        else:
            flattened_energygrad = min_result.jac
            num_jac_evals = min_result.njev
        energy = min_result.fun
        converged = min_result.success
        message = f"{min_result.message} Total iters: {min_result.nit}, function evals: {min_result.nfev}, jac evals: {num_jac_evals}"

        dest = MinimizerResult(
            flattened_paramvec,
            flattened_energygrad,
            self.cfg.method,
            energy,
            converged,
            message,
        )
        self.min_result = dest
        return dest

    def save(self, output_dir="."):
        if self.min_result is not None:
            sys_cfg = self.evaluator_manager.system_cfg

            couplings_str = f"gel_{sys_cfg.g_el}_gmag_{sys_cfg.g_mag}_gint_{sys_cfg.g_int}_gmass_{sys_cfg.g_mass}_gchem_{np.array2string(sys_cfg.g_chem, separator=',')}"

            fname_mc_summary = f"summary_min_L_{sys_cfg.lattice.nx:02d}-{sys_cfg.lattice.ny:02d}_{couplings_str}_ncopy_{sys_cfg.ncopy:02d}_nlayer_{sys_cfg.nlayer:02d}.pkl"
            fname_result_min = f"result_min_L_{sys_cfg.lattice.nx:02d}-{sys_cfg.lattice.ny:02d}_{couplings_str}_ncopy_{sys_cfg.ncopy:02d}_nlayer_{sys_cfg.nlayer:02d}.pkl"

            self.last_result.save_summary(os.path.join(output_dir, fname_mc_summary))
            with open(os.path.join(output_dir, fname_result_min), "wb") as outfile:
                pickle.dump(self.min_result, outfile)


def print_callback(x, minimizer):

    res = minimizer.last_result
    paramvec = minimizer.evaluator_manager.system_cfg.paramvec

    # If caching is on, the first time we call this function, the last_result will be None
    if res is None:
        logger.info("Callback message due to caching. The last_result is None.")
        return

    energy = res.get_obs_mean("energy")
    avg_occupation = res.get_obs_mean("average_occupation")
    if minimizer.evaluator_manager.cfg.compute_grads:
        grad_paramvec = res.get_obs_mean("energy_grad")
        max_grad_paramvec = np.max(np.abs(grad_paramvec))
    else:
        grad_paramvec = None
        max_grad_paramvec = np.nan

    mass_energy = res.get_obs_mean("mass_energy")
    int_energy = res.get_obs_mean("int_energy")
    el_energy = res.get_obs_mean("el_energy")
    mag_energy = res.get_obs_mean("mag_energy")
    chem_energy = res.get_obs_mean("chem_energy")
    plaquette = res.get_obs_mean("wilson_loop_0-0_1x1")
    occ = ", ".join([f"{val:.4f}" for val in avg_occupation])

    message = f"Energy: {energy:.9f}, Occupation: {occ}, Plaquette: {plaquette:.6f}, Max grad paramvec: {max_grad_paramvec:.6f}"
    if minimizer.cfg.method == "CUSTOM":
        # We only have access to the iteration number if we are handling the minimization (via the CUSTOM method)
        message = f"Iter: {x:03d}, {message}"
    if "mc" in minimizer.evaluator_manager.type:
        # Acceptance probability is only defined for MC
        acceptance_prob = res.get_obs_mean("acceptance_prob")
        message += f", acceptance prob: {acceptance_prob:.6f}"
    logger.info(message)

    logger.debug(
        f"el: {el_energy:.6f}, mag: {mag_energy:.6f}, mass: {mass_energy:.6f}, int: {int_energy:.6f}, chem: {chem_energy:.6f}"
    )
    logger.debug(f"Parametervec: {paramvec}")

    # If we're at the lowest energy seen so far, log the parameters
    # if current_iter == 0 or energy < lowest_energy:
    #    lowest_energy = energy
    #    #logger.info(f"New best energy. Parametervec: {paramvec}")

    # If python recieves a signal to stop computation gracefully, we catch it here.
    # There have been recent developments within scipy's handling of these callbacks.
    # See (eg): https://github.com/scipy/scipy/issues/9412
    #           https://github.com/scipy/scipy/issues/7306#issuecomment-301183706
    # If/when this is implemented for scipy minimize, the handling of this should only be done here (not in the CUSTOM methods above)
    # if STOP_AFTER_CURRENT_ITERATION:
    #    return True
    #    raise StopIteration
