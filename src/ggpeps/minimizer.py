from typing import Optional

import os
import copy
import pickle
import logging

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import pandas as pd

import ggpeps
from ggpeps import utils
from ggpeps.caching import Cache
from ggpeps.evaluator_manager import EvaluatorManager

logger = logging.getLogger(ggpeps.LOGGER_NAME)

####################### Minimizer #######################


class MinimizerResult:
    def __init__(
        self,
        paramvec: np.ndarray,
        energygrad: Optional[np.ndarray],
        method: str,
        value: float,
        converged: bool,
        message: str,
    ) -> None:
        self.paramvec = paramvec
        self.energygrad = energygrad
        self.method = method
        self.value = value
        self.converged = converged
        self.message = message

    def __str__(self) -> str:
        dest = "==== Minimizer Result ====\n"
        dest += f"converged: {self.converged}\n"
        dest += f"Value: {self.value}\n"
        dest += f"Method: {self.method}\n"
        dest += f"Parameters: {self.paramvec}\n"
        dest += f"Message: {self.message}\n"
        dest += "==========================\n"
        return dest


class MinimizerConfig:

    def __init__(self, max_iter: int = 100, tol: float = 1e-5, alpha: float = 1e-2, method: str = "CG") -> None:
        """Initialize a MinimizerConfig

        Args:
            max_iter (int, optional): Maximum number of iterations. Defaults to 100.
            tol (float, optional): Convergence tolerance. Defaults to 1e-5.
            alpha (float, optional): Learning rate. Defaults to 1e-2.
            method (str, optional): Minimization algorithm. Defaults to "CG".
        """
        self.max_iter: int = max_iter
        self.tol: float = tol
        self.alpha: float = alpha
        self._method: str = method

    @property
    def method(self) -> str:
        return self._method

    @method.setter
    def method(self, val: str) -> None:
        self._method = val.upper()


class Minimizer:
    grad_methods: list[str] = ["CG", "BFGS", "L-BFGS-B", "TNC", "CUSTOM", "ADAM", "NEVMC"]
    no_grad_methods: list[str] = ["POWELL", "NELDER-MEAD"]
    supported_scipy_methods = grad_methods[:-3] + no_grad_methods

    def __init__(self, cfg: MinimizerConfig, evaluator_manager: EvaluatorManager) -> None:
        self.cfg: MinimizerConfig = cfg
        # We use the polymorphism of python classes.
        # Below, we will have to be careful to only call valid functions
        self.evaluator_manager: EvaluatorManager = evaluator_manager
        self.last_paramvec: Optional[np.ndarray] = None
        self.last_result: Optional[pd.DataFrame] = None  # track the most recent evaluation
        self.best_result: Optional[pd.DataFrame] = None  # track the best evaluation (lowest energy) seen so far
        self.min_result: Optional[MinimizerResult] = None

        # Cache for the energy values and gradients
        self.cache: Cache = ggpeps.global_vars["cache"]

    def minimize(self) -> Optional[MinimizerResult]:
        if self.cfg.method == "CUSTOM":
            return self.minimize_custom()
        if self.cfg.method == "ADAM":
            return self.minimize_adam()
        elif self.cfg.method == "NEVMC":
            return self.minimize_NEVMC()
        elif self.cfg.method in self.supported_scipy_methods:
            return self.minimize_scipy()
        else:
            logger.error(f"Unkown minimization method '{self.cfg.method}'. Aborting...")
            return None

    def minimize_custom(self) -> MinimizerResult:
        """
        Minimize the energy using a custom vanilla gradient descent method.

        Returns:
            MinimizerResult: The result of the minimization.
        """
        paramvec = self.evaluator_manager.system_cfg.paramvec

        num_evals = 0
        for ind in range(1, self.cfg.max_iter + 1):
            if self.last_paramvec is None or not np.allclose(self.last_paramvec, paramvec):
                # We copy here to get a new set of variables.
                # We will change paramvec below and do not want to change last_paramvec
                self.last_paramvec = np.copy(paramvec)

                # Monte Carlo part of the optimizer
                result = self.evaluator_manager.simulate()
                self.last_result = result
                num_evals += 1

            # Energy and gradient
            energy = utils.get_obs_mean_df(result, "energy")
            grad_paramvec = utils.get_obs_mean_df(result, "energy_grad")

            max_grad_paramvec = np.max(np.abs(grad_paramvec))

            # Save the best result
            if self.best_result is None or energy < utils.get_obs_mean_df(self.best_result, "energy"):
                self.best_result = utils.deepcopy_summary_df(result)

            # Update logs
            print_callback(ind, self)

            # Check if the maximum of the gradient is smaller than convergence tolerance
            if max_grad_paramvec < abs(self.cfg.tol):
                message = f"Reached convergence: max grad paramvec < {self.cfg.tol}. Total evals: {num_evals}."
                logger.info(message)

                self.min_result = MinimizerResult(
                    utils.get_obs_mean_df(self.best_result, "energy", column="paramvec"),
                    utils.get_obs_mean_df(self.best_result, "energy_grad"),
                    self.cfg.method,
                    utils.get_obs_mean_df(self.best_result, "energy"),
                    True,
                    message,
                )
                return self.min_result

            # Adapt the parametervec according to the gradient
            # TODO: Implement stochastic reconfiguration

            self.evaluator_manager.system_cfg.paramvec -= self.cfg.alpha * grad_paramvec

        # helps with static type checking
        assert self.best_result is not None

        message = f"Reached maximum number of iterations without convergence. Total evals: {num_evals}."
        logger.warning(message)

        self.min_result = MinimizerResult(
            utils.get_obs_mean_df(self.best_result, "energy", column="paramvec"),
            utils.get_obs_mean_df(self.best_result, "energy_grad"),
            self.cfg.method,
            utils.get_obs_mean_df(self.best_result, "energy"),
            False,
            message,
        )
        return self.min_result

    def minimize_adam(self) -> MinimizerResult:
        """
        Minimize the energy using a custom adam implementation.

        Returns:
            MinimizerResult: The result of the minimization.
        """
        # TODO: once the entire evalutor is jax jit compatible, we can use the optax adam implementation
        # which is also jit compatible.

        # Adam parameters -- TODO: make configurable
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        lr = self.cfg.alpha

        paramvec = self.evaluator_manager.system_cfg.paramvec

        m = np.zeros_like(paramvec)
        v = np.zeros_like(paramvec)

        num_evals = 0
        for ind in range(1, self.cfg.max_iter + 1):
            if self.last_paramvec is None or not np.allclose(self.last_paramvec, paramvec):
                # We copy here to get a new set of variables.
                # We will change paramvec below and do not want to change last_paramvec
                self.last_paramvec = np.copy(paramvec)

                # Monte Carlo part of the optimizer
                result = self.evaluator_manager.simulate()
                self.last_result = result
                num_evals += 1

            # Energy and gradient
            energy = utils.get_obs_mean_df(result, "energy")
            grad_paramvec = utils.get_obs_mean_df(result, "energy_grad")

            # Save the best result
            if self.best_result is None or energy < utils.get_obs_mean_df(self.best_result, "energy"):
                self.best_result = utils.deepcopy_summary_df(result)

            m = beta1 * m + (1 - beta1) * grad_paramvec
            v = beta2 * v + (1 - beta2) * (grad_paramvec * grad_paramvec)

            m_hat = m / (1 - beta1**ind)
            v_hat = v / (1 - beta2**ind)

            step = lr * m_hat / (np.sqrt(v_hat) + eps)
            # lr = lr / (1 + ind / 100)

            # Update logs
            print_callback(ind, self)

            # Check if the maximum of the planned step is smaller than convergence tolerance
            if np.linalg.norm(step) < abs(self.cfg.tol):
                message = f"Reached convergence: step < {self.cfg.tol}. Total evals: {num_evals}."
                logger.info(message)

                self.min_result = MinimizerResult(
                    utils.get_obs_mean_df(self.best_result, "energy", column="paramvec"),
                    utils.get_obs_mean_df(self.best_result, "energy_grad"),
                    self.cfg.method,
                    utils.get_obs_mean_df(self.best_result, "energy"),
                    True,
                    message,
                )
                return self.min_result

            self.evaluator_manager.system_cfg.paramvec -= step

        message = f"Reached maximum number of iterations without convergence. Total evals: {num_evals}."
        logger.warning(message)

        # helps with static type checking
        assert self.best_result is not None

        self.min_result = MinimizerResult(
            utils.get_obs_mean_df(self.best_result, "energy", column="paramvec"),
            utils.get_obs_mean_df(self.best_result, "energy_grad"),
            self.cfg.method,
            utils.get_obs_mean_df(self.best_result, "energy"),
            False,
            message,
        )
        return self.min_result

    def minimize_scipy(self) -> MinimizerResult:
        """
        Minimize the energy using scipy's optimization methods.

        Returns:
            MinimizerResult: The result of the minimization.
        """

        # Energy wrapper
        def energy_wrapper(flattened_paramvec: np.ndarray) -> float:
            """
            Wrapper for the energy
            Args:
                flattened_paramvec (np.ndarray): parameters, arranged as a 1D array
            Returns:
                energy (float): value of the total energy
            """
            # Check if value is stored in cache (e.g. from previous minimization)
            energy = self.cache.load_obs_from_local_cache(flattened_paramvec, "energy")
            if energy is not None:
                # logger.debug(f"Found cached value for energy: {energy}")
                return energy

            if self.last_paramvec is None or not np.allclose(self.last_paramvec, flattened_paramvec):
                # We only set the parametervec and do a simulation if the parametervec is new
                self.last_paramvec = flattened_paramvec
                self.evaluator_manager.system_cfg.paramvec = np.reshape(
                    flattened_paramvec,
                    self.evaluator_manager.system_cfg.param_shape(),
                )
                self.last_result = self.evaluator_manager.simulate()

            # helps with static type checking
            assert self.last_result is not None

            # Save to cache -
            #   it is important to save energy and gradients (even though the last_result stores both)
            #   so that if the computation is interrupted (which loses the last_paramvec),
            #   we can still use the cached values
            energy = utils.get_obs_mean_df(self.last_result, "energy")
            self.cache.add_obs_to_cache(flattened_paramvec, "energy", energy)
            if self.evaluator_manager.cfg.compute_grads:
                parametergrad = utils.get_obs_mean_df(self.last_result, "energy_grad")
                self.cache.add_obs_to_cache(flattened_paramvec, "energy_grad", parametergrad)
            # logger.debug(f"Calculated energy: {energy}")

            # Save the best result
            if self.best_result is None or energy < utils.get_obs_mean_df(self.best_result, "energy"):
                self.best_result = utils.deepcopy_summary_df(self.last_result)

            return energy

        # Jacobian wrapper
        def gradient_wrapper(flattened_paramvec: np.ndarray) -> np.ndarray:
            """Wrapper for the gradient of the total energy

            Args:
                paramvec (np.ndarray): parameters, arranged as a 1D array

            Returns:
                gradients (np.ndarray): gradients of the total energy, arranged as a 1D array
            """

            # Check if value is stored in cache (e.g. from previous minimization)
            parametergrad = self.cache.load_obs_from_local_cache(flattened_paramvec, "energy_grad")
            if parametergrad is not None:
                # logger.debug('Found cached value for energy_grad')
                return parametergrad.reshape((-1))

            if self.last_paramvec is None or not np.allclose(self.last_paramvec, flattened_paramvec):
                # We only set the parametervec and start the simulation if the parametervec is new
                self.last_paramvec = flattened_paramvec
                # self.evaluator.mc_cfg.compute_grads = True # make sure to calculate derivatives
                self.evaluator_manager.system_cfg.paramvec = np.reshape(
                    flattened_paramvec,
                    self.evaluator_manager.system_cfg.param_shape(),
                )
                self.last_result = self.evaluator_manager.simulate()

            # helps with static type checking
            assert self.last_result is not None

            # Save to cache
            energy = utils.get_obs_mean_df(self.last_result, "energy")
            self.cache.add_obs_to_cache(flattened_paramvec, "energy", energy)
            parametergrad = utils.get_obs_mean_df(self.last_result, "energy_grad")
            self.cache.add_obs_to_cache(flattened_paramvec, "energy_grad", parametergrad)

            # Save the best result
            if self.best_result is None or energy < utils.get_obs_mean_df(self.best_result, "energy"):
                self.best_result = utils.deepcopy_summary_df(self.last_result)

            return parametergrad.reshape((-1))

        # Manage settings for different minimization algorithms
        options_dict = {}
        if self.cfg.method != "TNC":
            # TNC does not support a maximum number of iterations
            options_dict["maxiter"] = self.cfg.max_iter

        # Use the random initialization from the system.initialize as first guess.
        # We might want to change this later.
        flattened_paramvec = np.reshape(self.evaluator_manager.system_cfg.paramvec, (-1))

        min_result = minimize(  # type: ignore
            energy_wrapper,
            flattened_paramvec,
            method=self.cfg.method,  # TODO: fix type hint - should be literal of supported method, not a string
            jac=gradient_wrapper if self.cfg.method in self.grad_methods else None,
            tol=self.cfg.tol,
            callback=lambda x: print_callback(x, self),
            options=options_dict,  # TODO: this also causes a type error, but unclear why
        )
        flattened_paramvec = min_result.x
        if self.cfg.method in self.no_grad_methods:
            # these methods do not use the gradient
            # flattened_energygrad = None
            num_jac_evals = 0
        else:
            # flattened_energygrad = min_result.jac
            num_jac_evals = min_result.njev
        # energy = min_result.fun
        converged = min_result.success
        message = f"{min_result.message} "
        message += f"Total iters: {min_result.nit}, function evals: {min_result.nfev}, jac evals: {num_jac_evals}"

        # helps with static type checking
        assert self.best_result is not None

        dest = MinimizerResult(
            utils.get_obs_mean_df(self.best_result, "energy", column="paramvec"),
            utils.get_obs_mean_df(self.best_result, "energy_grad"),
            self.cfg.method,
            utils.get_obs_mean_df(self.best_result, "energy"),
            converged,
            message,
        )
        self.min_result = dest
        return dest

    def save(self, output_dir: str = ".") -> None:
        """
        Save the minimization results to the output directory.

        Args:
            output_dir (str): Directory to save the results.

        Returns:
            None
        """
        if self.min_result is not None:
            sys_cfg = self.evaluator_manager.system_cfg

            chem_str = ",".join([f"{val:.3f}" for val in sys_cfg.g_chem])
            couplings_str = (
                f"gel_{sys_cfg.g_el:.3f}_gmag_{sys_cfg.g_mag:.3f}_gint_{sys_cfg.g_int:.3f}"
                f"_gmass_{sys_cfg.g_mass:.3f}_gchem_{chem_str}"
            )

            cfg_str = (
                f"{sys_cfg.lattice.nx:02d}-{sys_cfg.lattice.ny:02d}_{couplings_str}"
                f"_ncopy_{sys_cfg.ncopy:02d}_nlayer_{sys_cfg.nlayer:02d}"
            )
            fname_mc_summary = f"summary_min_L_{cfg_str}.pkl"
            fname_result_min = f"result_min_L_{cfg_str}.pkl"

            if self.best_result is not None:
                # result may be None if caching is on and the last result was not computed
                utils.save_summary_df(self.best_result, os.path.join(output_dir, fname_mc_summary))
            with open(os.path.join(output_dir, fname_result_min), "wb") as outfile:
                pickle.dump(self.min_result, outfile)

    ### beg NEVMC ###

    def minimize_NEVMC(self):

        paramvec = self.evaluator_manager.system_cfg.paramvec

        for ind in range(self.cfg.max_iter):

            if ind == 0:
                # First run at equilibrium
                self.evaluator_manager.cfg.first_warmup = True
                self.evaluator_manager.cfg.scanning = False
                result0_df = self.evaluator_manager.simulate()
                result0 = self.evaluator_manager.get_evaluator()

                # Standard calculation energy and grads
                energy = result0.get_obs_mean("energy")
                grad_paramvec = result0.get_obs_mean("energy_grad")

                max_grad_paramvec = np.max(np.abs(grad_paramvec))
                self.last_result = result0_df
                # Update logs
                print_callback(ind, self)

                # Standard minimization
                self.evaluator_manager.system_cfg.paramvec -= self.cfg.alpha * grad_paramvec

                # First reweighting: only scanning
                self.evaluator_manager.cfg.first_warmup = False
                self.evaluator_manager.cfg.scanning = True
                self.evaluator_manager.simulate()
                result1 = self.evaluator_manager.get_evaluator()

                # Compute DF
                Wmean = result1.obsdict["work"].mean()
                free_energy = -np.asarray(copy.deepcopy(result1.obsdict["work"].datavec))
                free_energy = -logsumexp(free_energy) + np.log(len(result1.obsdict["work"].datavec))

                # Compute exp(-Wd)
                expW = result1.obsdict["work"].__expDF__(free_energy)

                # Reweight energy
                EnergyExpW = result1.obsdict["energy"].__mul__(expW)
                energy = EnergyExpW.mean()
                # Compute reweighted gradients
                grad_paramvec = result1.NEVMC_energy_gradient_mc(expW)

                max_grad_paramvec = np.max(np.abs(grad_paramvec))
                self.last_result = [energy, max_grad_paramvec, Wmean, free_energy]

                #################################################################################
                # DEBUG
                # print("Paramvec: ", self.evaluator_manager.system_cfg.paramvec)
                # print("Second energy: ", energy)
                # print("Second grad_paramvec: ", grad_paramvec)
                # print("El energy: ", result1.get_obs_mean("el_energy"))
                # print("Mag energy: ", result1.get_obs_mean("mag_energy"))
                # print("Mass energy: ", result1.get_obs_mean("mass_energy"))
                # print("Int energy: ", result1.get_obs_mean("int_energy"))
                # print("Chem energy: ", result1.get_obs_mean("chem_energy"))
                # exit()
                #################################################################################

                # Update logs
                NEVMC_print_callback(ind, self.last_result)

                next_paramvec = copy.deepcopy(self.evaluator_manager.system_cfg.paramvec)
                next_paramvec -= self.cfg.alpha * grad_paramvec

            else:
                # if self.last_paramvec is None or not np.allclose(
                #    self.last_paramvec, next_paramvec
                # ):

                # We copy here to get a new set of variables.
                # We will change paramvec below and do not want to change last_paramvec
                self.last_paramvec = np.copy(paramvec)

                # Monte Carlo part of the optimizer
                self.evaluator_manager.cfg.first_warmup = False
                self.evaluator_manager.cfg.scanning = False
                self.evaluator_manager.simulate()
                result0 = self.evaluator_manager.get_evaluator()
                self.evaluator_manager.system_cfg.paramvec = copy.deepcopy(next_paramvec)
                self.evaluator_manager.cfg.first_warmup = False
                self.evaluator_manager.cfg.scanning = True
                self.evaluator_manager.simulate()
                result1 = self.evaluator_manager.get_evaluator()

                # Compute DF
                Wmean = copy.deepcopy(result1.obsdict["work"].mean())
                free_energy = -np.asarray(copy.deepcopy(result1.obsdict["work"].datavec))
                free_energy = -logsumexp(free_energy) + np.log(len(result1.obsdict["work"].datavec))

                # Compute exp(-Wd)
                expW = result1.obsdict["work"].__expDF__(free_energy)

                # Reweight energy
                EnergyExpW = result0.obsdict["energy"].__mul__(expW)
                energy = EnergyExpW.mean()

                # Compute reweighted gradients
                grad_paramvec = result1.NEVMC_energy_gradient_mc(expW)

                max_grad_paramvec = np.max(np.abs(grad_paramvec))
                self.last_result = [energy, max_grad_paramvec, Wmean, free_energy]

                # Update logs
                ### TODO modify this function, it is not printing the reweighted results
                NEVMC_print_callback(ind, self.last_result)

                # Check if the maximum of the gradient is smaller than tolerance
                if max_grad_paramvec < abs(self.cfg.tol):
                    message = f"Reached convergence: max grad paramvec < {self.cfg.tol}"
                    logger.info(message)

                    self.min_result = MinimizerResult(
                        paramvec,
                        grad_paramvec,
                        self.cfg.method,
                        energy,
                        True,
                        message,
                    )
                    return self.min_result

                next_paramvec = copy.deepcopy(self.evaluator_manager.system_cfg.paramvec)
                next_paramvec -= self.cfg.alpha * grad_paramvec

        message = "Reached maximum number of iterations without convergence."
        logger.warning(message)

        self.min_result = MinimizerResult(
            paramvec,
            grad_paramvec,
            self.cfg.method,
            energy,
            False,
            message,
        )
        return self.min_result


def NEVMC_print_callback(x, res):

    message = f"Energy: {res[0]:.9f}, Max grad paramvec: {res[1]:.6f}, Work: {res[2]:.6f}, "
    message += f"Free energy: {res[3]:.6f}, Wd: {res[2] - res[3]:.6f}"
    logger.info(message)


### end NEVMC ###


def print_callback(x: int, minimizer: Minimizer) -> None:
    """Print the current status of the minimization process.

    Args:
        x (int): Current iteration number.
        minimizer (Minimizer): The minimizer instance containing the results.

    Returns:
        None
    """
    res = minimizer.last_result

    # If caching is on, the first time we call this function, the last_result will be None
    if res is None:
        logger.info("Callback message due to caching. The last_result is None.")
        return

    energy = utils.get_obs_mean_df(res, "energy")
    if minimizer.evaluator_manager.cfg.compute_grads:
        grad_paramvec = utils.get_obs_mean_df(res, "energy_grad")
        max_grad_paramvec = np.max(np.abs(grad_paramvec))
    else:
        grad_paramvec = None
        max_grad_paramvec = np.nan

    # In observables_mode == "energy" only the objective observables exist (not the loops,
    # occupations or per-term energies); skip the parts of the report that need the rest.
    full_obs = getattr(minimizer.evaluator_manager.cfg, "observables_mode", "all") != "energy"

    if full_obs or abs(minimizer.evaluator_manager.system_cfg.g_mass) > 0:
        mass_energy_op = utils.get_obs_mean_df(res, "mass_energy_op")
    else:
        mass_energy_op = "N/A"
    plaquette = f"{utils.get_obs_mean_df(res, 'wilson_loop_0-0_1x1'):.6f}" if full_obs else "N/A"
    if full_obs and minimizer.evaluator_manager.system_cfg.num_fermionic_layer > 0:
        # If we have fermionic layers, we can compute the average occupation
        avg_occupation = utils.get_obs_mean_df(res, "average_occupation")
        avg_occ = ", ".join([f"{val:.4f}" for val in avg_occupation])
    else:
        avg_occ = "None"

    message = f"Energy: {energy:.9f}, Total Mass: {mass_energy_op}, Occupation: {avg_occ}, "
    message += f"Plaquette: {plaquette}, Max grad paramvec: {max_grad_paramvec:.6f}"
    if minimizer.cfg.method in ["CUSTOM", "ADAM"]:
        # We only have access to the iteration number if we are using our own minimizer implementation
        message = f"Iter: {x:03d}, {message}"
    if "mc" in minimizer.evaluator_manager.type:
        # Acceptance probability is only defined for MC
        acceptance_prob = utils.get_obs_mean_df(res, "acceptance_prob")
        message += f", acceptance prob: {acceptance_prob:.6f}"
    logger.info(message)

    if full_obs:
        if minimizer.evaluator_manager.system_cfg.num_fermionic_layer > 0:
            all_occupations = utils.get_obs_mean_df(res, "all_occupations")
            occ_str = ""
            for lay in range(len(all_occupations)):
                occ_str += ", ".join([f"{val:.10f}" for val in all_occupations[lay]])
                occ_str += " | "  # layer separator
        else:
            occ_str = "None"
        logger.debug(f"Occupations: {occ_str}")

        el_energy = utils.get_obs_mean_df(res, "el_energy")
        mag_energy = utils.get_obs_mean_df(res, "mag_energy")
        int_energy = utils.get_obs_mean_df(res, "int_energy")
        mass_energy = utils.get_obs_mean_df(res, "mass_energy")
        chem_energy = utils.get_obs_mean_df(res, "chem_energy")

        logger.debug(
            f"el: {el_energy:.6f}, mag: {mag_energy:.6f}, int: {int_energy:.6f}, mass: {mass_energy:.6f}, "
            f"chem: {chem_energy:.6f}"
        )
    # logger.debug(f"Paramvec: {utils.get_obs_mean_df(res, 'energy', column='paramvec')}")

    # If python recieves a signal to stop computation gracefully, we catch it here.
    # There have been recent developments within scipy's handling of these callbacks.
    # See (eg): https://github.com/scipy/scipy/issues/9412
    #           https://github.com/scipy/scipy/issues/7306#issuecomment-301183706
    # If/when this is implemented for scipy minimize, the handling of this should only be done here
    # (not in the CUSTOM methods above)
    # if STOP_AFTER_CURRENT_ITERATION:
    #    return True
    #    raise StopIteration
