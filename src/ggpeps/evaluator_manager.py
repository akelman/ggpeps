from typing import Union, Optional

import ray
import copy
import logging

import ggpeps
from ggpeps import utils
from ggpeps.evaluator import Evaluator
from ggpeps.system.system_base import System2DBase, Config2DBase
from ggpeps.exacteval import ExactEvaluator, ExactEvaluatorConfig
from ggpeps.mc import MonteCarloEvaluator, MonteCarloEvaluatorConfig
from ggpeps.nevmc import NEVMC_Evaluator, NEVMC_EvaluatorConfig

logger = logging.getLogger(ggpeps.LOGGER_NAME)


####################### Multiprocessing layer #######################
@ray.remote
def run_mc(
    runner_id: int,
    evaluator_class: type[Evaluator],
    evaluator_cfg: Union[MonteCarloEvaluatorConfig, NEVMC_EvaluatorConfig],
    system_cls: type[System2DBase],
    system_cfg: Config2DBase,
    logger_info: dict,
    eval_args: dict = {},
) -> Evaluator:
    """Worker for running part of a MC simulation.

    Args:
        runner_id (int): Runner ID
        evaluator_class: (type[Evaluator]): MonteCarloEvaluator or NEVMC_Evaluator class type
        mc_cfg (MonteCarloEvaluatorConfig): the evaluator config
        system_cls (type[System2DBase]): a system class type (must inherit from System2DBase)
        system_cfg (Config2DBase): the system config
        logger_info (dict): configs for the logger (logger needs to be set up in each worker)
        eval_args (dict): Arguments for the evaluator

    Returns:
        Evaluator after running the simulation.
    """

    # Setup logger for each worker
    logger_file = logger_info["filename"]
    level = logger_info["logger_level"]
    logger = logging.getLogger(ggpeps.LOGGER_NAME)
    utils.setup_logger(logger, logger_file, level, runner_msg=f"Runner {runner_id}-")

    system = system_cls(copy.deepcopy(system_cfg))
    system.initialize()
    mc = evaluator_class(evaluator_cfg, system)
    mc.evaluate(**eval_args)
    return mc


@ray.remote
def run_nevmc(
    runner_id: int,
    evaluator_class: Union[MonteCarloEvaluator, NEVMC_Evaluator],
    evaluator_cfg: Union[MonteCarloEvaluatorConfig, NEVMC_EvaluatorConfig],
    system_cls,
    system_cfg,
    store_gauge,
    store_weights,
    store_work,
    logger_info: dict,
    eval_args: dict = {},
):
    """Worker for running part of a NEVMC simulation.

    Returns:
        NEVMC_Evaluator
    """

    # Setup logger
    logger_file = logger_info["filename"]
    level = logger_info["logger_level"]
    logger = logging.getLogger(ggpeps.LOGGER_NAME)
    utils.setup_logger(logger, logger_file, level, runner_msg=f"Runner {runner_id}-")

    system = system_cls(copy.deepcopy(system_cfg))
    system.initialize()
    mc = evaluator_class(evaluator_cfg, system, store_gauge, store_weights, store_work)
    mc.evaluate(**eval_args)
    return mc


####################### Evaluator Manager #######################
class EvaluatorManager:
    """The EvaluatorManager is a wrapper around the different evaluators (ExactEvaluator and MonteCarloEvaluator).
    It allows the execution of a simulation with multiple cores.
    The parallelization is handled with ray; currently this is only supported for Monte Carlo (not Exact Contraction).

    If an MC simulation is distributed across N runners, each runner performs the full
    warm-up but only 1/N of the total measurement steps.

    This is the general interface for simulations that is used in the manager and minimizer.
    """

    def __init__(
        self,
        system_cls: type[System2DBase],
        system_cfg: Config2DBase,
        cfg: Union[
            MonteCarloEvaluatorConfig, ExactEvaluatorConfig, NEVMC_EvaluatorConfig
        ],
        nrunner: int,
    ):

        self.system_cls = system_cls
        self.system_cfg = system_cfg
        self.cfg = cfg
        self.nrunner = nrunner

        ### beg NEVMC ###
        self.store_gauge = []
        self.store_weights = []
        self.store_work = []
        ### end NEVMC ###

        if isinstance(self.cfg, ExactEvaluatorConfig):
            self.type = "exact"
        elif isinstance(self.cfg, MonteCarloEvaluatorConfig):
            self.type = "mc"
        elif isinstance(self.cfg, NEVMC_EvaluatorConfig):
            self.type = "nevmc"
        else:
            raise ValueError("Unrecognized type of evaluator config.")

        # Set the evaluator
        self.evaluator: Evaluator = self.reset_evaluator()

    def reset_evaluator(self) -> Evaluator:
        """Reset the evaluator to a new instance with the current configuration."""

        system = self.system_cls(self.system_cfg)
        system.initialize()

        if self.type == "exact":
            assert isinstance(self.cfg, ExactEvaluatorConfig)
            self.evaluator = ExactEvaluator(self.cfg, system)
        elif self.type == "mc":
            assert isinstance(self.cfg, MonteCarloEvaluatorConfig)
            self.evaluator = MonteCarloEvaluator(self.cfg, system)
        elif self.type == "nevmc":
            assert isinstance(self.cfg, NEVMC_EvaluatorConfig)
            self.evaluator = NEVMC_Evaluator(
                self.cfg, system, self.store_gauge, self.store_weights, self.store_work
            )
        else:
            raise ValueError(f"Unknown evaluator type {self.type}")
        return self.evaluator

    def get_evaluator_class(self) -> type[Evaluator]:
        """Get the evaluator class based on the type of evaluator."""

        if self.type == "exact":
            return ExactEvaluator
        elif self.type == "mc":
            return MonteCarloEvaluator
        elif self.type == "nevmc":
            return NEVMC_Evaluator
        else:
            raise ValueError(f"Unknown evaluator type {self.type}")

    def get_evaluator(self):
        """Get the evaluator instance.

        Returns:
            Evaluator: The current evaluator instance.
        """
        return self.evaluator

    def simulate(self, eval_args: dict = {}):
        """Simulate

        Args:
            eval_args (dict): Arguments for the evaluator (e.g. for NEVMC).
        """

        if self.type == "nevmc" and self.nrunner > 0:

            resultvec = []
            reduced_meas_steps = self.cfg.meas_steps // self.nrunner
            logger.info(
                f"Starting {self.nrunner} ray runners with {reduced_meas_steps} measurement steps each (total: {self.nrunner * reduced_meas_steps})."
            )

            for i in range(self.nrunner):
                cfg = copy.deepcopy(self.cfg)
                cfg.seed = self.cfg.seed + i
                cfg.meas_steps = reduced_meas_steps

                logger_info = {
                    "filename": ggpeps.logger_file,
                    "logger_level": ggpeps.global_vars["args"].level,
                }

                cpu_frac = (
                    1 / ggpeps.global_vars["args"].nrunner
                )  # multiplied by the number of available cpus?
                gpu_frac = 0.0
                if ggpeps.GPU_AVAILABLE:
                    gpu_frac = 1 / ggpeps.global_vars["args"].nrunner
                evaluator_class = self.get_evaluator_class()

                local_gauge = self.store_gauge[
                    i * reduced_meas_steps : (i + 1) * reduced_meas_steps
                ]
                local_weights = self.store_weights[
                    i * reduced_meas_steps : (i + 1) * reduced_meas_steps
                ]
                local_work = self.store_work[
                    i * reduced_meas_steps : (i + 1) * reduced_meas_steps
                ]

                run_mc_modified = run_nevmc.options(
                    num_gpus=gpu_frac
                )  # according to the ray documentation, we should also specify num_cpus
                resultvec.append(
                    run_mc_modified.remote(
                        i,
                        evaluator_class,
                        cfg,
                        self.system_cls,
                        self.system_cfg,
                        local_gauge,
                        local_weights,
                        local_work,
                        logger_info,
                        eval_args=eval_args,
                    )
                )

            resultvec = ray.get(resultvec)
            self.evaluator = self.collect_NEVMC(resultvec)
            result_df = self.evaluator.summary()

        elif "mc" in self.type and self.nrunner > 0:
            """Start the simulation of the runners.
            Currently only Monte Carlo is supported (the exacteval implementation currently only supports one runner),
            and multiple runners cannot be resumed from where they left off.
            """
            assert isinstance(self.cfg, MonteCarloEvaluatorConfig) or isinstance(
                self.cfg, NEVMC_EvaluatorConfig
            )

            resultvec = []
            # system_cfg_id = ray.put(self.system_cfg)
            reduced_meas_steps = self.cfg.meas_steps // self.nrunner
            logger.info(
                f"Starting {self.nrunner} ray runners with {reduced_meas_steps} measurement steps each (total: {self.nrunner * reduced_meas_steps})."
            )

            for i in range(self.nrunner):
                # Make a copy of the MC config, and change the seed for each runner
                cfg = copy.deepcopy(self.cfg)
                cfg.seed = self.cfg.seed + i
                cfg.meas_steps = reduced_meas_steps

                # Make a copy of the system config
                # This is necessary, because otherwise an error is raised when we try to modify the params
                # in enforce_parameter_conditions()
                # For some unclear reason, making a deep copy here does not prevent this, so instead a deep
                # copy is made inside run_mc()
                # sys_cfg = copy.deepcopy(self.system_cfg)

                # package logger info
                logger_info = {
                    "filename": ggpeps.logger_file,
                    "logger_level": ggpeps.global_vars["args"].level,
                }

                cpu_frac = (
                    1 / ggpeps.global_vars["args"].nrunner
                )  # multiplied by the number of available cpus?
                gpu_frac = 0.0
                if ggpeps.GPU_AVAILABLE:
                    gpu_frac = 1 / ggpeps.global_vars["args"].nrunner
                evaluator_class = self.get_evaluator_class()

                run_mc_modified = run_mc.options(
                    num_gpus=gpu_frac
                )  # according to the ray documentation, we should also specify num_cpus
                resultvec.append(
                    run_mc_modified.remote(
                        i,
                        evaluator_class,
                        cfg,
                        self.system_cls,
                        self.system_cfg,
                        logger_info,
                        eval_args=eval_args,
                    )
                )
            resultvec = ray.get(resultvec)
            self.evaluator = self.collect(resultvec)
            result_df = self.evaluator.summary()
        else:
            self.reset_evaluator()
            self.evaluator.evaluate(**eval_args)
            result_df = self.evaluator.summary()

            if self.type == "nevmc":
                self.store_gauge = self.evaluator.store_gauge
                self.store_weights = self.evaluator.store_weights
                self.store_work = self.evaluator.store_work

        return result_df

    def collect(self, resultvec):
        """Unify the results of multiple runners

        Args:
            resultvec (list): List of Estimators from the different runners

        Returns:
            Evaluator: evaluator with information from all runners
        """
        system = self.system_cls(self.system_cfg)
        dest = MonteCarloEvaluator(self.cfg, system)
        if len(resultvec) > 1:
            dest.obsdict = utils.mergeDict(resultvec[0].obsdict, resultvec[1].obsdict)
            for mc_runner in resultvec[2:]:
                dest.obsdict = utils.mergeDict(dest.obsdict, mc_runner.obsdict)
        else:
            dest = resultvec[0]
        return dest

    def collect_NEVMC(self, resultvec):
        """Unify the results of multiple runners

        Args:
            resultvec (list): List of Estimators from the different runners

        Returns:
            Estimator: estimator with information from all runners
        """
        system = self.system_cls(self.system_cfg)
        dest = NEVMC_Evaluator(self.cfg, system, [], [], [])
        if len(resultvec) > 1:
            dest.obsdict = utils.mergeDict(resultvec[0].obsdict, resultvec[1].obsdict)
            dest.store_gauge = resultvec[0].store_gauge + resultvec[1].store_gauge
            dest.store_weights = resultvec[0].store_weights + resultvec[1].store_weights
            dest.store_work = resultvec[0].store_work + resultvec[1].store_work

            for mc_runner in resultvec[2:]:
                dest.obsdict = utils.mergeDict(dest.obsdict, mc_runner.obsdict)
                dest.store_gauge += mc_runner.store_gauge
                dest.store_weights += mc_runner.store_weights
                dest.store_work += mc_runner.store_work

        else:
            dest = resultvec[0]
            dest.store_gauge = resultvec[0].store_gauge
            dest.store_weights = resultvec[0].store_weights
            dest.store_work = resultvec[0].store_work

        self.store_gauge = dest.store_gauge
        self.store_weights = dest.store_weights
        self.store_work = dest.store_work

        return dest
