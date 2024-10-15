from typing import Union
from abc import ABC, abstractmethod

import ray
import copy
import logging
import numpy as np

import ggpeps
from ggpeps import utils
from ggpeps.exacteval import ExactEvaluator, ExactEvaluatorConfig
from ggpeps.mc import MonteCarloEvaluator, MonteCarloEvaluatorConfig, run_mc
from ggpeps.system import SystemType, SystemConfigType

logger = logging.getLogger(ggpeps.LOGGER_NAME)


class EvaluatorManager:
    """The EvaluatorManager is a wrapper around the different evaluators (ExactEvaluator and MonteCarloEvaluator).
    It allows the execution of a simulation with multiple cores.
    The parallelization is performed with ray; currently this is only supported for Monte Carlo (not Exact Contraction).

    If an MC simulation is distributed across N runners, each runner performs the full warm-up but only 1/N of the total measurement steps.

    This is the general interface for simulations that is used in the manager and minimizer.
    """

    def __init__(
        self,
        system_cls: SystemType,
        system_cfg: SystemConfigType,
        cfg: Union[MonteCarloEvaluatorConfig, ExactEvaluatorConfig],
        nrunner: int,
    ):

        self.system_cls = system_cls
        self.system_cfg = system_cfg
        self.cfg = cfg
        self.nrunner = nrunner

        self.evaluator = None
        self.simulation_in_progress: bool = (
            False  # Flag to indicate whether a simulation should be resumed
        )

        if isinstance(self.cfg, ExactEvaluatorConfig):
            self.type = "exact"
        elif isinstance(self.cfg, MonteCarloEvaluatorConfig):
            self.type = "mc"
        else:
            raise ValueError("Unrecognized type of evaluator config.")

    def reset_evaluator(self):
        system = self.system_cls(self.system_cfg)
        system.initialize()
        if self.type == "exact":
            self.evaluator = ExactEvaluator(self.cfg, system)
        elif self.type == "mc":
            self.evaluator = MonteCarloEvaluator(self.cfg, system)
        else:
            raise ValueError(f"Unknown evaluator type {self.type}")

    def simulate(self):
        if (
            self.type == "mc" and self.nrunner > 0
        ):  # The exacteval implementation currently only supports a single runner
            """Start the simulation of the runners.
            Currently only Monte Carlo is supported, and multiple runners cannot be resumed from where they left off.
            """
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

                run_mc_modified = run_mc.options(
                    num_gpus=gpu_frac
                )  # according to the ray documentation, we should also specify num_cpus
                resultvec.append(
                    run_mc_modified.remote(
                        i, cfg, self.system_cls, self.system_cfg, logger_info
                    )
                )

            resultvec = ray.get(resultvec)
            return self.collect(resultvec)
        else:
            if (
                self.type == "mc" and self.simulation_in_progress
            ):  # exacteval does not support resuming an evaluation
                self.evaluator.system.invalidate_gauge_update()
            else:
                self.reset_evaluator()
            self.simulation_in_progress = True
            self.evaluator.evaluate()
            self.simulation_in_progress = False
            return self.evaluator

    def collect(self, resultvec):
        """Unify the results of multiple runners

        Args:
            resultvec (list): List of Estimators from the different runners

        Returns:
            Estimator: estimator with information from all runners
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
