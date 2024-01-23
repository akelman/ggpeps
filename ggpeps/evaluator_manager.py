from typing import Union
from abc import ABC, abstractmethod

import ray
import copy
import numpy as np

from ggpeps import utils
from ggpeps import logger
from ggpeps.exacteval import ExactEvaluator
from ggpeps.mc import MonteCarloEvaluator, MonteCarloEvaluatorConfig, run_mc
from ggpeps.system import SystemType, SystemConfigType

class EvaluatorManager:
    """The EvaluatorManager is a wrapper around the different evaluators (ExactEvaluator and MonteCarloEvaluator).
    It allows the execution of a simulation with multiple cores.
    The parallelization is performed with ray; currently this is only supported for Monte Carlo (not Exact Contraction).

    If an MC simulation is distributed across N runners, each runner performs the full warm-up but only 1/N of the total measurement steps.
    
    This is the general interface for simulations that is used in the manager and minimizer.
    """

    def __init__(self, 
                 system_cls: SystemType,
                 system_cfg: SystemConfigType,
                 mc_cfg: Union[MonteCarloEvaluatorConfig, None],
                 nrunner: int):
        
        self.system_cls = system_cls
        self.system_cfg = system_cfg
        self.mc_cfg = mc_cfg
        self.nrunner = nrunner

        self.evaluator = None
        self.reset_evaluator_before_next_eval: bool = True

        if self.mc_cfg is None:
            self.type = 'exact'
        else:
            self.type = 'mc'

    def reset_evaluator(self):
        system = self.system_cls(self.system_cfg)
        system.initialize()
        if self.type == 'exact':
            self.evaluator = ExactEvaluator(None, system)
        elif self.type == 'mc':
            self.evaluator = MonteCarloEvaluator(self.mc_cfg, system)
        else:
            raise ValueError(f"Unknown evaluator type {self.type}")

    def simulate(self):
        if self.type == 'mc' and self.nrunner > 0: # The exacteval implementation currently only supports a single runner
            """Start the simulation of the runners.
            Currently only Monte Carlo is supported, and the runners cannot be resumed from where they left off."""
            resultvec = []
            #system_cfg_id = ray.put(self.system_cfg)
            reduced_meas_steps = self.mc_cfg.meas_steps // self.nrunner
            logger.info(f"Starting {self.nrunner} runners with {reduced_meas_steps} measurement steps each (total: {self.nrunner * reduced_meas_steps}).")
            for i in range(self.nrunner):
                # Make a copy of the MC config, and change the seed for each runner
                cfg = copy.deepcopy(self.mc_cfg)
                cfg.seed = self.mc_cfg.seed + i
                cfg.meas_steps = reduced_meas_steps

                # Make a copy of the system config 
                # This is necessary, because otherwise an error is raised when we try to modify the params
                # in enforce_parameter_conditions()
                # For some unclear reason, making a deep copy here does not prevent this, so instead a deep 
                # copy is made inside run_mc()
                #sys_cfg = copy.deepcopy(self.system_cfg)
                
                resultvec.append(run_mc.remote(i, cfg, self.system_cls, self.system_cfg))
            resultvec = ray.get(resultvec)
            return self.collect(resultvec)
        else:
            self.reset_evaluator()
            self.evaluator.evaluate()
            return self.evaluator
    
    def resume_simulation(self):
        if self.type == 'mc' and self.nrunner > 0: 
            raise NotImplementedError("Resuming MC simulation with multiple runners is not yet implemented.")
        else:
            self.evaluator.evaluate()
            return self.evaluator
    
    def collect(self, resultvec):
        """Unify the results of multiple runners

        Args:
            resultvec (list): List of Estimators from the different runners

        Returns:
            Estimator: estimator with information from all runners
        """
        system = self.system_cls(self.system_cfg)
        dest = MonteCarloEvaluator(self.mc_cfg, system)
        if len(resultvec) > 1:
            dest.obsdict = utils.mergeDict(resultvec[0].obsdict, resultvec[1].obsdict)
            for mc_runner in resultvec[2:]:
                dest.obsdict = utils.mergeDict(dest.obsdict, mc_runner.obsdict)
        else:
            dest = resultvec[0]
        return dest

