from typing import Union
from abc import ABC, abstractmethod

import ray
import copy
import numpy as np
import pandas as pd

from ggpeps import utils
from ggpeps import logger
from ggpeps.mc import MonteCarloEstimator, run_mc
from ggpeps.exacteval import ExactEvaluator
from ggpeps.mc import MonteCarloEstimatorConfig
#from ggpeps.system import System2DBase, Config2DBase


class EvaluatorManager:
    """The EvaluatorManager allows the execution of a simulation with multiple cores.
    The parallelization is performed with ray; currently this is only supported for Monte Carlo (not Exact Contraction).

    If the simulation is distributed across N runners, each runner performs the full warm-up but only 1/N of the total measurement steps.
    
    This is the general interface for simulations that is used in the manager and minimizer.
    """

    def __init__(self, 
                 system_cls, #: System2DBase, 
                 system_cfg, #: Config2DBase,
                 mc_cfg: Union[MonteCarloEstimatorConfig, None],
                 nrunner: int):
        
        self.system_cfg = system_cfg
        self.system_cls = system_cls
        self.mc_cfg = mc_cfg
        self.nrunner = nrunner

        if self.mc_cfg is None:
            self.type = 'exact'
        else:
            self.type = 'mc'
    
    def simulate(self):
        if self.type == 'exact':
            # The exacteval implementation currently only supports a single runner
            system = self.system_cls(self.system_cfg)
            system.initialize()
            exact_eval = ExactEvaluator(system)
            exact_eval.evaluate()
            return exact_eval
        
        elif self.type == 'mc':
            """Start the simulation of the runners"""
            resultvec = []
            if self.nrunner > 0:
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
                system = self.system_cls(self.system_cfg)
                system.initialize()
                mc = MonteCarloEstimator(self.mc_cfg, system)
                mc.simulate()
                return mc
        
        raise ValueError(f"Unknown evaluator type {self.type}")
    
    def collect(self, resultvec):
        """Unify the results of the different Monte Carlo runners

        Args:
            resultvec (list): List of MonteCarloEstimators from the different runners

        Returns:
            MonteCarloEstimator: Monte Carlo estimator with information from all runners
        """
        system = self.system_cls(self.system_cfg)
        dest = MonteCarloEstimator(self.mc_cfg, system)
        if len(resultvec) > 1:
            dest.obsdict = utils.mergeDict(resultvec[0].obsdict, resultvec[1].obsdict)
            for mc_runner in resultvec[2:]:
                dest.obsdict = utils.mergeDict(dest.obsdict, mc_runner.obsdict)
        else:
            dest = resultvec[0]
        return dest