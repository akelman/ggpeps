from abc import ABC, abstractmethod


# This is not yet used, but should be the parent class of ExactEvaluator and MonteCarloEvaluator
class Evaluator(ABC):
    def __init__(self, evaluator_cfg, system):
        self.system = system
        self.obsdict: dict = None
        self.cfg = evaluator_cfg
        self.type: str | None = None # exact or mc

    @abstractmethod
    def simulate(self) -> dict:
        """Simulate the system and return the results as a dictionary of observables

        Raises:
            NotImplementedError: _description_

        Returns:
            dict: Dictionary of observables
                  Each key-val pair is of the form (obs: List) where List is a list of values for the observable for the simulated gauge configurations
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")
    
    @abstractmethod
    def get_obs_mean(self, obs: str):
        """Get the mean value of an observable

        Args:
            obs (str): Name of the observable

        Returns:
            float: Mean value of the observable
        """
        raise NotImplementedError("This is an abstract method. Implement in child class please.")
    