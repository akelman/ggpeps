from abc import ABC, abstractmethod


class Evaluator(ABC):
    """Base class for the different evaluators (ExactEvaluator and MonteCarloEvaluator)."""

    def __init__(self, evaluator_cfg, system):
        self.system = system
        self.obsdict: dict = {}
        self.cfg = evaluator_cfg
        self.evaluator_type: str | None = None  # exact or mc

    @abstractmethod
    def evaluate(self):
        """Simulate the system and return the results as a dictionary of observables

        Raises:
            NotImplementedError: raised if the method is not implemented in the subclass.

        Returns:
            dict: Dictionary of observables
                  Each key-val pair is of the form (obs: List) where List is a list of
                  values for the observable for the simulated gauge configurations
        """
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )

    @abstractmethod
    def get_obs_mean(self, obs: str):
        """Get the mean value of an observable

        Args:
            obs (str): Name of the observable

        Returns:
            float or array: Mean value of the observable.
                            If the observable is an array, the returned value is also an array.
        """
        raise NotImplementedError(
            "This is an abstract method. Implement in child class please."
        )

    def save_summary(self, fname_summary: str) -> None:
        """Save the summary of the computation to a given filename

        Args:
            fname_summary (str): Output filename for the summary
        """
        df_summary = self.summary()
        df_summary.to_pickle(fname_summary)
