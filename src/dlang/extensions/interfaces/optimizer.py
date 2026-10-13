# Interface for optimizer plugins
from abc import ABC, abstractmethod

class OptimizerInterface(ABC):
    @abstractmethod
    def get_optimizer(self, parameters):
        """
        Return an optimizer instance for the given parameters.
        """
        pass