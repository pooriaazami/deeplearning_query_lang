# Interface for logger plugins
from abc import ABC, abstractmethod

class LoggerInterface(ABC):
    @abstractmethod
    def log(self, metrics, step):
        """
        Log metrics at a given step.
        """
        pass