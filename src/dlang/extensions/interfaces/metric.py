# Interface for metric plugins
from abc import ABC, abstractmethod

class MetricInterface(ABC):
    @abstractmethod
    def compute(self, y_true, y_pred):
        """
        Compute the metric given true and predicted values.
        """
        pass