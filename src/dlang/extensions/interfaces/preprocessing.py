# Interface for preprocessing step plugins
from abc import ABC, abstractmethod

class PreprocessingInterface(ABC):
    @abstractmethod
    def apply(self, data):
        """
        Apply preprocessing to the data.
        """
        pass