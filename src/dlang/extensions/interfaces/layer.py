# Interface for custom layer plugins
# Plugins implementing this interface can define new layer types.

from abc import ABC, abstractmethod

class LayerInterface(ABC):
    @abstractmethod
    def get_layer_class(self):
        """
        Return a class that implements the layer's forward pass.
        The class should be a subclass of torch.nn.Module (for PyTorch backend)
        or the equivalent for other backends.
        """
        pass

    @abstractmethod
    def get_parameters(self):
        """
        Return a dictionary of parameter names and their initializations.
        """
        pass

    @abstractmethod
    def get_input_shape(self):
        """
        Return the expected input shape (optional, can be None for dynamic).
        """
        pass

    @abstractmethod
    def get_output_shape(self, input_shape):
        """
        Return the output shape given an input shape.
        """
        pass