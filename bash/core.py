from abc import ABC, abstractmethod
from typing import Dict, Any

import torch


# Abstract Components
class Components(ABC):
    def __init__(
            self,
            model_name: str,
            torch_dtype: torch.dtype = None,
    ):
        """
        Initialize the CogView4 model by loading its components and passing them to the parent class constructor.

        :param model_name: The path to the directory containing the model files.
        :param torch_dtype: The data type for PyTorch tensors.
        """
        self.model_name = model_name
        self.torch_dtype = torch_dtype

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """
        Load and return the custom components of the model.

        :return: A dictionary containing the model components.
        """
        pass
