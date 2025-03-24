from typing import Dict, Any

import torch
from diffusers import BitsAndBytesConfig, CogView4Transformer2DModel
from transformers import GlmModel

from bash import core


# CogView4 Components
class CogView4Components(core.Components):
    def __init__(self, model_name: str, torch_dtype: torch.dtype = None):
        super().__init__(model_name, torch_dtype)
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        self.component_map: Dict[str, Any] = self.load()

    def load(self) -> Dict[str, Any]:
        # TextEncoder
        text_encoder = GlmModel.from_pretrained(
            self.model_name,
            subfolder='text_encoder',
            quantization_config=self.quantization_config,
            torch_dtype=self.torch_dtype,
        )
        # Transformer
        transformer = CogView4Transformer2DModel.from_pretrained(
            self.model_name,
            subfolder="transformer",
            quantization_config=self.quantization_config,
            torch_dtype=self.torch_dtype,
        )
        # Return the components
        return {
            'text_encoder': text_encoder,
            'transformer': transformer,
        }
