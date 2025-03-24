from typing import Dict, Any

import torch
from diffusers import SD3Transformer2DModel, BitsAndBytesConfig
from transformers import T5EncoderModel, CLIPTextModelWithProjection

from bash import core


# StableDiffusion3.5LargeTurbo Components
class SD35LTComponents(core.Components):
    def __init__(
            self,
            model_name: str = "stabilityai/stable-diffusion-3.5-large-turbo",
            torch_dtype: torch.dtype = None,
    ):
        super().__init__(model_name, torch_dtype)
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        self.component_map: Dict[str, Any] = self.load()

    def load(self) -> Dict[str, Any]:
        # Transformer
        transformer = SD3Transformer2DModel.from_pretrained(
            self.model_name,
            subfolder="transformer",
            quantization_config=self.quantization_config,
            torch_dtype=self.torch_dtype
        )
        # TextEncoder
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            self.model_name,
            subfolder="text_encoder",
            quantization_config=self.quantization_config,
            torch_dtype=self.torch_dtype
        )
        # TextEncoder2
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.model_name,
            subfolder="text_encoder_2",
            quantization_config=self.quantization_config,
            torch_dtype=self.torch_dtype
        )
        # TextEncoder3
        text_encoder_3 = T5EncoderModel.from_pretrained(
            self.model_name,
            subfolder="text_encoder_3",
            quantization_config=self.quantization_config,
            torch_dtype=self.torch_dtype
        )
        # Return customized components
        return {
            'transformer': transformer,
            'text_encoder': text_encoder,
            'text_encoder_2': text_encoder_2,
            'text_encoder_3': text_encoder_3,
        }
