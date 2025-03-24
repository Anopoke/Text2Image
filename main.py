import warnings

import torch
import yaml
from diffusers import StableDiffusion3Pipeline
from loguru import logger

from component.sd35lt import SD35LTComponents

warnings.filterwarnings("ignore")

TORCH_DTYPE = torch.bfloat16

logger.add("./logs/image_generate.log", rotation="2 MB", level="DEBUG")

if __name__ == '__main__':
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    logger.info(f"Loading model {config["model_path"]}...")
    components = SD35LTComponents(config["model_path"], torch_dtype=TORCH_DTYPE)
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config["model_path"],
        torch_dtype=TORCH_DTYPE,
        **components.component_map,
    )

    # Reduce video memory usage
    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    # Generate image
    for i, prompt in enumerate(config["prompts"]):
        logger.info(f"Processing Prompt: {prompt}")
        image_ = pipeline(
            prompt=prompt,
            num_inference_steps=config.get("num_inference_steps", 6),
            guidance_scale=config.get("guidance_scale", 0.7),
            width=config.get("width", 512),
            height=config.get("height", 512),
            num_images_per_prompt=config.get("num_images_per_prompt", 1),
            # max_sequence_length=512,
        )
        for j, image in enumerate(image_.images):
            if len(image_.images) > 1:
                image_path = f"./statics/images/sd35lt_{i}_{j}.png"
            else:
                image_path = f"./statics/images/sd35lt_{i}.png"
            image.save(image_path)
            logger.info(f"Image saved to {image_path}")
