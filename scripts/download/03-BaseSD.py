#!/usr/bin/env python3
from diffusers.pipelines import StableDiffusionPipeline

from animatediff import get_dir

out_dir = get_dir("data/models/huggingface/stable-diffusion-v1-5")

pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    use_safetensors=True,
    kwargs=dict(safety_checker=None, requires_safety_checker=False),
)
pipeline.save_pretrained(
    save_directory=str(out_dir),
    safe_serialization=True,
)
