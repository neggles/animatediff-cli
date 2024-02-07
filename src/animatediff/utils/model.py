import logging
from pathlib import Path
from typing import Optional, TypeVar

import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
from torch import nn

from animatediff import HF_HUB_CACHE, HF_MODULE_REPO, get_dir
from animatediff.utils.huggingface import get_hf_pipeline
from animatediff.utils.util import relative_path

logger = logging.getLogger(__name__)

data_dir = get_dir("data")
checkpoint_dir = data_dir.joinpath("models/sd")
pipeline_dir = data_dir.joinpath("models/huggingface")

# for the nop_train() monkeypatch
T = TypeVar("T", bound=nn.Module)

MMV2_DIM_KEY = (
    "up_blocks.0.motion_modules.1.temporal_transformer.transformer_blocks.0.attention_blocks.0.pos_encoder.pe"
)


def nop_train(self: T, mode: bool = True) -> T:
    """No-op for monkeypatching train() call to prevent unfreezing module"""
    return self


def autocast_device(device: torch.device | str | None):
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        return torch.device(device).type
    return "cuda"


def get_base_model(model_name_or_path: str, local_dir: Path, force: bool = False) -> Path:
    model_name_or_path = Path(model_name_or_path)

    model_save_dir = local_dir.joinpath(str(model_name_or_path).split("/")[-1]).resolve()
    model_is_repo_id = False if model_name_or_path.joinpath("model_index.json").exists() else True

    # if we have a HF repo ID, download it
    if model_is_repo_id:
        logger.debug("Base model is a HuggingFace repo ID")
        if model_save_dir.joinpath("model_index.json").exists():
            logger.debug(f"Base model already downloaded to: {relative_path(model_save_dir)}")
        else:
            logger.info(f"Downloading base model from {model_name_or_path}...")
            _ = get_hf_pipeline(model_name_or_path, model_save_dir, save=True, force_download=force)
        model_name_or_path = model_save_dir

    return Path(model_name_or_path)


def checkpoint_to_pipeline(
    checkpoint: Path,
    target_dir: Optional[Path] = None,
    save: bool = True,
) -> StableDiffusionPipeline:
    logger.debug(f"Converting checkpoint {relative_path(checkpoint)}")
    if target_dir is None:
        target_dir = pipeline_dir.joinpath(checkpoint.stem)

    pipeline = StableDiffusionPipeline.from_single_file(
        pretrained_model_link_or_path=str(checkpoint.absolute()),
        local_files_only=True,
        load_safety_checker=False,
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    if save:
        logger.info(f"Saving pipeline to {relative_path(target_dir)}")
        pipeline.save_pretrained(target_dir, safe_serialization=True)
    return pipeline, target_dir


def get_checkpoint_weights(checkpoint: Path):
    temp_pipeline: StableDiffusionPipeline
    temp_pipeline, _ = checkpoint_to_pipeline(checkpoint, save=False)
    unet_state_dict = temp_pipeline.unet.state_dict()
    tenc_state_dict = temp_pipeline.text_encoder.state_dict()
    vae_state_dict = temp_pipeline.vae.state_dict()
    return unet_state_dict, tenc_state_dict, vae_state_dict


def ensure_motion_modules(
    repo_id: str = HF_MODULE_REPO,
    module_files: list[str] = ["mm_sd_v14", "mm_sd_v15", "mm_sd_v15_v2"],
    fp16: bool = False,
    force: bool = False,
):
    """Retrieve the motion modules from HuggingFace Hub."""
    module_dir = get_dir("data/models/motion-module")
    for file in module_files:
        target_path = module_dir.joinpath(file).with_suffix(".safetensors")
        if fp16:
            target_path = target_path.with_suffix(".fp16.safetensors")
        if target_path.exists() and force is not True:
            logger.debug(f"File {relative_path(target_path)} already exists, skipping download")
        else:
            result = hf_hub_download(
                repo_id=repo_id,
                filename=target_path.name,
                cache_dir=HF_HUB_CACHE,
                local_dir=module_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            logger.debug(f"Downloaded {relative_path(result)}")
