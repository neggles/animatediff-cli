import logging
from pathlib import Path
from typing import Optional

from diffusers import StableDiffusionPipeline
from huggingface_hub import snapshot_download
from tqdm.rich import tqdm

from animatediff import HF_HUB_CACHE, get_dir

logger = logging.getLogger(__name__)

data_dir = get_dir("data")
checkpoint_dir = data_dir.joinpath("models/sd")
pipeline_dir = data_dir.joinpath("models/huggingface")

IGNORE_TF = ["*.git*", "*.h5", "tf_*"]
IGNORE_FLAX = ["*.git*", "flax_*", "*.msgpack"]
IGNORE_TF_FLAX = IGNORE_TF + IGNORE_FLAX

ALLOW_ST = ["*.safetensors", "*.yaml", "*.md", "*.json"]


class DownloadTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            {
                "ncols": 100,
                "dynamic_ncols": False,
                "disable": None,
            }
        )
        super().__init__(*args, **kwargs)


def get_model(repo_id: Path, target_dir: Path):
    target_dir.mkdir(exist_ok=True, parents=True)
    snapshot_download(
        repo_id=repo_id,
        revision="main",
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=IGNORE_TF_FLAX,
        cache_dir=HF_HUB_CACHE,
        tqdm_class=DownloadTqdm,
        max_workers=2,
        resume_download=True,
    )


def checkpoint_to_pipeline(
    checkpoint: Path,
    target_dir: Optional[Path] = None,
    save: bool = True,
) -> StableDiffusionPipeline:
    logger.info(f"Converting checkpoint: {checkpoint}")
    if target_dir is None:
        target_dir = pipeline_dir.joinpath(checkpoint.stem)
        logger.info(f"Using default output directory: {target_dir}")

    pipeline = StableDiffusionPipeline.from_single_file(
        pretrained_model_link_or_path=str(checkpoint.absolute()),
        local_files_only=True,
        load_safety_checker=False,
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    pipeline.save_pretrained(target_dir, safe_serialization=True)
    return pipeline, target_dir


def get_checkpoint_weights(checkpoint: Path):
    temp_pipeline: StableDiffusionPipeline
    temp_pipeline, _ = checkpoint_to_pipeline(checkpoint, save=False)
    unet_state_dict = temp_pipeline.unet.state_dict()
    tenc_state_dict = temp_pipeline.text_encoder.state_dict()
    vae_state_dict = temp_pipeline.vae.state_dict()
    return unet_state_dict, tenc_state_dict, vae_state_dict
