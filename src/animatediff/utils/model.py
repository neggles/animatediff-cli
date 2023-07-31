import logging
from functools import wraps
from os import PathLike
from pathlib import Path
from typing import Optional

from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download, snapshot_download
from torch import nn
from tqdm.rich import tqdm

from animatediff import HF_HUB_CACHE, HF_LIB_NAME, HF_LIB_VER, HF_MODULE_REPO, get_dir
from animatediff.utils.util import path_from_cwd

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


@wraps(nn.Module.train)
def nop_train(self, mode: bool = True):
    """No-op for monkeypatching train() call to prevent unfreezing module"""
    return self


def get_hf_file(
    repo_id: Path,
    filename: str,
    target_dir: Path,
    subfolder: Optional[PathLike] = None,
    revision: Optional[str] = None,
    force: bool = False,
):
    target_path = target_dir.joinpath(filename)
    if target_path.exists() and force is not True:
        raise FileExistsError(
            f"File {path_from_cwd(target_path)} already exists! Pass force=True to overwrite"
        )

    target_dir.mkdir(exist_ok=True, parents=True)
    hf_hub_download(
        repo_id=str(repo_id),
        filename=filename,
        revision=revision or "main",
        subfolder=subfolder,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        cache_dir=HF_HUB_CACHE,
        tqdm_class=DownloadTqdm,
        resume_download=True,
    )


def get_hf_repo(
    repo_id: Path,
    target_dir: Path,
    subfolder: Optional[PathLike] = None,
    revision: Optional[str] = None,
    force: bool = False,
):
    if target_dir.exists() and force is not True:
        raise FileExistsError(
            f"Target dir {path_from_cwd(target_dir)} already exists! Pass force=True to overwrite"
        )

    target_dir.mkdir(exist_ok=True, parents=True)
    snapshot_download(
        repo_id=str(repo_id),
        revision=revision or "main",
        subfolder=subfolder,
        library_name=HF_LIB_NAME,
        library_version=HF_LIB_VER,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=IGNORE_TF_FLAX,
        cache_dir=HF_HUB_CACHE,
        tqdm_class=DownloadTqdm,
        max_workers=2,
        resume_download=True,
    )


def get_hf_pipeline(
    repo_id: Path,
    target_dir: Path,
    save: bool = True,
    force_download: bool = False,
) -> StableDiffusionPipeline:
    pipeline_exists = target_dir.joinpath("model_index.json").exists()
    if pipeline_exists and force_download is not True:
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=target_dir,
            local_files_only=True,
        )
    else:
        target_dir.mkdir(exist_ok=True, parents=True)
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=str(repo_id).lstrip("./").replace("\\", "/"),
            cache_dir=HF_HUB_CACHE,
            resume_download=True,
        )
        if save and force_download:
            logger.warning(f"Pipeline already exists at {path_from_cwd(target_dir)}. Overwriting!")
            pipeline.save_pretrained(target_dir, safe_serialization=True)
        elif save and not pipeline_exists:
            logger.info(f"Saving pipeline to {path_from_cwd(target_dir)}")
            pipeline.save_pretrained(target_dir, safe_serialization=True)
    return pipeline


def checkpoint_to_pipeline(
    checkpoint: Path,
    target_dir: Optional[Path] = None,
    save: bool = True,
) -> StableDiffusionPipeline:
    logger.debug(f"Converting checkpoint {path_from_cwd(checkpoint)}")
    if target_dir is None:
        target_dir = pipeline_dir.joinpath(checkpoint.stem)

    pipeline = StableDiffusionPipeline.from_single_file(
        pretrained_model_link_or_path=str(checkpoint.absolute()),
        local_files_only=True,
        load_safety_checker=False,
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    if save:
        logger.info(f"Saving pipeline to {path_from_cwd(target_dir)}")
        pipeline.save_pretrained(target_dir, safe_serialization=True)
    return pipeline, target_dir


def get_checkpoint_weights(checkpoint: Path):
    temp_pipeline: StableDiffusionPipeline
    temp_pipeline, _ = checkpoint_to_pipeline(checkpoint, save=False)
    unet_state_dict = temp_pipeline.unet.state_dict()
    tenc_state_dict = temp_pipeline.text_encoder.state_dict()
    vae_state_dict = temp_pipeline.vae.state_dict()
    return unet_state_dict, tenc_state_dict, vae_state_dict


def get_motion_modules(
    repo_id: str = HF_MODULE_REPO,
    fp16: bool = False,
    force: bool = False,
):
    """Retrieve the motion modules from HuggingFace Hub."""
    module_files = ["mm_sd_v14.safetensors", "mm_sd_v15.safetensors"]
    module_dir = get_dir("data/models/motion-module")
    for file in module_files:
        target_path = module_dir.joinpath(file)
        if fp16:
            target_path = target_path.with_suffix(".fp16.safetensors")
        if target_path.exists() and force is not True:
            logger.debug(f"File {path_from_cwd(target_path)} already exists! Skipping download")
        else:
            result = hf_hub_download(
                repo_id=repo_id,
                filename=target_path.name,
                cache_dir=HF_HUB_CACHE,
                local_dir=module_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            logger.debug(f"Downloaded {path_from_cwd(result)}")


def get_base_model(model_name_or_path: str, local_dir: Path, force: bool = False):
    model_name_or_path = Path(model_name_or_path)

    model_save_dir = local_dir.joinpath(str(model_name_or_path).split("/")[-1])
    model_is_repo_id = False if model_name_or_path.joinpath("model_index.json").exists() else True

    # if we have a HF repo ID, download it
    if model_is_repo_id:
        logger.debug("Base model is a HuggingFace repo ID")
        if model_save_dir.joinpath("model_index.json").exists():
            logger.debug(f"Base model already downloaded to: {path_from_cwd(model_save_dir)}")
        else:
            logger.info(f"Downloading base model from {model_name_or_path}...")
            _ = get_hf_pipeline(model_name_or_path, model_save_dir.absolute(), save=True)
        model_name_or_path = model_save_dir

    return model_name_or_path
