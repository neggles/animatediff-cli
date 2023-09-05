import logging
from os import PathLike
from pathlib import Path
from typing import Optional

from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.rich import tqdm

from animatediff import HF_HUB_CACHE, HF_LIB_NAME, HF_LIB_VER, get_dir
from animatediff.utils.util import relative_path

logger = logging.getLogger(__name__)

data_dir = get_dir("data")
checkpoint_dir = data_dir.joinpath("models/sd")
pipeline_dir = data_dir.joinpath("models/huggingface")

IGNORE_TF = ["*.git*", "*.h5", "tf_*"]
IGNORE_FLAX = ["*.git*", "flax_*", "*.msgpack"]
IGNORE_TF_FLAX = IGNORE_TF + IGNORE_FLAX


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


def get_hf_file(
    repo_id: Path,
    filename: str,
    target_dir: Path,
    subfolder: Optional[PathLike] = None,
    revision: Optional[str] = None,
    force: bool = False,
) -> Path:
    target_path = target_dir.joinpath(filename)
    if target_path.exists() and force is not True:
        raise FileExistsError(
            f"File {relative_path(target_path)} already exists! Pass force=True to overwrite"
        )

    target_dir.mkdir(exist_ok=True, parents=True)
    save_path = hf_hub_download(
        repo_id=str(repo_id),
        filename=filename,
        revision=revision or "main",
        subfolder=subfolder,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        cache_dir=HF_HUB_CACHE,
        resume_download=True,
    )
    return Path(save_path)


def get_hf_repo(
    repo_id: Path,
    target_dir: Path,
    subfolder: Optional[PathLike] = None,
    revision: Optional[str] = None,
    force: bool = False,
) -> Path:
    if target_dir.exists() and force is not True:
        raise FileExistsError(
            f"Target dir {relative_path(target_dir)} already exists! Pass force=True to overwrite"
        )

    target_dir.mkdir(exist_ok=True, parents=True)
    save_path = snapshot_download(
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
    return Path(save_path)


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
            logger.warning(f"Pipeline already exists at {relative_path(target_dir)}. Overwriting!")
            pipeline.save_pretrained(target_dir, safe_serialization=True)
        elif save and not pipeline_exists:
            logger.info(f"Saving pipeline to {relative_path(target_dir)}")
            pipeline.save_pretrained(target_dir, safe_serialization=True)
    return pipeline
