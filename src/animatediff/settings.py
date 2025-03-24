import json
import logging
from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from pydantic import BaseConfig, BaseSettings, Field, root_validator
from pydantic.env_settings import (
    EnvSettingsSource,
    InitSettingsSource,
    SecretsSettingsSource,
    SettingsSourceCallable,
)

from animatediff import get_dir
from animatediff.schedulers import DiffusionScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CKPT_EXTENSIONS = [".pt", ".ckpt", ".pth", ".safetensors"]


class JsonSettingsSource:
    __slots__ = ["json_config_path"]

    def __init__(
        self,
        json_config_path: Optional[Union[PathLike, list[PathLike]]] = list(),
    ) -> None:
        if isinstance(json_config_path, list):
            self.json_config_path = [Path(path) for path in json_config_path]
        else:
            self.json_config_path = [Path(json_config_path)] if json_config_path is not None else []

    def __call__(self, settings: BaseSettings) -> Dict[str, Any]:  # noqa C901
        classname = settings.__class__.__name__
        encoding = settings.__config__.env_file_encoding
        if len(self.json_config_path) == 0:
            pass  # no json config provided

        merged_config = dict()  # create an empty dict to merge configs into
        for idx, path in enumerate(self.json_config_path):
            if path.exists() and path.is_file():  # check if the path exists and is a file
                logger.debug(f"{classname}: loading config #{idx + 1} from {path}")
                merged_config.update(json.loads(path.read_text(encoding=encoding)))
                logger.debug(f"{classname}: config state #{idx + 1}: {merged_config}")
            else:
                raise FileNotFoundError(f"{classname}: config #{idx + 1} at {path} not found or not a file")

        logger.debug(f"{classname}: loaded config: {merged_config}")
        return merged_config  # return the merged config

    def __repr__(self) -> str:
        return f"JsonSettingsSource(json_config_path={repr(self.json_config_path)})"


class JsonConfig(BaseConfig):
    json_config_path: Optional[Union[Path, list[Path]]] = None
    env_file_encoding: str = "utf-8"

    @classmethod
    def customise_sources(
        cls,
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> Tuple[SettingsSourceCallable, ...]:
        # pull json_config_path from init_settings if passed, otherwise use the class var
        json_config_path = init_settings.init_kwargs.pop("json_config_path", cls.json_config_path)

        logger.debug(f"Using JsonSettingsSource for {cls.__name__}")
        json_settings = JsonSettingsSource(json_config_path=json_config_path)

        # return the new settings sources
        return (
            init_settings,
            json_settings,
        )


class InferenceConfig(BaseSettings):
    unet_additional_kwargs: dict[str, Any]
    noise_scheduler_kwargs: dict[str, Any]

    class Config(JsonConfig):
        json_config_path: Path


@lru_cache(maxsize=2)
def get_infer_config(
    config_path: Path = get_dir("config").joinpath("inference/default.json"),
) -> InferenceConfig:
    settings = InferenceConfig(json_config_path=config_path)
    return settings


class ModelConfig(BaseSettings):
    name: str = Field(...)  # Config name, not actually used for much of anything
    base: Optional[Path] = Field(None)  # Path to base checkpoint (if using a LoRA)
    path: Path = Field(...)  # Path to the model or LoRA checkpoint
    motion_module: Path = Field(...)  # Path to the motion module
    compile: bool = Field(False)  # whether to compile the model with TorchDynamo
    seed: list[int] = Field([])  # Seed(s) for the random number generators
    scheduler: DiffusionScheduler = Field(DiffusionScheduler.k_dpmpp_2m)  # Scheduler to use
    steps: int = 25  # Number of inference steps to run
    guidance_scale: float = 7.5  # CFG scale to use
    clip_skip: int = 1  # skip the last N-1 layers of the CLIP text encoder
    prompts: list[str | dict[int, str]] = Field([])  # Prompt(s) or prompt map(s) to use
    n_prompts: list[str] = Field([])  # Anti-prompt(s) to use

    class Config(JsonConfig):
        json_config_path: Path

    @root_validator(pre=True)
    def remap_old_names(cls, values: dict[str, Any]) -> dict[str, Any]:
        REMAPS = {
            "prompt": "prompts",
            "n_prompt": "n_prompts",
        }
        for old, new in REMAPS.items():
            if old in values:
                logger.warning(f"ModelConfig: {old} has been renamed to {new}, please update your config!")
                values[new] = values.pop(old)
        return values

    @property
    def save_name(self):
        if self.base is not None and str(self.base) != ".":
            return f"{self.name.lower()}-{self.path.stem.lower()}-{self.base.stem.lower()}"
        else:
            return f"{self.name.lower()}-{self.path.stem.lower()}"


@lru_cache(maxsize=2)
def get_model_config(config_path: Path) -> ModelConfig:
    settings = ModelConfig(json_config_path=config_path)
    return settings
