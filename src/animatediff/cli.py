import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from diffusers.utils.logging import set_verbosity_error as set_diffusers_verbosity_error
from rich.logging import RichHandler
from tqdm import TqdmExperimentalWarning

from animatediff import __version__, console, get_dir
from animatediff.generate import create_pipeline, run_inference
from animatediff.settings import (
    CKPT_EXTENSIONS,
    InferenceConfig,
    ModelConfig,
    get_infer_config,
    get_model_config,
)
from animatediff.utils.model import checkpoint_to_pipeline, get_hf_pipeline
from animatediff.utils.util import device_info_str, save_frames, save_video

cli: typer.Typer = typer.Typer(
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

data_dir = get_dir("data")
checkpoint_dir = data_dir.joinpath("models/sd")
pipeline_dir = data_dir.joinpath("models/huggingface")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        RichHandler(console=console, rich_tracebacks=True),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# shhh torch, don't worry about it it's fine
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
# you too tqdm
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def version_callback(value: bool):
    if value:
        console.print(f"AnimateDiff v{__version__}")
        raise typer.Exit()


@cli.command()
def generate(
    model_name_or_path: Annotated[
        Path,
        typer.Option(
            ...,
            "--model-path",
            "-m",
            path_type=Path,
            help="Base model to use for generation. Can be a local path or a HuggingFace model name.",
        ),
    ] = Path("runwayml/stable-diffusion-v1-5"),
    config_path: Annotated[
        Path,
        typer.Option(
            "--config-path",
            "-c",
            path_type=Path,
            exists=True,
            readable=True,
            dir_okay=False,
            help="Path to a prompt/generation config file",
        ),
    ] = Path("config/prompts/01-ToonYou.json"),
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            "-o",
            path_type=Path,
            file_okay=False,
            help="Base output directory (subdirectory will be created for each run)",
        ),
    ] = Path("output/"),
    width: Annotated[
        int,
        typer.Option("--width", "-W", min=512, max=3840),
    ] = 512,
    height: Annotated[
        int,
        typer.Option("--height", "-H", min=512, max=2160),
    ] = 512,
    length: Annotated[
        int,
        typer.Option("--length", "-L", min=1, max=60),
    ] = 16,
    device: Annotated[
        str,
        typer.Option("--device", "-d", help="Device to run on (cpu, cuda, cuda:id)"),
    ] = "cuda",
    force_half_vae: Annotated[
        bool,
        typer.Option("--half-vae", is_flag=True, help="Force VAE to use fp16 if bf16 is not supported"),
    ] = False,
    no_frames: Annotated[
        bool,
        typer.Option("--no-frames", "-N", is_flag=True, help="Don't save frames, only the animation"),
    ] = False,
    save_merged: Annotated[
        bool,
        typer.Option("--save-merged", "-m", is_flag=True, help="Save a merged animation of all prompts"),
    ] = False,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            callback=version_callback,
            is_eager=True,
            is_flag=True,
            help="Show version",
        ),
    ] = None,
):
    """
    Do the thing. Make the animation happen. Waow.
    """

    config_path = config_path.absolute()
    logger.info(f"Using generation config: {config_path}")
    model_config: ModelConfig = get_model_config(config_path)
    infer_config: InferenceConfig = get_infer_config()

    # bluh bluh safety checker
    set_diffusers_verbosity_error()

    device = torch.device(device)
    device_info = torch.cuda.get_device_properties(device)

    logger.info(device_info_str(device_info))
    has_bf16 = torch.cuda.is_bf16_supported()
    if has_bf16:
        logger.info("Device supports bfloat16, will run VAE in bf16")
        vae_dtype = torch.bfloat16
    elif force_half_vae:
        logger.warn("bfloat16 not supported, but VAE forced to fp16!")
        vae_dtype = torch.float16
    else:
        logger.info("bfloat16 not supported, will run VAE in fp32")
        vae_dtype = torch.float32

    unet_dtype = torch.float16
    tenc_dtype = torch.float16

    logger.info(f"Using model: {model_name_or_path}")
    model_is_repo_id = False if model_name_or_path.exists() else True

    # if we have a HF repo ID, download it
    if model_is_repo_id:
        model_save_dir = get_dir("data/models/huggingface").joinpath(str(model_name_or_path).split("/")[-1])
        if model_save_dir.exists():
            logger.info(f"Model already downloaded to: {model_save_dir}")
        else:
            logger.info(f"Downloading model from huggingface repo: {model_name_or_path}")
            get_hf_pipeline(model_name_or_path, model_save_dir)
        model_name_or_path = model_save_dir

    # get a timestamp for the output directory
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # make the output directory
    save_dir = out_dir.joinpath(f"{time_str}-{model_config.save_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving output to {save_dir}")

    # beware
    logger.info("Creating pipeline...")
    pipeline = create_pipeline(
        base_model=model_name_or_path,
        model_config=model_config,
        infer_config=infer_config,
    )
    pipeline.unet = pipeline.unet.to(device=device, dtype=unet_dtype)
    pipeline.text_encoder = pipeline.text_encoder.to(device=device, dtype=tenc_dtype)
    pipeline.vae = pipeline.vae.to(device=device, dtype=vae_dtype)

    num_prompts = len(model_config.prompt)
    logger.info(f"Generating {num_prompts} animations")
    outputs = []
    for idx, prompt in enumerate(model_config.prompt):
        logger.info(f"Running prompt {idx + 1}/{num_prompts}")
        n_prompt = model_config.n_prompt[idx] if len(model_config.n_prompt) > 1 else model_config.n_prompt[0]
        seed = seed = model_config.seed[idx] if len(model_config.seed) > 1 else model_config.seed[0]
        output = run_inference(
            pipeline=pipeline,
            prompt=prompt,
            n_prompt=n_prompt,
            seed=seed,
            steps=model_config.steps,
            guidance_scale=model_config.guidance_scale,
            width=width,
            height=height,
            duration=length,
            idx=idx,
            out_dir=save_dir,
        )
        outputs.append(output)
        torch.cuda.empty_cache()
        if no_frames is not True:
            save_frames(output, save_dir.joinpath(f"{idx}"))

    logger.info("Generation complete!")
    if save_merged:
        logger.info("Saving merged output video...")
        merged_output = torch.concat(outputs, dim=0)
        save_video(merged_output, save_dir.joinpath("final.gif"), n_rows=num_prompts)

    logger.info("Done, exiting...")
    exit(0)


@cli.command()
def convert(
    checkpoint: Annotated[
        Path,
        typer.Option(
            "--checkpoint",
            "-i",
            path_type=Path,
            dir_okay=False,
            exists=True,
            help="Path to a model checkpoint file",
        ),
    ] = ...,
    out_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--out-dir",
            "-o",
            path_type=Path,
            file_okay=False,
            help="Target directory for converted model",
        ),
    ] = None,
):
    """Convert a StableDiffusion checkpoint into a Diffusers pipeline"""
    logger.info(f"Converting checkpoint: {checkpoint}")
    _, pipeline_dir = checkpoint_to_pipeline(checkpoint, target_dir=out_dir)
    logger.info(f"Converted to HuggingFace pipeline at {pipeline_dir}")


@cli.command()
def merge(
    checkpoint: Annotated[
        Path,
        typer.Option(
            "--checkpoint",
            "-i",
            path_type=Path,
            dir_okay=False,
            exists=True,
            help="Path to a model checkpoint file",
        ),
    ] = ...,
    out_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--out-dir",
            "-o",
            path_type=Path,
            file_okay=False,
            help="Target directory for converted model",
        ),
    ] = None,
):
    """Convert a StableDiffusion checkpoint into an AnimationPipeline"""
    raise NotImplementedError("Sorry, haven't implemented this yet!")

    # if we have a checkpoint, convert it to HF automagically
    if checkpoint.is_file() and checkpoint.suffix in CKPT_EXTENSIONS:
        logger.info(f"Loading model from checkpoint: {checkpoint}")
        # check if we've already converted this model
        model_dir = pipeline_dir.joinpath(checkpoint.stem)
        if model_dir.joinpath("model_index.json").exists():
            # we have, so just use that
            logger.info("Found converted model in {model_dir}, will not convert")
            logger.info("Delete the output directory to re-run conversion.")
        else:
            # we haven't, so convert it
            logger.info("Converting checkpoint to HuggingFace pipeline...")
            pipeline, model_dir = checkpoint_to_pipeline(checkpoint)
    logger.info("Done!")
