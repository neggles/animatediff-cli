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
from animatediff.utils.device import device_info_str, model_dtype
from animatediff.utils.model import checkpoint_to_pipeline, get_hf_pipeline
from animatediff.utils.util import save_frames, save_video

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

try:
    from animatediff.rife import app as rife_app

    cli.add_typer(rife_app, name="rife")
except ImportError:
    logger.debug("RIFE not available, skipping...", exc_info=True)
    rife_app = None


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
            help="Directory for output folders (frames, gifs, etc)",
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
    use_xformers: Annotated[
        bool,
        typer.Option("--xformers", "-x", is_flag=True, help="Use XFormers instead of SDP Attention"),
    ] = False,
    force_half_vae: Annotated[
        bool,
        typer.Option("--half-vae", is_flag=True, help="Force VAE to use fp16 (not recommended)"),
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

    device: torch.device = torch.device(device)
    if device.type == "cpu":
        logger.warn("Device explicitly set to CPU, will run everything in fp32")
        logger.warn("This is likely to be *incredibly* slow, but I don't tell you how to live.")
        if use_xformers:
            logger.error("XFormers is not supported on CPU! Disabling it and continuing...")
            use_xformers = False
    elif device.type == "cuda":
        if torch.cuda.is_available():
            logger.info("CUDA is available, will use mixed-precision inference")
            device_info = torch.cuda.get_device_properties(device)
            logger.info(f"Using device: {device_info_str(device_info)}")
        else:
            logger.critical("CUDA is not available but device is set to CUDA! Exiting...")
            raise RuntimeError("CUDA is not available but device is set to CUDA!")
    else:
        logger.info(f"Using non-CUDA device: {device.type}{device.index if device.index is not None else ''}")
        if use_xformers:
            logger.warn("XFormers may or may not work on this device! Will try anyway...")

    # get dtypes for each model, based on the device we're using.
    unet_dtype = model_dtype("unet", device)  # fp16 unless on CPU
    tenc_dtype = model_dtype("tenc", device)  # fp16 unless on CPU
    vae_dtype = model_dtype("vae", device)  # bfloat16 if available, otherwise fp32

    if device.type == "cpu" and force_half_vae:
        logger.critical("Can't force VAE to fp16 mode on CPU! Exiting...")
        raise RuntimeError("Can't force VAE to fp16 mode on CPU!")

    if force_half_vae:
        # you probably shouldn't do this, but I'm not your mom
        if torch.cuda.is_bf16_supported():
            logger.warn("Forcing VAE to use fp16 despite bfloat16 support! This is a bad idea!")
            logger.warn("If you're not sure why you're doing this, you probably shouldn't be.")
            vae_dtype = torch.float16
        else:
            logger.warn("Forcing VAE to use fp16 instead of fp32 on CUDA! This may result in black outputs!")
            logger.warn("Running a VAE in fp16 can result in black images or poor output quality.")
            logger.warn("I don't tell you how to live, but you probably shouldn't do this.")
            vae_dtype = torch.float16

    logger.info(f"Selected data types: {unet_dtype=}, {tenc_dtype=}, {vae_dtype=}")

    # Get the base model if we don't have it already
    logger.info(f"Using base model: {model_name_or_path}")
    model_save_dir = get_dir("data/models/huggingface").joinpath(str(model_name_or_path).split("/")[-1])
    model_is_repo_id = False if model_name_or_path.joinpath("model_index.json").exists() else True
    # if we have a HF repo ID, download it
    if model_is_repo_id:
        logger.info("Base model is a HuggingFace repo ID")
        if model_save_dir.joinpath("model_index.json").exists():
            logger.info(f"Base model already downloaded to: {model_save_dir}")
        else:
            logger.info(f"Downloading from {model_name_or_path}")
            get_hf_pipeline(model_name_or_path, model_save_dir.absolute())
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
        use_xformers=use_xformers,
    )
    logger.info("Loading pipeline into device...")
    pipeline.unet = pipeline.unet.to(device=device, dtype=unet_dtype)
    pipeline.text_encoder = pipeline.text_encoder.to(device=device, dtype=tenc_dtype)
    pipeline.vae = pipeline.vae.to(device=device, dtype=vae_dtype)

    # save config to output directory
    logger.info("Saving prompt config to output dir")
    save_config_path = save_dir.joinpath("prompt.json")
    save_config_path.write_text(model_config.json(), encoding="utf-8")

    num_prompts = len(model_config.prompt)
    logger.info(f"Initialization complete! Starting generation for {num_prompts} animations...")
    outputs = []
    for idx, prompt in enumerate(model_config.prompt):
        logger.info(f"Running prompt {idx + 1}/{num_prompts}")
        n_prompt = model_config.n_prompt[idx] if len(model_config.n_prompt) > 1 else model_config.n_prompt[0]
        seed = seed = model_config.seed[idx] if len(model_config.seed) > 1 else model_config.seed[0]

        # duplicated in run_inference, but this lets us use it for frame save dirs
        # TODO: Move gif saving out of run_inference...
        if seed == -1:
            seed = torch.seed()
        logger.info(f"Using seed {seed}")

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
            save_frames(output, save_dir.joinpath(f"{idx:02d}-{seed}"))

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
