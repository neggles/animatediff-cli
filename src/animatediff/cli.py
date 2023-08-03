import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from diffusers.utils.logging import set_verbosity_error as set_diffusers_verbosity_error
from rich.logging import RichHandler

from animatediff import __version__, console, get_dir
from animatediff.generate import create_pipeline, run_inference
from animatediff.pipelines.animation import AnimationPipeline
from animatediff.settings import (
    CKPT_EXTENSIONS,
    InferenceConfig,
    ModelConfig,
    get_infer_config,
    get_model_config,
)
from animatediff.utils.model import checkpoint_to_pipeline, get_base_model
from animatediff.utils.pipeline import get_context_params, send_to_device
from animatediff.utils.util import path_from_cwd, save_frames, save_video

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


try:
    from animatediff.rife import app as rife_app

    cli.add_typer(rife_app, name="rife")
except ImportError:
    logger.debug("RIFE not available, skipping...", exc_info=True)
    rife_app = None

# mildly cursed globals to allow for reuse of the pipeline if we're being called as a module
pipeline: Optional[AnimationPipeline] = None
last_model_path: Optional[Path] = None


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
            help="Base model to use (path or HF repo ID). You probably don't need to change this.",
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
            help="Path to a prompt configuration JSON file",
        ),
    ] = Path("config/prompts/01-ToonYou.json"),
    width: Annotated[
        int,
        typer.Option(
            "--width",
            "-W",
            min=512,
            max=3840,
            help="Width of generated frames",
            rich_help_panel="Generation",
        ),
    ] = 512,
    height: Annotated[
        int,
        typer.Option(
            "--height",
            "-H",
            min=512,
            max=2160,
            help="Height of generated frames",
            rich_help_panel="Generation",
        ),
    ] = 512,
    length: Annotated[
        int,
        typer.Option(
            "--length",
            "-L",
            min=1,
            max=999,
            help="Number of frames to generate",
            rich_help_panel="Generation",
        ),
    ] = 16,
    context: Annotated[
        Optional[int],
        typer.Option(
            "--context",
            "-C",
            min=1,
            max=24,
            help="Number of frames to condition on (default: max of <length> or 24)",
            show_default=False,
            rich_help_panel="Generation",
        ),
    ] = None,
    overlap: Annotated[
        Optional[int],
        typer.Option(
            "--overlap",
            "-O",
            min=1,
            max=12,
            help="Number of frames to overlap in context (default: context//2)",
            show_default=False,
            rich_help_panel="Generation",
        ),
    ] = None,
    stride: Annotated[
        Optional[int],
        typer.Option(
            "--stride",
            "-S",
            min=1,
            max=8,
            help="Max motion stride as a power of 2 (default: 4)",
            show_default=False,
            rich_help_panel="Generation",
        ),
    ] = None,
    repeats: Annotated[
        int,
        typer.Option(
            "--repeats",
            "-r",
            min=1,
            max=99,
            help="Number of times to repeat the prompt (default: 1)",
            show_default=False,
            rich_help_panel="Generation",
        ),
    ] = 1,
    device: Annotated[
        str,
        typer.Option(
            "--device", "-d", help="Device to run on (cpu, cuda, cuda:id)", rich_help_panel="Advanced"
        ),
    ] = "cuda",
    use_xformers: Annotated[
        bool,
        typer.Option(
            "--xformers",
            "-x",
            is_flag=True,
            help="Use XFormers instead of SDP Attention",
            rich_help_panel="Advanced",
        ),
    ] = False,
    force_half_vae: Annotated[
        bool,
        typer.Option(
            "--half-vae",
            is_flag=True,
            help="Force VAE to use fp16 (not recommended)",
            rich_help_panel="Advanced",
        ),
    ] = False,
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            "-o",
            path_type=Path,
            file_okay=False,
            help="Directory for output folders (frames, gifs, etc)",
            rich_help_panel="Output",
        ),
    ] = Path("output/"),
    no_frames: Annotated[
        bool,
        typer.Option(
            "--no-frames",
            "-N",
            is_flag=True,
            help="Don't save frames, only the animation",
            rich_help_panel="Output",
        ),
    ] = False,
    save_merged: Annotated[
        bool,
        typer.Option(
            "--save-merged",
            "-m",
            is_flag=True,
            help="Save a merged animation of all prompts",
            rich_help_panel="Output",
        ),
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

    # be quiet, diffusers. we care not for your safety checker
    set_diffusers_verbosity_error()

    config_path = config_path.absolute()
    logger.info(f"Using generation config: {path_from_cwd(config_path)}")
    model_config: ModelConfig = get_model_config(config_path)
    infer_config: InferenceConfig = get_infer_config()

    # set sane defaults for context, overlap, and stride if not supplied
    context, overlap, stride = get_context_params(length, context, overlap, stride)

    # turn the device string into a torch.device
    device: torch.device = torch.device(device)

    # Get the base model if we don't have it already
    logger.info(f"Using base model: {model_name_or_path}")
    base_model_path: Path = get_base_model(model_name_or_path, local_dir=get_dir("data/models/huggingface"))

    # get a timestamp for the output directory
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # make the output directory
    save_dir = out_dir.joinpath(f"{time_str}-{model_config.save_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Will save outputs to ./{path_from_cwd(save_dir)}")

    # beware the pipeline
    global pipeline
    global last_model_path
    if pipeline is None or last_model_path != model_config.base.resolve():
        pipeline = create_pipeline(
            base_model=base_model_path,
            model_config=model_config,
            infer_config=infer_config,
            use_xformers=use_xformers,
        )
        last_model_path = model_config.base.resolve()
    else:
        logger.info("Pipeline already loaded, skipping initialization")

    if pipeline.device == device:
        logger.info("Pipeline already on the correct device, skipping device transfer")
    else:
        pipeline = send_to_device(
            pipeline, device, freeze=True, force_half=force_half_vae, compile=model_config.compile
        )

    # save config to output directory
    logger.info("Saving prompt config to output directory")
    save_config_path = save_dir.joinpath("prompt.json")
    save_config_path.write_text(model_config.json(), encoding="utf-8")

    num_prompts = len(model_config.prompt)
    num_negatives = len(model_config.n_prompt)
    num_seeds = len(model_config.seed)
    gen_total = num_prompts * repeats  # total number of generations

    logger.info("Initialization complete!")
    logger.info(f"Generating {gen_total} animations from {num_prompts} prompts")
    outputs = []

    gen_num = 0  # global generation index
    # repeat the prompts if we're doing multiple runs
    for _ in range(repeats):
        for prompt in model_config.prompt:
            # get the index of the prompt, negative, and seed
            idx = gen_num % num_prompts
            logger.info(f"Running generation {gen_num + 1} of {gen_total} (prompt {idx + 1})")

            # allow for reusing the same negative prompt(s) and seed(s) for multiple prompts
            n_prompt = model_config.n_prompt[idx % num_negatives]
            seed = seed = model_config.seed[idx % num_seeds]

            # duplicated in run_inference, but this lets us use it for frame save dirs
            # TODO: Move gif Output out of run_inference...
            if seed == -1:
                seed = torch.seed()
            logger.info(f"Generation seed: {seed}")

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
                idx=gen_num,
                out_dir=save_dir,
                context_frames=context,
                context_overlap=overlap,
                context_stride=stride,
            )
            outputs.append(output)
            torch.cuda.empty_cache()
            if no_frames is not True:
                save_frames(output, save_dir.joinpath(f"{gen_num:02d}-{seed}"))

            # increment the generation number
            gen_num += 1

    logger.info("Generation complete!")
    if save_merged:
        logger.info("Output merged output video...")
        merged_output = torch.concat(outputs, dim=0)
        save_video(merged_output, save_dir.joinpath("final.gif"))

    logger.info("Done, exiting...")
    cli.info

    return save_dir


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
