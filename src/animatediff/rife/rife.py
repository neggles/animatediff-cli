import logging
import subprocess
from pathlib import Path
from typing import Annotated, Optional

import typer

from animatediff import get_dir
from animatediff.utils.util import relative_path

from .ffmpeg import FfmpegEncoder, VideoCodec, codec_extn
from .ncnn import RifeNCNNOptions

rife_dir = get_dir("data/rife")
rife_ncnn_vulkan = rife_dir.joinpath("rife-ncnn-vulkan")

logger = logging.getLogger(__name__)

app: typer.Typer = typer.Typer(
    name="rife",
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    help="RIFE motion flow interpolation (MORE FPS!)",
)


@app.command(no_args_is_help=True)
def interpolate(
    rife_model: Annotated[
        str,
        typer.Option("--rife-model", "-m", help="RIFE model to use (subdirectory of data/rife/)"),
    ] = "rife-v4.6",
    in_fps: Annotated[
        int,
        typer.Option("--in-fps", "-I", help="Input frame FPS (8 for AnimateDiff)", show_default=True),
    ] = 8,
    frame_multiplier: Annotated[
        int,
        typer.Option(
            "--frame-multiplier", "-M", help="Multiply total frame count by this", show_default=True
        ),
    ] = 8,
    out_fps: Annotated[
        Optional[int],
        typer.Option(
            "--out-fps", "-F", help="Target FPS (uses minterpolate, not recommended)", show_default=True
        ),
    ] = None,
    codec: Annotated[
        VideoCodec,
        typer.Option("--codec", "-c", help="Output video codec", show_default=True),
    ] = VideoCodec.webm,
    lossless: Annotated[
        bool,
        typer.Option("--lossless", "-L", is_flag=True, help="Use lossless encoding (WebP only)"),
    ] = False,
    spatial_tta: Annotated[
        bool,
        typer.Option("--spatial-tta", "-x", is_flag=True, help="Enable RIFE Spatial TTA mode"),
    ] = False,
    temporal_tta: Annotated[
        bool,
        typer.Option("--temporal-tta", "-z", is_flag=True, help="Enable RIFE Temporal TTA mode"),
    ] = False,
    uhd: Annotated[
        bool,
        typer.Option("--uhd", "-u", is_flag=True, help="Enable RIFE UHD mode"),
    ] = False,
    frames_dir: Annotated[
        Path,
        typer.Argument(path_type=Path, file_okay=False, exists=True, help="Path to source frames directory"),
    ] = ...,
    out_file: Annotated[
        Optional[Path],
        typer.Argument(
            dir_okay=False,
            help="Path to output file (default: frames_dir/rife-output.<out_type>)",
            show_default=False,
        ),
    ] = None,
):
    rife_model_dir = rife_dir.joinpath(rife_model)
    if not rife_model_dir.joinpath("flownet.bin").exists():
        raise FileNotFoundError(f"RIFE model dir {rife_model_dir} does not have a model in it!")

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory {frames_dir} does not exist!")

    # where to put the RIFE interpolated frames (default: frames_dir/../<frames_dir>-rife)
    # TODO: make this configurable?
    rife_frames_dir = frames_dir.parent.joinpath(f"{frames_dir.name}-rife")
    rife_frames_dir.mkdir(exist_ok=True, parents=True)

    # build output file path
    file_extn = codec_extn(codec)
    if out_file is None:
        out_file = frames_dir.parent.joinpath(f"{frames_dir.name}-rife.{file_extn}")
    elif out_file.suffix != file_extn:
        logger.warn("Output file extension does not match codec, changing extension")
        out_file = out_file.with_suffix(file_extn)

    # build RIFE command and get args
    # This doesn't need to be a Pydantic model tbh. It could just be a function/class.
    rife_opts = RifeNCNNOptions(
        model_path=rife_model_dir,
        input_path=frames_dir,
        output_path=rife_frames_dir,
        time_step=1 / in_fps,  # TODO: make this configurable?
        spatial_tta=spatial_tta,
        temporal_tta=temporal_tta,
        uhd=uhd,
    )
    rife_args = rife_opts.get_args(frame_multiplier=frame_multiplier)

    # actually run RIFE
    logger.info("Running RIFE, this may take a little while...")
    with subprocess.Popen(
        [rife_ncnn_vulkan, *rife_args], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        errs = []
        for line in proc.stderr:
            line = line.decode("utf-8").strip()
            if line:
                logger.debug(line)
        stdout, _ = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"RIFE failed with code {proc.returncode}:\n" + "\n".join(errs))

    # now it is ffmpeg time
    logger.info("Creating ffmpeg encoder...")
    if out_fps is None:
        out_fps = in_fps * frame_multiplier

    encoder = FfmpegEncoder(
        frames_dir=rife_frames_dir,
        out_file=out_file,
        codec=codec,
        in_fps=min(out_fps, in_fps * frame_multiplier),
        out_fps=out_fps,
        lossless=lossless,
        interpolate=False,
    )
    logger.info("Encoding interpolated frames with ffmpeg...")
    result = encoder.encode()

    logger.debug(f"ffmpeg result: {result}")

    logger.info(f"Find the RIFE frames at: {relative_path(rife_frames_dir.absolute())}")
    logger.info(f"Find the output file at: {relative_path(out_file.absolute())}")
    logger.info("Done!")
