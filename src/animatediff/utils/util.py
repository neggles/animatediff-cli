from math import ceil
from os import PathLike
from pathlib import Path
from typing import Union

import imageio.v3 as iio
import numpy as np
import torch
from einops import rearrange
from networkx import tensor_product
from PIL import Image
from torch import Tensor
from torchvision.utils import save_image
from tqdm.rich import tqdm

from animatediff.pipelines.pipeline_animation import AnimationPipeline


def save_frames(video: Tensor, frames_dir: PathLike):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = rearrange(video, "b c t h w -> t b c h w")
    for idx, frame in enumerate(tqdm(frames, desc=f"Saving frames to {frames_dir.stem}")):
        save_image(frame, frames_dir.joinpath(f"{idx:03d}.png"))


def save_video(video: Tensor, save_path: PathLike, fps: int = 8):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if video.ndim == 5:
        # batch, channels, frame, width, height -> frame, channels, width, height
        frames = video.permute(0, 2, 1, 3, 4).squeeze(0)
    elif video.ndim == 4:
        # channels, frame, width, height -> frame, channels, width, height
        frames = video.permute(1, 0, 2, 3)
    else:
        raise ValueError(f"video must be 4 or 5 dimensional, got {video.ndim}")

    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    frames = frames.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        fp=save_path, format="GIF", append_images=images[1:], save_all=True, duration=(1 / fps * 1000), loop=0
    )


def device_info_str(device: torch.device) -> str:
    device_info = torch.cuda.get_device_properties(device)
    return (
        f"Device: {device_info.name} {ceil(device_info.total_memory / 1024 ** 3)}GB, "
        + f"CC {device_info.major}.{device_info.minor}, {device_info.multi_processor_count} SM(s)"
    )
