from functools import lru_cache
from math import ceil
from typing import Union

import torch


@lru_cache(maxsize=16, typed=True)
def maybe_bfloat16(
    device: Union[str, torch.device],
    fallback: torch.dtype = torch.float32,
) -> torch.dtype:
    """Returns torch.bfloat16 if available, otherwise the fallback dtype (defaults to float32)"""
    # make sure device is a torch.device
    device = torch.device(device) if not isinstance(device, torch.device) else device
    if torch.cuda.is_available() and device.type == "cuda":
        with device:  # use context manager to make sure device is set
            if torch.cuda.is_bf16_supported():
                # return bf16 if supported
                return torch.bfloat16
    # otherwise, return the fallback dtype
    return fallback


def device_info_str(device: torch.device) -> str:
    device_info = torch.cuda.get_device_properties(device)
    return (
        f"{device_info.name} {ceil(device_info.total_memory / 1024 ** 3)}GB, "
        + f"CC {device_info.major}.{device_info.minor}, {device_info.multi_processor_count} SM(s)"
    )


def model_dtype(model: str, device: torch.device) -> torch.dtype:
    match model:
        case "unet":
            return unet_dtype(device)
        case "tenc":
            return tenc_dtype(device)
        case "vae":
            return vae_dtype(device)
        case unknown:
            raise ValueError(f"Invalid model {unknown}")


def unet_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "cpu" else torch.float16


def tenc_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "cpu" else torch.float16


def vae_dtype(device: torch.device) -> torch.dtype:
    return maybe_bfloat16(device, fallback=torch.float32)
