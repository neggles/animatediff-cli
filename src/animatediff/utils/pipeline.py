import logging
from typing import Optional

import torch
import torch._dynamo as dynamo
from einops._torch_specific import allow_ops_in_compiled_graph

from animatediff.pipelines import AnimationPipeline
from animatediff.utils.device import get_memory_format, get_model_dtypes
from animatediff.utils.model import nop_train

logger = logging.getLogger(__name__)


def send_to_device(
    pipeline: AnimationPipeline,
    device: torch.device,
    freeze: bool = True,
    force_half: bool = False,
    compile: bool = False,
) -> AnimationPipeline:
    logger.info(f"Sending pipeline to device \"{device.type}{device.index if device.index else ''}\"")

    # Freeze model weights and force-disable training
    if freeze or compile:
        pipeline.freeze()
        pipeline.unet.train = nop_train
        pipeline.vae.train = nop_train
        pipeline.text_encoder.train = nop_train

    unet_dtype, tenc_dtype, vae_dtype = get_model_dtypes(device, force_half)
    model_memory_format = get_memory_format(device)

    pipeline.unet = pipeline.unet.to(device=device, dtype=unet_dtype, memory_format=model_memory_format)
    pipeline.text_encoder = pipeline.text_encoder.to(device=device, dtype=tenc_dtype)
    pipeline.vae = pipeline.vae.to(device=device, dtype=vae_dtype, memory_format=model_memory_format)

    # Compile model if enabled
    if compile:
        if not isinstance(pipeline.unet, dynamo.OptimizedModule):
            allow_ops_in_compiled_graph()  # make einops behave
            logger.warn("Enabling model compilation with TorchDynamo, this may take a while...")
            logger.warn("Model compilation is experimental and may not work as expected!")
            pipeline.unet = torch.compile(
                pipeline.unet,
                fullgraph=True,
                backend="inductor",
                mode="reduce-overhead",
            )
        else:
            logger.debug("Skipping model compilation, already compiled!")

    return pipeline


def get_context_params(
    length: int,
    context: Optional[int] = None,
    overlap: Optional[int] = None,
    stride: Optional[int] = None,
):
    if context is None:
        context = min(length, 16)
    if overlap is None:
        overlap = context // 2
    if stride is None:
        stride = 4
    return context, overlap, stride
