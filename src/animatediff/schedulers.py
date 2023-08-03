import logging
from enum import Enum

from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)

logger = logging.getLogger(__name__)


# See https://github.com/huggingface/diffusers/issues/4167 for more details on sched mapping from A1111
class DiffusionScheduler(str, Enum):
    ddim = "ddim"  # DDIM
    pndm = "pndm"  # PNDM
    heun = "heun"  # Heun
    unipc = "unipc"  # UniPC
    euler = "euler"  # Euler
    euler_a = "euler_a"  # Euler a

    lms = "lms"  # LMS
    k_lms = "k_lms"  # LMS Karras

    dpm_2 = "dpm_2"  # DPM2
    k_dpm_2 = "k_dpm_2"  # DPM2 Karras

    dpm_2_a = "dpm_2_a"  # DPM2 a
    k_dpm_2_a = "k_dpm_2_a"  # DPM2 a Karras

    dpmpp_2m = "dpmpp_2m"  # DPM++ 2M
    k_dpmpp_2m = "k_dpmpp_2m"  # DPM++ 2M Karras

    dpmpp_sde = "dpmpp_sde"  # DPM++ SDE
    k_dpmpp_sde = "k_dpmpp_sde"  # DPM++ SDE Karras

    dpmpp_2m_sde = "dpmpp_2m_sde"  # DPM++ 2M SDE
    k_dpmpp_2m_sde = "k_dpmpp_2m_sde"  # DPM++ 2M SDE Karras


def get_scheduler(name: str, config: dict = {}):
    is_karras = name.startswith("k_")
    if is_karras:
        # strip the k_ prefix and add the karras sigma flag to config
        name = name.lstrip("k_")
        config["use_karras_sigmas"] = True

    match name:
        case DiffusionScheduler.ddim:
            sched_class = DDIMScheduler
        case DiffusionScheduler.pndm:
            sched_class = PNDMScheduler
        case DiffusionScheduler.heun:
            sched_class = HeunDiscreteScheduler
        case DiffusionScheduler.unipc:
            sched_class = UniPCMultistepScheduler
        case DiffusionScheduler.euler:
            sched_class = EulerDiscreteScheduler
        case DiffusionScheduler.euler_a:
            sched_class = EulerAncestralDiscreteScheduler
        case DiffusionScheduler.lms:
            sched_class = LMSDiscreteScheduler
        case DiffusionScheduler.dpm_2:
            # Equivalent to DPM2 in K-Diffusion
            sched_class = KDPM2DiscreteScheduler
        case DiffusionScheduler.dpm_2_a:
            # Equivalent to `DPM2 a`` in K-Diffusion
            sched_class = KDPM2AncestralDiscreteScheduler
        case DiffusionScheduler.dpmpp_2m:
            # Equivalent to `DPM++ 2M` in K-Diffusion
            sched_class = DPMSolverMultistepScheduler
            config["algorithm_type"] = "dpmsolver++"
            config["solver_order"] = 2
        case DiffusionScheduler.dpmpp_sde:
            # Equivalent to `DPM++ SDE` in K-Diffusion
            sched_class = DPMSolverSinglestepScheduler
        case DiffusionScheduler.dpmpp_2m_sde:
            # Equivalent to `DPM++ 2M SDE` in K-Diffusion
            sched_class = DPMSolverMultistepScheduler
            config["algorithm_type"] = "sde-dpmsolver++"
        case _:
            raise ValueError(f"Invalid scheduler '{'k_' if is_karras else ''}{name}'")

    return sched_class.from_config(config)
