import logging
from enum import Enum

from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

logger = logging.getLogger(__name__)


class AnimateDiffusionScheduler(str, Enum):
    ddim = "ddim"
    pndm = "pndm"
    lms = "lms"
    euler = "euler"
    euler_a = "euler_a"
    dpm = "dpm_2"
    k_dpm = "k_dpm_2"
    dpm_a = "dpm_2_a"
    k_dpm_a = "k_dpm_2_a"
    dpmpp = "dpmpp"
    k_dpmpp = "k_dpmpp"


def get_scheduler(name: str, config: dict = {}):
    is_karras = name.startswith("k_")
    if is_karras:
        name = name.lstrip("k_")
        config["use_karras_sigmas"] = True

    match name:
        case "ddim":
            sched_class = DDIMScheduler
        case "pndm":
            sched_class = PNDMScheduler
        case "lms":
            sched_class = LMSDiscreteScheduler
        case "euler":
            sched_class = EulerDiscreteScheduler
        case "euler_a":
            sched_class = EulerAncestralDiscreteScheduler
        case val if val in ["dpm", "dpmpp"]:
            sched_class = DPMSolverMultistepScheduler
            if val == "dpm":
                config["algorithm_type"] = "dpmsolver"
            elif val == "dpmpp":
                config["algorithm_type"] = "dpmsolver++"
        case "dpm_2":
            sched_class = KDPM2DiscreteScheduler
        case "dpm_2_a":
            sched_class = KDPM2AncestralDiscreteScheduler
        case _:
            raise ValueError(f"Invalid scheduler {'k_' if is_karras else ''}{name}")

    if is_karras:
        config["use_karras_sigmas"] = True

    return sched_class.from_config(config)
