from animatediff import get_dir
from animatediff.cli import generate, logger

config_dir = get_dir("config")

config_path = config_dir.joinpath("prompts/test.json")
width = 512
height = 512
length = 32
context = 16
stride = 4

logger.warn("Running first-round generation test, this should load the full model.\n\n")
out_dir = generate(
    config_path=config_path,
    width=width,
    height=height,
    length=length,
    context=context,
    stride=stride,
)
logger.warn(f"Generated animation to {out_dir}")

logger.warn("\n\nRunning second-round generation test, this should reuse the already loaded model.\n\n")
out_dir = generate(
    config_path=config_path,
    width=width,
    height=height,
    length=length,
    context=context,
    stride=stride,
)
logger.warn(f"Generated animation to {out_dir}")

logger.error("If the second round didn't talk about reloading the model, it worked! yay!")
