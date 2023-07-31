try:
    from ._version import (
        version as __version__,
        version_tuple,
    )
except ImportError:
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")

from functools import lru_cache
from os import environ
from pathlib import Path
from warnings import filterwarnings

from rich.console import Console
from tqdm import TqdmExperimentalWarning

PACKAGE = __package__.replace("_", "-")
PACKAGE_ROOT = Path(__file__).parent.parent

HF_HOME = Path(environ.get("HF_HOME", "~/.cache/huggingface"))
HF_HUB_CACHE = Path(environ.get("HUGGINGFACE_HUB_CACHE", HF_HOME.joinpath("hub")))

HF_LIB_NAME = "animatediff-cli"
HF_LIB_VER = __version__
HF_MODULE_REPO = "neggles/animatediff-modules"

console = Console(highlight=True)
err_console = Console(stderr=True)

# shhh torch, don't worry about it it's fine
filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
# you too tqdm
filterwarnings("ignore", category=TqdmExperimentalWarning)


@lru_cache(maxsize=4)
def get_dir(dirname: str = "data") -> Path:
    if PACKAGE_ROOT.name == "src":
        # we're installed in editable mode from within the repo
        dirpath = PACKAGE_ROOT.parent.joinpath(dirname)
    else:
        # we're installed normally, so we just use the current working directory
        dirpath = Path.cwd().joinpath(dirname)
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath.absolute()


__all__ = [
    "__version__",
    "version_tuple",
    "PACKAGE",
    "PACKAGE_ROOT",
    "HF_HOME",
    "HF_HUB_CACHE",
    "console",
    "err_console",
    "get_dir",
    "models",
    "pipelines",
    "rife",
    "utils",
    "cli",
    "generate",
    "schedulers",
    "settings",
]
