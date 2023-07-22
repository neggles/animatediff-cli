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

from rich.console import Console

PACKAGE = __package__.replace("_", "-")
PACKAGE_ROOT = Path(__file__).parent.parent

HF_HOME = Path(environ.get("HF_HOME", "~/.cache/huggingface"))
HF_HUB_CACHE = Path(environ.get("HUGGINGFACE_HUB_CACHE", HF_HOME.joinpath("hub")))

console = Console(highlight=True)
err_console = Console(stderr=True)


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
