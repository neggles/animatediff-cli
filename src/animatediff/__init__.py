try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")

from rich.console import Console

console = Console(highlight=True)
err_console = Console(stderr=True)
