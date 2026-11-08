"""hklforge - diffractometer angle solvers generated exactly from a declared geometry."""
from .geometry import Circle, Geometry, wavevector
from .rotation import Degenerate
from .solve import Diffractometer, ModeError, Plan, plan

try:
    from importlib.metadata import PackageNotFoundError, version
    __version__ = version("hklforge")
except PackageNotFoundError:     # running from a source tree that was not installed
    __version__ = "unknown"

__all__ = ["Circle", "Geometry", "wavevector", "Diffractometer", "ModeError", "Degenerate", "Plan", "plan"]
