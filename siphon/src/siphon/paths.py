"""Path resolution: CLI paths resolve against cwd, config-file paths against the package root."""
from __future__ import annotations

from pathlib import Path

from .config import PACKAGE_ROOT


def out(path: Path | str) -> Path:
    """Resolve a write-target path to an absolute one, relative to cwd if not already absolute."""
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def existing(path: Path | str, what: str = "path") -> Path:
    """Resolve a read path and raise, with a package-root hint, if it doesn't exist."""
    p = out(path)
    if not p.exists():
        hint = ""
        alt = (PACKAGE_ROOT / Path(path)).resolve()  # check the common mistake: package-root-relative
        if alt.exists():
            hint = f"\n  did you mean {alt}? (paths are relative to where you are)"
        raise FileNotFoundError(f"{what} not found: {p}{hint}")
    return p


def traces_in(path: Path) -> int:
    """Count trace_*.txt files in a directory (0 if it isn't one)."""
    return len(list(Path(path).glob("trace_*.txt"))) if Path(path).is_dir() else 0


def has_traces(path: Path) -> bool:
    """Whether a directory already holds captured traces."""
    return traces_in(path) > 0
