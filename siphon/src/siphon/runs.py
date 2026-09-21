"""Per-run output folder: config snapshot, log and results kept together."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .config import PACKAGE_ROOT


@dataclass
class Run:
    """A single run's output folder and the standard files inside it."""
    root: Path

    @property
    def log(self) -> Path:
        """Path to this run's log file."""
        return self.root / "siphon.log"

    @property
    def config_snapshot(self) -> Path:
        """Path to the copied config (single-config case)."""
        return self.root / "config.yaml"

    @property
    def report(self) -> Path:
        """Path to this run's markdown report."""
        return self.root / "report.md"

    @property
    def results(self) -> Path:
        """Path to this run's JSON results."""
        return self.root / "results.json"



def create(name: str, config_path: Path | Sequence[Path]) -> Run:
    """Create a timestamped, slugged run folder and copy the config(s) into it."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = "".join(c if c.isalnum() else "-" for c in name).strip("-").lower()  # filesystem-safe name
    root = PACKAGE_ROOT / "runs" / f"{stamp}-{slug}"
    root.mkdir(parents=True, exist_ok=True)
    run = Run(root)
    if isinstance(config_path, Path):
        shutil.copy(config_path, run.config_snapshot)
    else:
        snapshots = root / "configs"  # multiple configs (e.g. layer/transfer runs) go in a subfolder
        snapshots.mkdir(exist_ok=True)
        for path in config_path:
            shutil.copy(path, snapshots / path.name)
    return run


