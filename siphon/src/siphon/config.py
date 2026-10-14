# Configuration: one YAML file describes one attack target.
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]      # .../siphon


@dataclass
class Config:
    name: str
    column: int                      # which network input carries the hot spot
    neurons: int                     # neurons in the attacked layer
    profiling: Path                  # campaign directory (k0..kN + manifest.txt)
    attack: Path                     # attack traces, real weights in place
    firmware_config: Path            # network_config.h, for ground truth
    profiling_column: int | None = None   # column the campaign was captured on
    window: tuple[int, int] = (0, 10000)  # trace crop
    profiling_traces: int = 1000
    attack_traces: int = 300
    pois_per_neuron: int = 1
    poi_separation: int = 300        # minimum gap between two neurons' POIs
    poi_method: str = "snr"          # "snr" over |v|, or "tvla" t-test per class
    template_key: str = "v"          # index templates by v, or by "hw" = HW(v)
    # zero is a legal weight and real networks have them
    hypotheses: tuple[int, ...] = field(
        default_factory=lambda: tuple(range(-7, 8))
    )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Config":
        # Load YAML and resolve the three path fields against the package root.
        raw = yaml.safe_load(Path(path).read_text())
        for key in ("profiling", "attack", "firmware_config"):
            raw[key] = (PACKAGE_ROOT / raw[key]).resolve()
        raw["window"] = tuple(raw.get("window", (0, 10000)))
        if "hypotheses" in raw:
            raw["hypotheses"] = tuple(raw["hypotheses"])
        return cls(**raw)

    @property
    def prof_column(self) -> int:
        # The column whose inputs drove profiling (differs from `column` when a
        # campaign from one column scores another column's attack set).
        return self.column if self.profiling_column is None else self.profiling_column

    @property
    def transferred(self) -> bool:
        # True when profiling and attack columns differ (template transfer).
        return self.prof_column != self.column
