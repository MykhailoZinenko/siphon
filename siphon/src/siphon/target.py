"""Flash a synthetic target network and save its ground truth, for scoring the attack against it."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import PACKAGE_ROOT
from .firmware import CONFIG_HEADER, PRISTINE, build, set_array, set_row

STUDENT_BACKUP = PACKAGE_ROOT / "firmware" / "network_config.h.student"


def generate(seed: int, neurons: int = 6, inputs: int = 10,
             weights: tuple[int, int] = (-7, 7), bias: int = 20,
             inv_scale: tuple[float, float] = (22.0, 65.0)) -> dict:
    """A plausible quantised network: int8 weights, small biases, sane scales.

    `weights` is the alphabet the layer is quantised to, e.g. (-7, 7) for the
    signed 4-bit grid or (0, 15) for the unsigned one. The scales follow it, so
    a wider alphabet does not simply saturate every activation.
    """
    rng = np.random.default_rng(seed)
    shift = 4                                     # fixed requantisation shift
    span = max(abs(weights[0]), abs(weights[1])) / 7.0      # relative to the -7..7 grid
    inv = rng.uniform(*inv_scale, neurons) * span  # inverse scale, sampled per neuron
    out = {
        "seed": seed,
        "L1_W": rng.integers(weights[0], weights[1] + 1, (neurons, inputs)).tolist(),
        "L1_B": rng.integers(-bias, bias + 1, neurons).tolist(),
        "L1_M0": np.round(2.0 ** (31 + shift) / inv).astype(np.int64).tolist(),  # requant multiplier
        "L1_N": [shift] * neurons,
        "L2_W": rng.integers(weights[0], weights[1] + 1, neurons).tolist(),
        "L2_B": [int(rng.integers(-5, 6))],
    }
    out["inv_scale"] = [2.0 ** (31 + shift) / m for m in out["L1_M0"]]   # actual scale after rounding
    return out


def flash(model: dict, progress=lambda _m: None) -> Path:
    """Patch the network into network_config.h, make it the new pristine, and rebuild."""
    if not STUDENT_BACKUP.exists():
        shutil.copy(PRISTINE, STUDENT_BACKUP)   # keep the original network once, before overwriting
        progress(f"kept the original network as {STUDENT_BACKUP.name}")

    shutil.copy(PRISTINE, CONFIG_HEADER)
    # L1_W is written row by row, since set_row patches one row at a time
    for row, values in enumerate(model["L1_W"]):
        set_row("L1_W", [int(v) for v in values], row=row)
    set_array("L1_B", [int(v) for v in model["L1_B"]])
    set_array("L1_M0", [int(v) for v in model["L1_M0"]])
    set_array("L1_N", [int(v) for v in model["L1_N"]])
    set_row("L2_W", [int(v) for v in model["L2_W"]])
    set_array("L2_B", [int(v) for v in model["L2_B"]])
    # only a trained target supplies its own layer-2 scale; a random one keeps the header's
    if "L2_M0" in model:
        set_array("L2_M0", [int(v) for v in model["L2_M0"]])
        set_array("L2_N", [int(v) for v in model["L2_N"]])

    shutil.copy(CONFIG_HEADER, PRISTINE)
    progress("new network is now the pristine reference")
    return build()


def save_truth(model: dict, path: Path) -> None:
    """Save the flashed network plus a timestamp as ground truth for scoring."""
    path.write_text(json.dumps(
        {**model, "written": datetime.now().isoformat(timespec="seconds")}, indent=2))
