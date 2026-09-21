# Reading power traces (one sample per line), with a .npy cache per crop window.
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_traces(folder: Path, count: int, window: tuple[int, int],
                reporter=None) -> np.ndarray:
    # Load `count` traces cropped to `window`; parse text once, then hit the cache.
    lo, hi = window
    cache = Path(folder) / f"_cache_{count}_{lo}_{hi}.npy"    # key includes the crop
    if cache.exists():
        return np.load(cache)

    out = np.empty((count, hi - lo), dtype=np.float32)
    indices = range(count)
    if reporter is not None:
        indices = reporter.track(indices, f"parsing {Path(folder).name}", total=count)
    for i in indices:
        with open(Path(folder) / f"trace_{i}.txt") as fh:
            out[i] = np.array(fh.read().split(), dtype=np.float32)[lo:hi]   # crop on load
    np.save(cache, out)
    return out


def load_inputs(folder: Path, count: int, column: int) -> np.ndarray:
    # The input value at `column` for each trace (the hot-spot value).
    return np.loadtxt(Path(folder) / "inputs.txt")[:count, column]


def load_manifest(folder: Path) -> dict[int, list[int]]:
    # capture index -> the weight column that was flashed for it.
    out = {}
    for line in (Path(folder) / "manifest.txt").read_text().splitlines():
        parts = line.split()
        out[int(parts[0])] = [int(v) for v in parts[1:]]
    return out


def ground_truth(config_header: Path, column: int) -> list[int]:
    # Read L1_W[:][column] from network_config.h (scoring only, never the attack).
    import re
    text = Path(config_header).read_text()
    block = re.search(r"L1_W\[NET_L1\]\[NET_INPUTS\]\s*=\s*\{(.*?)\};", text, re.S).group(1)
    rows = re.findall(r"\{([^}]*)\}", block)             # one brace group per neuron
    return [int(r.split(",")[column]) for r in rows]


def ground_truth_l2(config_header: Path) -> list[int]:
    # Read L2_W[0] from network_config.h (scoring only).
    import re
    text = Path(config_header).read_text()
    row = re.search(r"L2_W\[NET_L2\]\[NET_L1\]\s*=\s*\{\s*\{([^}]*)\}", text, re.S)
    return [int(v) for v in row.group(1).split(",")]
