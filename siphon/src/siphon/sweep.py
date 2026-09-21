# Crafting inputs that drive the layer-1 accumulator (for bias/scale and layer 2).
# Weight-attack hot spots pin acc into -49..49 where a is always 0; bias and
# scale only show where a moves, so acc must be swept across a few hundred units.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

INT8_LO, INT8_HI = -128, 127


def recovered_weights(results: Path) -> np.ndarray:
    # Layer-1 matrix (neurons, inputs) read from a `siphon layer` results.json.
    data = json.loads(Path(results).read_text())
    columns = sorted(int(c) for c in data["weights"])
    return np.array([data["weights"][str(c)] for c in columns]).T


def craft(weights: np.ndarray, count: int, span: tuple[int, int] = (-60, 420),
          seed: int = 0, pool: int = 40) -> tuple[np.ndarray, np.ndarray]:
    # Inputs whose accumulators spread over `span`, plus the achieved accumulators.
    # W1 x = acc has a 4-D kernel, so any acc can be requested for all six neurons;
    # but rounding+int8 clipping throws acc outside the window, so oversample and
    # keep the vectors that land inside on EVERY neuron (also breaks acc correlation).
    lo, hi = span
    pinv = np.linalg.pinv(weights.astype(float))         # right-inverse of W1
    rng = np.random.default_rng(seed)
    targets = rng.uniform(lo, hi, (count * pool, weights.shape[0]))    # desired accs
    cand = np.clip(np.rint(targets @ pinv.T), INT8_LO, INT8_HI).astype(int)   # int8 inputs
    acc = cand @ weights.T                               # accs actually achieved
    excess = (np.maximum(lo - acc, 0) + np.maximum(acc - hi, 0)).sum(axis=1)  # out-of-window
    keep = np.argsort(excess, kind="stable")[:count]     # the best-behaved `count`
    keep.sort()
    return cand[keep], acc[keep]


def coverage(accs: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> np.ndarray:
    # Activation histogram per neuron - a check, needs the truth so not part of the attack.
    out = np.zeros((accs.shape[1], 8), dtype=int)
    for o in range(accs.shape[1]):
        a = np.clip(np.floor((accs[:, o] + bias[o]) * scale[o] + 0.5), 0, 7).astype(int)
        out[o] = np.bincount(a, minlength=8)
    return out


def hidden_hotspot(weights: np.ndarray, params: list[tuple[float, int]], count: int,
                   levels: int = 4, seed: int = 0, tries: int = 400) -> tuple[np.ndarray, np.ndarray]:
    # Inputs that leave exactly one hidden activation non-zero (the layer-2 hot spot).
    # acc2 is a running sum over six activations; zeroing five collapses it to one
    # product w2[j]*a[j]. Uses recovered (1/s, b) per neuron, not the truth.
    first = np.array([0.5 * inv - b for inv, b in params])       # acc where a steps off 0
    pinv = np.linalg.pinv(weights.astype(float))
    rng = np.random.default_rng(seed)
    neurons = weights.shape[0]

    def activation(x):                                   # a per neuron for input x
        acc = x @ weights.T
        return np.array([min(7, max(0, int(np.floor((acc[o] + b) / inv + 0.5))))
                         for o, (inv, b) in enumerate(params)])

    pool: dict[tuple[int, int], list[np.ndarray]] = {}   # (neuron, level) -> inputs achieving it
    for j in range(neurons):
        for level in range(1, levels + 1):
            want = np.array([first[i] - 140.0 if i != j else level * params[j][0] - params[j][1]
                             for i in range(neurons)])    # push others below 0, j to `level`
            for _ in range(tries):
                x = np.clip(np.rint(pinv @ (want + rng.normal(0, 6, neurons))),
                            INT8_LO, INT8_HI).astype(int)
                a = activation(x)
                if a[j] == level and not a[np.arange(neurons) != j].any():   # only j non-zero
                    pool.setdefault((j, level), []).append(x)
    if not pool:
        raise RuntimeError("no hidden hot-spot is reachable with these parameters")

    keys = sorted(pool)                                  # cycle through reachable (neuron, level)
    vectors, tags = [], []
    for i in range(count):
        key = keys[i % len(keys)]
        options = pool[key]
        vectors.append(options[(i // len(keys)) % len(options)])
        tags.append(key)
    return np.array(vectors), np.array(tags)
