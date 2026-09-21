# Leakage models used ONLY to locate points of interest (the template attack needs none).
from __future__ import annotations

import struct

import numpy as np

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def hamming_float32(values: np.ndarray) -> np.ndarray:
    # Hamming weight of the IEEE-754 bits of v: not physical, but strongly
    # nonlinear, so it separates weight hypotheses that are otherwise scalar
    # multiples of one another.
    return np.array(
        [bin(struct.unpack("!I", struct.pack("!f", float(v)))[0]).count("1") for v in values],
        dtype=np.float64,
    )


def correlate(hypotheses: np.ndarray, traces: np.ndarray) -> np.ndarray:
    # Pearson correlation of every hypothesis row against every trace column.
    H = hypotheses - hypotheses.mean(axis=1, keepdims=True)
    T = traces - traces.mean(axis=0, keepdims=True)
    num = H @ T
    den = np.sqrt((H ** 2).sum(1)[:, None] * (T ** 2).sum(0)[None, :])
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den > 0)      # guard columns with zero variance
    return out
