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


def hamming_int32(values: np.ndarray) -> np.ndarray:
    # Hamming weight of v as the accumulator holds it: two's-complement int32.
    # Note the asymmetry this brings: HW(2v) = HW(v) for v > 0, but a negative v
    # loses a one bit on every doubling (HW(-3) = 31, HW(-6) = 30), because the
    # high bits are ones and the shift drops one of them.
    bits = np.asarray(values, dtype=np.int64) & 0xFFFFFFFF
    return sum(_POPCOUNT[(bits >> shift) & 0xFF] for shift in (0, 8, 16, 24))


def template_key(mode: str):
    # What a template is indexed by. "v" is the measured-leakage attack of
    # 4.3.2.3; "hw" collapses v into HW(v), which merges values differing by a
    # factor of two and so cannot separate weights like +3 and +6 - but only
    # where v > 0 (see hamming_int32). With hot-spot inputs of both signs the
    # pair is therefore not exactly tied: it is separated by the negative half
    # alone, which is weaker evidence, not none. Kept selectable to measure the
    # cost of the HW model instead of asserting it.
    if mode == "v":
        return lambda values: np.asarray(values, dtype=np.int64)
    if mode == "hw":
        return hamming_int32
    raise ValueError(f"unknown template key {mode!r}, expected 'v' or 'hw'")


def correlate(hypotheses: np.ndarray, traces: np.ndarray) -> np.ndarray:
    # Pearson correlation of every hypothesis row against every trace column.
    H = hypotheses - hypotheses.mean(axis=1, keepdims=True)
    T = traces - traces.mean(axis=0, keepdims=True)
    num = H @ T
    den = np.sqrt((H ** 2).sum(1)[:, None] * (T ** 2).sum(0)[None, :])
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den > 0)      # guard columns with zero variance
    return out
