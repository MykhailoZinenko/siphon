# Locating each neuron's multiply in the trace, via SNR over the true v = w*x.
from __future__ import annotations

import numpy as np


def leak_profile(traces_by_capture, inputs_by_capture, flashed_weights,
                 hypotheses=None, min_class: int = 30) -> np.ndarray:
    # Per-sample SNR of the trace grouped by v = w*x; no leakage model.
    # SNR = Var_v(E[trace|v]) / E_v(Var[trace|v]); v is known exactly here.
    # (HW model was blind to doubling, HW(2x)=HW(x), and picked wrong points.)
    # `hypotheses` is accepted and ignored so callers need not special-case it.

    # within a capture every neuron's v is a fixed multiple of the same input,
    # so grouping traces by one neuron's v groups by all of them
    values = np.concatenate([(weight * inputs).astype(np.int64)
                             for inputs, weight in zip(inputs_by_capture, flashed_weights)])
    traces = np.concatenate(traces_by_capture)

    present, counts = np.unique(values, return_counts=True)
    usable = present[counts >= min_class]              # drop v with too few traces
    if len(usable) < 2:
        raise ValueError("not enough distinct values of v to locate the leak")

    means = np.stack([traces[values == v].mean(0) for v in usable])    # E[trace|v]
    noise = np.stack([traces[values == v].var(0) for v in usable]).mean(0)  # E_v Var[trace|v]
    return means.var(0) / np.maximum(noise, 1e-12)     # between-class / within-class


def assign_pois(profiles: dict[int, np.ndarray], separation: int) -> list[int]:
    # One POI per neuron: greedily take each neuron's SNR peak, but keep peaks
    # `separation` apart (one input is read by every neuron, so all six leak).
    taken: list[int] = []
    for neuron in sorted(profiles, key=lambda n: -profiles[n].max()):   # strongest first
        candidate = profiles[neuron].copy()
        for t in taken:
            candidate[max(0, t - separation):t + separation] = -np.inf  # blank taken regions
        taken.append(int(candidate.argmax()))
    return sorted(taken)          # sorted = time order = firmware's neuron order


def extra_pois(profile: np.ndarray, primary: int, count: int, separation: int) -> list[int]:
    # More points for one neuron, when a template uses several samples.
    chosen = [primary]
    work = profile.copy()
    for t in chosen:
        work[max(0, t - separation):t + separation] = -np.inf
    while len(chosen) < count:
        nxt = int(work.argmax())
        if not np.isfinite(work[nxt]):     # nothing left above the blanked floor
            break
        chosen.append(nxt)
        work[max(0, nxt - separation):nxt + separation] = -np.inf
    return sorted(chosen)
