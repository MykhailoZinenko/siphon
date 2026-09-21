# Locating each neuron's multiply in the trace, via SNR over the true v = w*x.
from __future__ import annotations

import numpy as np


def leak_profile(traces_by_capture, inputs_by_capture, flashed_weights,
                 hypotheses=None, min_class: int = 30) -> np.ndarray:
    # Per-sample SNR over |v| computed WITHIN each sign of v = w*x; no model.
    # A leak that encodes only sign(v) has enormous SNR-over-signed-v (two tight,
    # far-apart clusters) but carries NO magnitude, so the old signed-v SNR chose
    # it and could not tell +/-2 from +/-5 when v was small. Grouping by |v|
    # inside each sign instead rewards points that are graded in magnitude, which
    # is what distinguishes the weight. Sign is still recovered from the templates.
    # `hypotheses` is accepted and ignored so callers need not special-case it.

    # within a capture every neuron's v is a fixed multiple of the same input,
    # so grouping traces by one neuron's v groups by all of them
    values = np.concatenate([(weight * inputs).astype(np.int64)
                             for inputs, weight in zip(inputs_by_capture, flashed_weights)])
    traces = np.concatenate(traces_by_capture)

    between = np.zeros(traces.shape[1])
    within = np.zeros(traces.shape[1])
    for sign in (1, -1):
        block = traces[np.sign(values) == sign]
        mags = np.abs(values[np.sign(values) == sign])
        present, counts = np.unique(mags, return_counts=True)
        usable = present[counts >= 2]                  # need >=2 traces for a variance
        if len(usable) < 3:                            # need >=3 magnitudes to see a gradient
            continue
        means = np.stack([block[mags == m].mean(0) for m in usable])       # E[trace | |v|]
        between += means.var(0)                        # spread of the per-magnitude means
        within += np.stack([block[mags == m].var(0) for m in usable]).mean(0)
    if not between.any():
        raise ValueError("not enough distinct magnitudes of v to locate the leak")
    return between / np.maximum(within, 1e-12)          # between-magnitude / within


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
