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


def _tvla_once(traces: np.ndarray, values: np.ndarray, min_class: int) -> np.ndarray:
    # Per-sample |Welch t| of each class of v against the rest, maxed over classes.
    # Each class's moments are accumulated once and the complement is derived from
    # the totals, so no class ever materialises a copy of the other traces.
    shift = traces[0].astype(np.float64)        # shifts cancel in both t terms
    counts, sums, squares = {}, {}, {}
    for value in np.unique(values):
        rows = traces[values == value].astype(np.float64) - shift
        counts[value] = len(rows)
        sums[value] = rows.sum(axis=0)
        squares[value] = np.einsum("ij,ij->j", rows, rows)

    total_n = sum(counts.values())
    total_sum = sum(sums.values())
    total_square = sum(squares.values())

    best = np.zeros(traces.shape[1])
    for value, n_a in counts.items():
        n_b = total_n - n_a
        if n_a < max(min_class, 2) or n_b < 2:
            continue
        mean_a = sums[value] / n_a
        mean_b = (total_sum - sums[value]) / n_b
        var_a = np.maximum(squares[value] - n_a * mean_a ** 2, 0.0) / (n_a - 1)
        var_b = np.maximum(total_square - squares[value] - n_b * mean_b ** 2, 0.0) / (n_b - 1)
        denom = np.sqrt(var_a / n_a + var_b / n_b)
        best = np.maximum(best, np.abs(mean_a - mean_b) / np.maximum(denom, 1e-12))
    return best


def tvla_profile(traces_by_capture, inputs_by_capture, flashed_weights,
                 hypotheses=None, min_class: int = 30,
                 scope: str = "pooled") -> np.ndarray:
    # Welch t over classes of v, as an alternative to the SNR of leak_profile:
    # TVLA asks "is this sample's mean different for some v", SNR asks "is it
    # graded in |v|".
    #
    # `scope` decides what the class is contrasted against, and it matters here.
    # "pooled" throws all captures together, which is the comparable grouping -
    # but a class like v = +-49 exists only in the capture holding w = 7, and
    # that capture differs from the others in EVERY neuron's weight, so a sample
    # carrying a neighbour's weight load separates it just as well as the
    # multiply does. "capture" runs the test inside each capture, where the
    # other weights are constant and cannot contribute, then averages the
    # per-capture profiles. The gap between the two is the size of that
    # confound, so both are reported rather than one being chosen here.
    values = [(weight * inputs).astype(np.int64)
              for inputs, weight in zip(inputs_by_capture, flashed_weights)]

    if scope == "pooled":
        best = _tvla_once(np.concatenate(traces_by_capture), np.concatenate(values), min_class)
    elif scope == "capture":
        per = [_tvla_once(t, v, min_class) for t, v in zip(traces_by_capture, values)]
        best = np.mean(per, axis=0)
    else:
        raise ValueError(f"unknown tvla scope {scope!r}, expected 'pooled' or 'capture'")

    if not best.any():
        raise ValueError("no class of v large enough for a t-test")
    return best


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


def extra_pois(profile: np.ndarray, primary: int, count: int, separation: int,
               window: int = 60) -> list[int]:
    # More points for one neuron, when a template uses several samples. They must
    # stay within `window` of the primary: every neuron leaks at every neuron's
    # point (they all read the same input), so the next highest peak in this
    # neuron's profile usually belongs to a NEIGHBOUR - measured, unconstrained
    # extra points land hundreds of samples away and cost a weight.
    chosen = [primary]
    work = np.full_like(profile, -np.inf)
    lo, hi = max(0, primary - window), min(len(profile), primary + window + 1)
    work[lo:hi] = profile[lo:hi]
    for t in chosen:
        work[max(0, t - separation):t + separation] = -np.inf
    while len(chosen) < count:
        nxt = int(work.argmax())
        if not np.isfinite(work[nxt]):     # nothing left above the blanked floor
            break
        chosen.append(nxt)
        work[max(0, nxt - separation):nxt + separation] = -np.inf
    return sorted(chosen)
