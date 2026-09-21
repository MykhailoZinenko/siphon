# Layer-2 weights: same template attack, hot spot now on the hidden activations.
# Drive five of six activations to zero (the clamp allows it) so acc2 collapses
# to one product w2[j]*a[j]. Correlation fails: HW(2x)=HW(x) makes +3 and +6
# collinear; templates keyed on v separate them.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .templates import build, coverage, score


@dataclass
class InputResult:
    index: int
    pois: list[int]
    truth: int | None
    recovered: int
    margin: float
    templates: int
    coverage: float

    @property
    def correct(self) -> bool | None:
        return None if self.truth is None else self.recovered == self.truth


def usable_captures(traces: dict[int, np.ndarray], tolerance: float = 5.0) -> list[int]:
    # Drop glitched captures: mean trace far from the campaign median (one bad
    # capture otherwise dominates every cross-capture statistic).
    keys = sorted(traces)
    means = np.stack([traces[k].mean(0) for k in keys])
    deviation = np.abs(means - np.median(means, 0)).mean(1)
    limit = tolerance * np.median(deviation)
    return [k for k, d in zip(keys, deviation) if d <= limit]


def leak_points(traces: np.ndarray, values: np.ndarray, weights: np.ndarray,
                count: int = 6, separation: int = 8, max_weight_trend: float = 0.05,
                min_eta: float = 0.5) -> list[int]:
    # Samples leaking v = w2*a, not the weight itself (the loop loads every
    # L2_W[i] even at zero activation, so load points carry the weight and smear
    # the v-classes that pool several (weight, activation) pairs).
    total = traces.var(0) * len(traces)
    within = np.zeros(traces.shape[1])
    by_weight = np.zeros(traces.shape[1])
    for value in np.unique(values):
        rows = values == value
        if rows.sum() < 8:
            continue
        block = traces[rows]
        mean = block.mean(0)
        within += ((block - mean) ** 2).sum(0)         # spread within one v
        for w in np.unique(weights[rows]):
            sub = block[weights[rows] == w]
            if len(sub) < 4:
                continue
            by_weight += len(sub) * (sub.mean(0) - mean) ** 2   # spread due to weight
    eta = 1 - within / np.maximum(total, 1e-12)                 # how much v explains
    trend = by_weight / np.maximum(within, 1e-12)              # how much weight leaks
    ok = np.where((eta > min_eta) & (trend < max_weight_trend))[0]
    if not len(ok):
        ok = np.array([int(np.argmax(eta))])           # fall back to the strongest v-leak

    chosen: list[int] = []
    for sample in ok[np.argsort(-eta[ok])]:
        if all(abs(int(sample) - t) > separation for t in chosen):
            chosen.append(int(sample))
        if len(chosen) == count:
            break
    return chosen


def recover(profiling: dict[int, np.ndarray], attack: np.ndarray, tags: np.ndarray,
            manifest: dict[int, list[int]], hypotheses, inputs: int = 6,
            truth: list[int] | None = None) -> tuple[list[InputResult], list[int]]:
    # Recover each layer-2 weight from its own hot-spot traces; returns kept captures.
    keep = usable_captures(profiling)
    results = []
    for j in range(inputs):
        rows = tags[:, 0] == j                          # traces where neuron j is the hot one
        if not rows.any():                              # dead neuron: activation is always 0,
            results.append(InputResult(                 #   so it can never be isolated and its
                index=j, pois=[], truth=None if truth is None else truth[j],
                recovered=0, margin=0.0, templates=0,   #   layer-2 weight is moot (a*w2 = 0)
                coverage=0.0))
            continue
        levels = tags[rows, 1]                           # its activation value a
        values = np.concatenate([manifest[k][j] * levels for k in keep]).astype(np.int64)  # v=w2*a
        weights = np.concatenate([np.full(rows.sum(), manifest[k][j]) for k in keep])
        traces = np.concatenate([profiling[k][rows] for k in keep]).astype(np.float64)

        pois = leak_points(traces, values, weights)
        templates = build(values, traces[:, pois])
        scored = score(templates, attack[rows][:, pois].astype(np.float64),
                       levels, hypotheses)
        ranked = sorted(scored.items(), key=lambda kv: -kv[1])
        results.append(InputResult(
            index=j, pois=pois, truth=None if truth is None else truth[j],
            recovered=ranked[0][0], margin=ranked[0][1] - ranked[1][1],
            templates=len(templates),
            coverage=coverage(templates, hypotheses, levels),
        ))
    return results, keep


def bias_points(traces: np.ndarray, values: np.ndarray, flashed: np.ndarray,
                count: int = 4, separation: int = 8, max_bias_trend: float = 0.05,
                min_eta: float = 0.5) -> list[int]:
    # Samples leaking acc2 + b2 without leaking which bias was flashed.
    total = traces.var(0) * len(traces)
    within = np.zeros(traces.shape[1])
    by_bias = np.zeros(traces.shape[1])
    for value in np.unique(values):
        rows = values == value
        if rows.sum() < 12:
            continue
        block = traces[rows]
        mean = block.mean(0)
        within += ((block - mean) ** 2).sum(0)
        for b in np.unique(flashed[rows]):
            sub = block[flashed[rows] == b]
            if len(sub) < 5:
                continue
            by_bias += len(sub) * (sub.mean(0) - mean) ** 2   # spread due to flashed bias
    eta = 1 - within / np.maximum(total, 1e-12)
    trend = by_bias / np.maximum(within, 1e-12)
    ok = np.where((eta > min_eta) & (trend < max_bias_trend))[0]
    if not len(ok):
        ok = np.array([int(np.argmax(eta))])
    chosen: list[int] = []
    for sample in ok[np.argsort(-eta[ok])]:
        if all(abs(int(sample) - t) > separation for t in chosen):
            chosen.append(int(sample))
        if len(chosen) == count:
            break
    return chosen


def recover_bias(profiling: dict[int, np.ndarray], flashed: dict[int, int],
                 attack: np.ndarray, acc2: np.ndarray,
                 hypotheses=range(-40, 41), min_class: int = 12) -> tuple[int, float]:
    # Layer-2 bias by likelihood over templates of acc2 + b2. One profiling bias
    # is not enough (a template exists only for values actually produced), so the
    # campaign flashes a spread of biases to tile the range.
    ids = sorted(profiling)
    values = np.concatenate([acc2 + flashed[i] for i in ids])
    flashed_col = np.concatenate([np.full(len(acc2), flashed[i]) for i in ids])
    traces = np.concatenate([profiling[i] for i in ids]).astype(np.float64)

    pois = bias_points(traces, values, flashed_col)
    counts = {int(v): int((values == v).sum()) for v in np.unique(values)}
    usable = {v for v, n in counts.items() if n >= min_class}       # drop thin classes
    keep = np.isin(values, list(usable))
    templates = build(values[keep], traces[np.ix_(np.where(keep)[0], pois)])

    query = attack[:, pois].astype(np.float64)
    best = (None, -np.inf)
    for h in hypotheses:
        wanted = acc2 + h                                # values this bias would produce
        usable = np.isin(wanted, list(templates))
        # a hypothesis may not cover every trace (thin classes dropped); score
        # per-trace and require broad coverage, or one would win by covering less
        if usable.mean() < 0.9:
            continue
        total, used = 0.0, 0
        for value in np.unique(wanted[usable]):
            rows = wanted == value
            t = templates[int(value)]
            delta = query[rows] - t.mean
            total -= (np.einsum("ij,jk,ik->i", delta, t.inverse, delta) + t.log_det).sum()
            used += int(rows.sum())
        mean_cost = total / max(used, 1)                 # per-trace so coverage is fair
        if mean_cost > best[1]:
            best = (int(h), mean_cost)
    return best[0], best[1]
