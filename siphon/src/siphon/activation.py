# Recovering layer-1 bias b and requant scale s, by profiling on a clone.
# a = clamp(floor((acc + b) * s + 1/2), 0, 7), s = m0 / 2^(31+n); need b and s.
# Correlation on HW(acc+b) fails (known acc dominates); blind staircase fits are
# degenerate; so flash a clone with known (b,s) and score (b,s) by likelihood.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_ACT = 7


@dataclass
class NeuronFit:
    neuron: int
    pois: list[int]
    inv_scale: float
    bias: int
    agreement: float | None = None      # against a known reference, when there is one

    @property
    def scale(self) -> float:
        return 1.0 / self.inv_scale


def activations(acc: np.ndarray, bias: float, inv_scale: float) -> np.ndarray:
    # The firmware's hidden value: clamp(round((acc+b)/ (1/s)), 0, 7).
    return np.clip(np.floor((acc + bias) / inv_scale + 0.5), 0, MAX_ACT).astype(int)


def leak_points(traces: np.ndarray, labels: np.ndarray, acc: np.ndarray,
                allowed: np.ndarray, count: int = 20, separation: int = 2,
                max_trend: float = 0.02, min_eta: float = 0.10) -> list[int]:
    # Samples that leak the activation `a`, not the accumulator `acc`.
    # a is monotone in acc, so grouping by a cannot tell them apart; the residual
    # acc-trend WITHIN an activation class does (small where a leaks, large at acc).
    total = traces.var(0) * len(traces)
    within = np.zeros(traces.shape[1])
    trend = np.zeros(traces.shape[1])
    for value in range(MAX_ACT + 1):
        rows = labels == value
        if rows.sum() < 25:                      # skip thin activation classes
            continue
        block = traces[rows]
        centred = block - block.mean(0)
        within += (centred ** 2).sum(0)          # within-class variance
        x = acc[rows] - acc[rows].mean()
        trend += (x @ centred) ** 2 / max(x @ x, 1e-9)   # variance explained by acc

    eta = 1 - within / np.maximum(total, 1e-12)          # how much a explains
    residual_trend = trend / np.maximum(within, 1e-12)   # how much acc still explains
    ok = np.where((residual_trend < max_trend) & (eta > min_eta) & allowed)[0]

    chosen: list[int] = []
    for sample in ok[np.argsort(-eta[ok])]:              # strongest a-leak first
        if all(abs(sample - t) > separation for t in chosen):
            chosen.append(int(sample))
        if len(chosen) == count:
            break
    return chosen


def _distances(profile: np.ndarray, labels: np.ndarray, target: np.ndarray) -> np.ndarray:
    # Mahalanobis distance of each target trace to each activation's template,
    # under a POOLED covariance shared across classes. A per-class covariance
    # needs >width traces per class, which the extreme classes (a saturated at 7)
    # never have, so they used to score inf and disqualify the true (b, s) the
    # moment it routed any trace there. The pooled estimate uses all within-class
    # residuals, so every class is templated; its log-det is common to all
    # classes and drops out of the argmin, so it is omitted.
    width = profile.shape[1]
    out = np.full((len(target), MAX_ACT + 1), np.inf)
    means, residuals = {}, []
    for value in range(MAX_ACT + 1):
        rows = profile[labels == value]
        if len(rows) < 2:                        # need a mean and a residual
            continue
        means[value] = rows.mean(0)
        residuals.append(rows - means[value])
    if not residuals:
        return out
    pooled = np.cov(np.concatenate(residuals).T, ddof=1).reshape(width, width)
    inv = np.linalg.inv(pooled + 1e-9 * np.eye(width))
    for value, mean in means.items():
        delta = target - mean
        out[:, value] = np.einsum("ij,jk,ik->i", delta, inv, delta)
    return out


def fit_neuron(profile: np.ndarray, labels: np.ndarray, target: np.ndarray,
               acc: np.ndarray, inv_scales: np.ndarray,
               biases: np.ndarray) -> tuple[float, int]:
    # Max-likelihood (bias, 1/scale): for each candidate, map acc->a and sum the
    # precomputed template distances; pick the cheapest.
    d = _distances(profile, labels, target)
    best = (np.inf, None)
    for inv in inv_scales:
        grid = np.clip(np.floor((acc[None, :] + biases[:, None]) / inv + 0.5),
                       0, MAX_ACT).astype(int)          # predicted a per (bias, trace)
        cost = np.take_along_axis(d.T[None, :, :], grid[:, None, :], 1)[:, 0, :].sum(1)
        i = int(np.argmin(cost))
        if cost[i] < best[0]:
            best = (cost[i], (float(inv), int(biases[i])))
    return best[1]
