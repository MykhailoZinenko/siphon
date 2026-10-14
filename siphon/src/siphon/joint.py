# Recovering (weight, bias) together: templates keyed on the accumulator.
#
# The weight attack keys templates on v = w*x, which assumes the profiling
# device adds the same bias as the victim - measured on 14.10.2026, a clone's
# bias alone drops layer 1 to 1/6 (notes/27). The accumulator actually holds
# acc = w*x + b, so key the templates on acc, sweep the bias during profiling,
# and let the hypothesis be the pair. Nothing of the victim's is then assumed.
#
# Identifiability: a wrong bias shifts the whole acc sequence by a constant, a
# wrong weight changes its slope in x. Sweeping x over both signs separates
# them - unlike |r| on v = w*x, which is scale invariant and degenerate.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .templates import RIDGE


@dataclass
class JointResult:
    neuron: int
    poi: int
    weight: int
    bias: int
    margin: float            # to the best hypothesis with a different weight
    bias_margin: float       # to the best hypothesis with the same weight
    covered: float           # fraction of hypotheses that had every template


def accumulators(inputs_by_capture, weights, biases) -> list[np.ndarray]:
    """acc = w*x + b per trace, per capture - known exactly while profiling."""
    return [(w * x + b).astype(np.int64)
            for x, w, b in zip(inputs_by_capture, weights, biases)]


def _snr_over_magnitude(traces: np.ndarray, values: np.ndarray,
                        min_class: int) -> np.ndarray:
    """Between-magnitude over within-magnitude variance, per sample, within each sign."""
    between = np.zeros(traces.shape[1])
    within = np.zeros(traces.shape[1])
    for sign in (1, -1):
        block = traces[np.sign(values) == sign]
        mags = np.abs(values[np.sign(values) == sign])
        present, counts = np.unique(mags, return_counts=True)
        usable = present[counts >= max(min_class, 2)]
        if len(usable) < 3:
            continue
        means = np.stack([block[mags == m].mean(0) for m in usable])
        between += means.var(0)
        within += np.stack([block[mags == m].var(0) for m in usable]).mean(0)
    if not between.any():
        return np.zeros(traces.shape[1])
    return between / np.maximum(within, 1e-12)


def acc_profile(traces_by_capture, values_by_capture, min_class: int = 20) -> np.ndarray:
    """Where the accumulator leaks: graded in acc inside a flash AND the same
    across flashes. The geometric mean of the two SNRs, because each alone is
    fooled in a different direction.

    Within one capture the bias is constant, so acc = w*x + b and v = w*x are
    perfectly correlated and the within-capture SNR cannot tell the accumulate
    from the multiply - measured on the board, the multiply point scored 443.9
    against the accumulator's 136.5, was chosen, and the attack returned 0/6.
    Pooling over acc across the flashed biases is what asks the real question
    ("is this sample a function of acc?"): the same multiply point scores 0.29
    there against the accumulator's 183.0, because one acc value arrives from
    many different products. Pooled alone would in turn reward a sample that
    merely identifies the flash, so both factors stay.
    """
    traces = np.concatenate(traces_by_capture)
    values = np.concatenate(values_by_capture)
    pooled = _snr_over_magnitude(traces, values, min_class)

    per = [_snr_over_magnitude(t, v, min_class=2)
           for t, v in zip(traces_by_capture, values_by_capture)]
    per = [p for p in per if p.any()]
    if not per or not pooled.any():
        raise ValueError("not enough distinct accumulator magnitudes to locate the leak")
    return np.sqrt(pooled * np.mean(per, axis=0))


def build(values: np.ndarray, leakage: np.ndarray, min_class: int = 8):
    """One Gaussian per accumulator value; thin classes are dropped, not trusted."""
    out = {}
    for value in np.unique(values):
        rows = leakage[values == value]
        if len(rows) < min_class:
            continue
        var = float(rows.var(ddof=1)) + RIDGE
        out[int(value)] = (float(rows.mean()), var, float(np.log(var)))
    return out


def score(templates, leakage: np.ndarray, inputs: np.ndarray,
          weights, biases, min_coverage: float = 0.9) -> dict[tuple[int, int], float]:
    """Log-likelihood of every (w, b), eq. 4.33 with a two-dimensional hypothesis.

    Only 14 input values exist, so the per-trace distances collapse into a table
    of (input value x accumulator class) and each hypothesis is 14 lookups.

    A swept bias cannot cover every accumulator value a hypothesis asks for, and
    demanding all of them silently eliminates the truth (measured: one missing
    class dropped the true pair to -inf while a wrong one won). So a hypothesis
    is kept when it covers `min_coverage` of the traces, and scored per covered
    trace - otherwise a guess could win by predicting fewer of them. Same rule
    as the layer-2 bias attack.
    """
    levels = np.unique(inputs)
    counts = np.array([int((inputs == level).sum()) for level in levels])
    classes = np.array(sorted(templates))
    index = {c: i for i, c in enumerate(classes)}
    means = np.array([templates[c][0] for c in classes])
    inverse = 1.0 / np.array([templates[c][1] for c in classes])
    log_det = np.array([templates[c][2] for c in classes])

    # distance of every trace to every class, summed per input level
    table = np.zeros((len(levels), len(classes)))
    for i, level in enumerate(levels):
        rows = leakage[inputs == level]
        delta = rows[:, None] - means[None, :]
        table[i] = (delta ** 2 * inverse[None, :] + log_det[None, :]).sum(axis=0)

    out = {}
    for w in weights:
        for b in biases:
            needed = (w * levels + b).astype(np.int64)
            covered = np.array([int(v) in index for v in needed])
            used = int(counts[covered].sum())
            if used < min_coverage * counts.sum():
                continue                      # too little of the attack set explained
            total = sum(table[i, index[int(v)]]
                        for i, v in enumerate(needed) if covered[i])
            out[(int(w), int(b))] = -total / used      # per trace, so coverage is fair
    return out


def recover(templates_by_neuron, leakage_by_neuron, pois, inputs,
            weights, biases) -> list[JointResult]:
    """Best (w, b) per neuron, with both margins reported separately."""
    results = []
    for neuron, (templates, leakage) in enumerate(zip(templates_by_neuron,
                                                      leakage_by_neuron)):
        scores = score(templates, leakage, inputs, weights, biases)
        if not scores:
            raise ValueError(f"neuron {neuron}: no hypothesis had full template "
                             f"coverage; widen the profiled bias range")
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        (best_w, best_b), best = ranked[0]
        other_weight = next((s for (w, _), s in ranked if w != best_w), -np.inf)
        same_weight = next((s for (w, b), s in ranked
                            if w == best_w and b != best_b), -np.inf)
        results.append(JointResult(
            neuron=neuron, poi=pois[neuron], weight=best_w, bias=best_b,
            margin=best - other_weight, bias_margin=best - same_weight,
            covered=len(scores) / (len(weights) * len(biases)),
        ))
    return results
