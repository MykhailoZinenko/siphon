# Templates and hypothesis scoring (textbook 4.3.2.3): measured leakage, no model.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RIDGE = 1e-9        # keeps the covariance invertible when a class is small


@dataclass
class Template:
    # The measured leakage distribution for one value of v: mean + covariance.
    mean: np.ndarray
    covariance: np.ndarray
    inverse: np.ndarray          # precomputed inverse covariance for scoring
    log_det: float               # precomputed log|covariance| for scoring


def build(values: np.ndarray, leakage: np.ndarray) -> dict[int, Template]:
    # One template per distinct v; leakage is (traces, pois).
    out: dict[int, Template] = {}
    width = leakage.shape[1]
    for value in np.unique(values):
        rows = leakage[values == value]          # traces whose v == value
        cov = np.atleast_2d(np.cov(rows, rowvar=False)) + RIDGE * np.eye(width)
        out[int(value)] = Template(
            mean=rows.mean(axis=0),
            covariance=cov,
            inverse=np.linalg.inv(cov),
            log_det=float(np.linalg.slogdet(cov)[1]),
        )
    return out


def coverage(templates: dict[int, Template], hypotheses, inputs) -> float:
    # Fraction of the v values the hypotheses need that actually have a template.
    needed = {int(w * x) for w in hypotheses for x in np.unique(inputs)}
    return len(needed & set(templates)) / len(needed)


def score(templates: dict[int, Template], leakage: np.ndarray, inputs: np.ndarray,
          hypotheses) -> dict[int, float]:
    # Eq. 4.33: summed log-likelihood of the attack traces under each weight guess.
    out: dict[int, float] = {}
    for w in hypotheses:
        values = (w * inputs).astype(np.int64)       # the v this weight would produce
        total = 0.0
        for j, value in enumerate(values):
            template = templates.get(int(value))
            if template is None:                     # a needed v has no template
                total = -np.inf                      # unusable, not merely unlikely
                break
            delta = leakage[j] - template.mean
            total -= template.log_det + delta @ template.inverse @ delta  # Mahalanobis
        out[int(w)] = total
    return out
