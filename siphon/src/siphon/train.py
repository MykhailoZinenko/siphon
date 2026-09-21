# Training a Phase-2 target that actually uses its hidden layer.
# Random/student nets saturate ~90% of activations, so 100% decision agreement is
# partly the target's doing. This trains on XOR of two input signs (no linear model
# solves it), directly in the firmware's integer forward model via a straight-through
# estimator, and emits the dict target.flash consumes. See notes/25-phase2.md.
from __future__ import annotations

import numpy as np

SHIFT = 4          # default L1_N, matches target.generate


def _quantise_scale(s: float) -> tuple[int, int]:
    # A float scale as gemmlowp does it: m0 in [2^30, 2^31), n absorbs the
    # magnitude, so s == m0 / 2^(31+n) and m0 fits the firmware's int32.
    n = -int(np.floor(np.log2(s))) - 1               # shift that normalises m0
    m0 = int(round(s * 2.0 ** (31 + n)))
    return m0, n


def nonlinear_task(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # `n` int8 input vectors and their XOR-of-signs labels; x2..x9 are distractors.
    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, (n, 10)).astype(np.int64)
    while (near := (np.abs(x[:, :2]) < 8).any(1)).any():   # keep x0,x1 off the axis
        x[near] = rng.integers(-128, 128, (near.sum(), 10))
    y = ((x[:, 0] >= 0) != (x[:, 1] >= 0)).astype(np.float64)   # XOR of the two signs
    return x, y


def _requant(acc: np.ndarray, m0: np.ndarray, n: np.ndarray) -> np.ndarray:
    # The firmware's (acc*m0 + half) >> (31+n), as integer arithmetic.
    shift = (31 + n).astype(np.int64)
    return (acc * m0.astype(np.int64) + (1 << (shift - 1))) >> shift


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train(seed: int = 0, steps: int = 8000, lr: float = 0.03,
          n_train: int = 4000, decay: float = 2e-3) -> dict:
    # Learn the task in the quantised forward model (STE), return a flashable dict.
    rng = np.random.default_rng(seed)
    x, y = nonlinear_task(n_train, seed)
    xf = x.astype(np.float64)

    # float weights; the two label-bearing inputs get a head start
    w1 = rng.normal(0, 1.0, (6, 10))
    w1[:, :2] += rng.normal(0, 3, (6, 2))
    b1 = np.zeros(6)
    s1 = np.full(6, 7.0 / np.abs(xf[:, :2] @ w1[:, :2].T).std())    # aim pre ~ 0..7
    w2 = rng.normal(0, 1.5, 6)
    b2 = 0.0
    s2 = 0.3

    def qw(w):
        return np.clip(np.round(w), -7, 7)           # weights quantised to -7..7

    m, v, t = {}, {}, 0                               # Adam state
    def adam(name, g):
        nonlocal t
        m[name] = 0.9 * m.get(name, 0) + 0.1 * g
        v[name] = 0.999 * v.get(name, 0) + 0.001 * g * g
        return lr * (m[name] / (1 - 0.9 ** t)) / (np.sqrt(v[name] / (1 - 0.999 ** t)) + 1e-8)

    for t in range(1, steps + 1):
        wq1, wq2 = qw(w1), qw(w2)                     # forward uses quantised weights
        acc1 = xf @ wq1.T + np.round(b1)             # layer-1 accumulator (N,6)
        pre1 = acc1 * s1
        h = np.clip(pre1, 0, 7)                       # ReLU6-style hidden
        hq = np.round(h)                             # quantised hidden 0..7
        acc2 = hq @ wq2 + np.round(b2)              # layer-2 accumulator
        logit = acc2 * s2
        p = _sigmoid(logit)

        dlogit = (p - y) / len(y)                     # BCE gradient
        ds2 = float((dlogit * acc2).sum())
        dacc2 = dlogit * s2
        dw2 = hq.T @ dacc2                            # STE through round + clip(-7,7)
        db2 = float(dacc2.sum())
        dhq = np.outer(dacc2, wq2)
        dpre1 = dhq * ((pre1 > 0) & (pre1 < 7))       # STE round; clip gates the gradient
        ds1 = (dpre1 * acc1).sum(0)
        dacc1 = dpre1 * s1
        dw1 = dacc1.T @ xf
        db1 = dacc1.sum(0)

        w1 = np.clip(w1 - adam("w1", dw1) - decay * w1, -7.5, 7.5)   # decay: distractors -> 0
        b1 -= adam("b1", db1)
        s1 = np.maximum(s1 - adam("s1", ds1), 1e-4)
        w2 = np.clip(w2 - adam("w2", dw2) - decay * w2, -7.5, 7.5)
        b2 -= adam("b2", db2)
        s2 = max(s2 - adam("s2", ds2), 1e-4)

    m0_1, n_1 = zip(*(_quantise_scale(v) for v in s1))   # per-neuron (m0, n)
    m0_2, n_2 = _quantise_scale(s2)
    return {
        "seed": seed, "task": "xor-of-signs",
        "L1_W": qw(w1).astype(int).tolist(),
        "L1_B": np.round(b1).astype(int).tolist(),
        "L1_M0": list(m0_1),
        "L1_N": list(n_1),
        "L2_W": qw(w2).astype(int).tolist(),
        "L2_B": [int(round(b2))],
        "L2_M0": [m0_2],
        "L2_N": [n_2],
        "inv_scale": [1.0 / v for v in s1],
    }


def forward(model: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Exact integer inference: returns (hidden 0..7 per neuron, logit).
    x = x.astype(np.int64)
    w1 = np.array(model["L1_W"]); b1 = np.array(model["L1_B"])
    m0 = np.array(model["L1_M0"]); n1 = np.array(model["L1_N"])
    w2 = np.array(model["L2_W"]); b2 = int(model["L2_B"][0])
    acc1 = x @ w1.T + b1
    h = np.clip(_requant(acc1, m0, n1), 0, 7).astype(np.int64)   # hidden activations
    logit = (h @ w2 + b2)
    return h, logit


def evaluate(model: dict, seed: int = 999, n: int = 20000) -> dict:
    # Task accuracy, hidden-layer usage, and the linear baseline it beats.
    x, y = nonlinear_task(n, seed)
    h, logit = forward(model, x)
    acc = float(((logit > 0) == (y > 0.5)).mean())
    hist = np.bincount(h.reshape(-1), minlength=8) / h.size      # activation histogram
    return {
        "task_accuracy": acc,
        "hidden_usage": float(1 - hist[0] - hist[7]),            # fraction off the 0/7 rails
        "activation_hist": hist.round(3).tolist(),
        "linear_accuracy": _linear_baseline(seed, n),            # XOR pins this at chance
    }


def task_report(truth: dict, recovered: dict, seed: int = 999, n: int = 20000) -> dict:
    # Recovered vs flashed network on the TASK's own inputs (the honest Phase-2 test).
    x, y = nonlinear_task(n, seed)
    _, true_logit = forward(truth, x)
    w1 = np.array(recovered["L1_W"]); b1 = np.array(recovered["L1_B"])
    inv = np.array(recovered["L1_inv_scale"]); w2 = np.array(recovered["L2_W"])
    b2 = int(recovered["L2_B"][0])
    h = np.stack([np.clip(np.floor((x @ w1[o] + b1[o]) / inv[o] + 0.5), 0, 7)   # recovered hidden
                  for o in range(6)], 1)
    rec_logit = h @ w2 + b2
    return {
        "true_task_accuracy": float(((true_logit > 0) == (y > 0.5)).mean()),
        "recovered_task_accuracy": float(((rec_logit > 0) == (y > 0.5)).mean()),
        "decision_agreement": float((np.sign(true_logit) == np.sign(rec_logit)).mean()),
        "samples": n,
    }


def _linear_baseline(seed: int, n: int) -> float:
    # Logistic regression on the raw inputs - the model without a hidden layer.
    x, y = nonlinear_task(n, seed)
    xf = np.hstack([x / 128.0, np.ones((n, 1))])     # inputs + bias column
    w = np.zeros(xf.shape[1])
    for _ in range(2000):
        w -= 0.5 * xf.T @ (_sigmoid(xf @ w) - y) / n # gradient descent on BCE
    return float(((xf @ w > 0) == (y > 0.5)).mean())
