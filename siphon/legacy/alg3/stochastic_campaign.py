"""
Algorithm 3c — stochastic model fitted the way the book intends.

Uses the multi-weight campaign: L1_W[0][1] was reflashed to w = 1..7 and traces
captured at each, all with the same build. Every (v, leakage) pair therefore
comes from the SAME instruction, which is what algorithm 3b lacked.

Two things are done carefully:

1. Locating neuron 0's multiply. Its POI cannot be identified by "which weight
   comes out", because the weight changes between captures. Instead we look for
   the sample where the recovered weight TRACKS the flashed weight across all
   seven captures — a far stronger test, and one only the true point passes.

2. Bit width. v = w*x reaches only +/-49, so in 8-bit two's complement bits 6
   and 7 are always equal (both encode the sign) and the design matrix can never
   exceed rank 7. A 7-bit representation covers -64..63 without that forced
   collinearity.
"""
import os
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import (  # noqa: E402
    DATA, FIRMWARE, STUDENT_ATTACK, STUDENT_PROFILING, REBUILT_ATTACK,
    CAMPAIGN_WEIGHT, CAMPAIGN_COLUMN,
)
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profiled_dpa import (          # noqa: E402
    correlate_at, hw_float32, GRAPHS, INPUT_COLUMN,
    HYPOTHESES, WINDOW, MAX_TRACES_CURVE,
)
from nonprofiled_dpa import correlate          # noqa: E402
from stochastic import traces_until_stable     # noqa: E402

# each legacy folder writes its own figures, not the one it imports helpers from
GRAPHS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = [1, 2, 3, 4, 5, 6, 7]
PROFILING_COUNT = 1000
ATTACK_DIR = REBUILT_ATTACK
ATTACK_COUNT = 5000
TRUE_WEIGHT = 4
BITS = 7            # |v| <= 49, so 7 bits suffice and bits 6,7 are not forced equal


def load_dir(folder, count):
    cache = os.path.join(folder, f"_cache_{count}_{WINDOW[0]}_{WINDOW[1]}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    lo, hi = WINDOW
    out = np.empty((count, hi - lo), dtype=np.float32)
    for i in range(count):
        with open(os.path.join(folder, f"trace_{i}.txt")) as fh:
            out[i] = np.array(fh.read().split(), dtype=np.float32)[lo:hi]
    np.save(cache, out)
    return out


def bits_of(values, bits=BITS):
    u = np.asarray(values, dtype=np.int64) & ((1 << bits) - 1)
    return np.stack([(u >> s) & 1 for s in range(bits)], axis=1).astype(np.float64)


def main():
    print("[*] loading campaign captures ...")
    sets = {}
    for w in WEIGHTS:
        folder = os.path.join(CAMPAIGN_WEIGHT, f"w{w}")
        T = load_dir(folder, PROFILING_COUNT)
        x = np.loadtxt(os.path.join(folder, "inputs.txt"))[:PROFILING_COUNT, INPUT_COLUMN]
        sets[w] = (T, x)
        print(f"    w={w}: {T.shape}")

    # --- locate neuron 0 by weight tracking ---------------------------------
    print("[*] locating neuron 0's multiply (weight must track the flashed value) ...")
    hits = np.zeros(WINDOW[1] - WINDOW[0], dtype=int)
    for w, (T, x) in sets.items():
        v = (HYPOTHESES[:, None] * x[None, :]).astype(np.int64)
        H = np.stack([hw_float32(r) for r in v])
        R = np.abs(correlate(H, T))
        hits += (HYPOTHESES[R.argmax(axis=0)] == w).astype(int)

    best = int(hits.argmax())
    print(f"    samples where the recovered weight matched all {len(WEIGHTS)} captures: "
          f"{int((hits == len(WEIGHTS)).sum())}")
    print(f"    chosen POI = {best}  (matched {hits[best]}/{len(WEIGHTS)} captures)")
    if hits[best] < len(WEIGHTS):
        print("    !! no sample tracks every weight — treat the result with suspicion")

    # --- pool and fit --------------------------------------------------------
    V, L = [], []
    for w, (T, x) in sets.items():
        col = T[:, best].astype(np.float64)
        V.append((w * x).astype(np.int64))
        L.append((col - col.mean()) / col.std())
    v_pool, l_pool = np.concatenate(V), np.concatenate(L)
    print(f"[*] pooled {len(v_pool)} pairs, {len(np.unique(v_pool))} distinct values of v"
          f" (range {v_pool.min()}..{v_pool.max()})")

    M = bits_of(v_pool)
    rank = np.linalg.matrix_rank(M)
    print(f"    rank(M_v) = {rank} of {BITS}"
          f"{'  FULL RANK' if rank == BITS else '  still deficient'}")
    alpha, *_ = np.linalg.lstsq(M, l_pool, rcond=None)
    print("    a_s = " + "  ".join(f"{a:+.3f}" for a in alpha))

    # --- attack --------------------------------------------------------------
    print("[*] attacking ...")
    atk = load_dir(ATTACK_DIR, ATTACK_COUNT)
    x_atk = np.loadtxt(os.path.join(ATTACK_DIR, "inputs.txt"))[:ATTACK_COUNT, INPUT_COLUMN]
    v_hat = (HYPOTHESES[:, None] * x_atk[None, :]).astype(np.int64)

    H_stoch = np.stack([bits_of(r) @ alpha for r in v_hat])
    H_hw = np.stack([hw_float32(r) for r in v_hat])

    def cross(H):
        C = np.corrcoef(H); np.fill_diagonal(C, np.nan); return np.nanmean(np.abs(C))

    print(f"    mean |corr| between hypotheses: stochastic {cross(H_stoch):.3f} "
          f"vs HW {cross(H_hw):.3f}")

    column = atk[:, best].astype(np.float64)
    counts = np.arange(2, MAX_TRACES_CURVE + 1)
    true_idx = int(np.where(HYPOTHESES == TRUE_WEIGHT)[0][0])

    results = {}
    for name, H in (("stochastic", H_stoch), ("Hamming weight", H_hw)):
        curves = np.zeros((len(HYPOTHESES), len(counts)))
        for j, m in enumerate(counts):
            curves[:, j] = correlate_at(H[:, :m], column[:m])
        full = correlate_at(H, column)
        order = np.argsort(-np.abs(full))
        rec, margin = HYPOTHESES[order[0]], abs(full[order[0]]) - abs(full[order[1]])
        stable = traces_until_stable(curves, counts, true_idx)
        results[name] = (curves, stable)
        flag = "OK" if rec == TRUE_WEIGHT else "WRONG"
        print(f"    {name:<16} recovered {rec:+d} [{flag}]  margin {margin:.4f}  "
              f"stable from {stable if stable else '-'}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, (name, (curves, stable)) in zip(axes, results.items()):
        for i, w in enumerate(HYPOTHESES):
            style = dict(color="blue", lw=2.2, zorder=10, label=f"true w={w}") if w == TRUE_WEIGHT \
                else dict(color="gray", alpha=0.45, lw=0.9)
            ax.plot(counts, curves[i], **style)
        if stable:
            ax.axvline(stable, color="green", ls="--", lw=1.2, label=f"stable from {stable}")
        ax.set_title(name, fontweight="bold"); ax.set_xlabel("number of traces")
        ax.grid(alpha=0.3); ax.legend()
    axes[0].set_ylabel("r")
    fig.suptitle(f"Algorithm 3c — stochastic model from the multi-weight campaign, POI={best}",
                 fontweight="bold")
    out = os.path.join(GRAPHS, "alg3c_campaign.png")
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"    graph -> {out}")


if __name__ == "__main__":
    main()
