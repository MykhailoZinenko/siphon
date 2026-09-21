"""
Algorithm 3b — stochastic model fitted across SEVERAL POIs.

Algorithm 3 failed because with w = 4 fixed, v = 4x only ever takes multiples of
four: bits 0 and 1 never vary, bits 5-7 only encode the sign, and rank(M_v) = 4
of 8. The book's precondition ("all possible values of v appear") is
unsatisfiable with a single fixed weight.

The fix needs no reflashing. Because the hot-spot input is read by every neuron,
the SAME traces contain several different weights, each leaking at its own POI:

    POI 4237 -> neuron 0, w = +4     v = 4x   (multiples of 4)
    POI 3746 -> neuron 1, w = -7     v = -7x  (odd values! revives bits 0,1)
    POI 4857 -> neuron 4, w = -4     v = -4x

Pooling (v, leakage) pairs across those POIs widens the value coverage enough to
make M_v full rank.

Leakage is standardised per POI before pooling: different time samples have
different DC offset and scale, and we only care about the shape of the bit
dependence, which is assumed common to the same instruction.
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
    load, correlate_at, hw_float32,
    PROFILING, ATTACK, GRAPHS, M_PROFILING, M_ATTACK,
    INPUT_COLUMN, TRUE_WEIGHT, HYPOTHESES, MAX_TRACES_CURVE,
)
from stochastic import bit_matrix, traces_until_stable   # noqa: E402

# each legacy folder writes its own figures, not the one it imports helpers from
GRAPHS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")

# POIs confirmed in alg2b, with the weight each one leaks.
POI_WEIGHTS = {4237: 4, 3746: -7, 4857: -4}
ATTACK_POI = 4237          # neuron 0 — the weight we actually want


def main():
    print("[*] profiling ...")
    prof = load(PROFILING, M_PROFILING)
    x = np.loadtxt(os.path.join(PROFILING, "inputs.txt"))[:M_PROFILING, INPUT_COLUMN]

    # --- pool (v, leakage) across POIs --------------------------------------
    V, L = [], []
    for poi, w in POI_WEIGHTS.items():
        col = prof[:, poi].astype(np.float64)
        col = (col - col.mean()) / col.std()        # standardise per POI
        V.append((w * x).astype(np.int64))
        L.append(col)
        vals = np.unique(w * x)
        print(f"    POI {poi} (w={w:+d}): v in [{vals.min():.0f}, {vals.max():.0f}], "
              f"{len(vals)} distinct")

    v_pool = np.concatenate(V)
    l_pool = np.concatenate(L)
    print(f"    pooled: {len(v_pool)} samples, "
          f"{len(np.unique(v_pool))} distinct values of v")

    # --- fit ----------------------------------------------------------------
    M_v = bit_matrix(v_pool)
    dead = [i for i in range(M_v.shape[1]) if M_v[:, i].std() == 0]
    rank = np.linalg.matrix_rank(M_v)
    print(f"    rank(M_v) = {rank} of {M_v.shape[1]}"
          f"{'  (FULL RANK)' if rank == M_v.shape[1] else '  still deficient'}"
          f"{'; dead bits ' + str(dead) if dead else ''}")

    alpha, *_ = np.linalg.lstsq(M_v, l_pool, rcond=None)
    print("    coefficients a_s (LSB first):")
    print("      " + "  ".join(f"a{i}={a:+.3f}" for i, a in enumerate(alpha)))

    # How far is this from Hamming weight, which assumes every a_s equal?
    spread = alpha.std() / (np.abs(alpha).mean() + 1e-12)
    print(f"    relative spread of coefficients: {spread:.2f} "
          f"({'far from' if spread > 0.3 else 'close to'} equal-weight HW)")

    # --- attack --------------------------------------------------------------
    print("[*] attacking at POI", ATTACK_POI)
    atk = load(ATTACK, M_ATTACK)
    x_atk = np.loadtxt(os.path.join(ATTACK, "inputs.txt"))[:M_ATTACK, INPUT_COLUMN]
    v_hat = (HYPOTHESES[:, None] * x_atk[None, :]).astype(np.int64)

    H_stoch = np.stack([bit_matrix(row) @ alpha for row in v_hat])
    H_hw = np.stack([hw_float32(row) for row in v_hat])

    # How distinguishable are the hypotheses now, compared to Hamming weight?
    def mean_cross_corr(H):
        C = np.corrcoef(H)
        np.fill_diagonal(C, np.nan)
        return np.nanmean(np.abs(C))

    print(f"    mean |corr| between hypotheses:  "
          f"stochastic {mean_cross_corr(H_stoch):.3f}  vs  HW {mean_cross_corr(H_hw):.3f}")
    print(f"    distinct predicted values:       "
          f"stochastic {len(np.unique(H_stoch[0].round(9)))}  vs  HW {len(np.unique(H_hw[0]))}")

    column = atk[:, ATTACK_POI].astype(np.float64)
    counts = np.arange(2, MAX_TRACES_CURVE + 1)
    true_idx = int(np.where(HYPOTHESES == TRUE_WEIGHT)[0][0])

    results = {}
    for name, H in (("stochastic (multi-POI)", H_stoch), ("Hamming weight", H_hw)):
        curves = np.zeros((len(HYPOTHESES), len(counts)))
        for j, m in enumerate(counts):
            curves[:, j] = correlate_at(H[:, :m], column[:m])
        full = correlate_at(H, column)
        order = np.argsort(-np.abs(full))
        recovered = HYPOTHESES[order[0]]
        margin = abs(full[order[0]]) - abs(full[order[1]])
        stable = traces_until_stable(curves, counts, true_idx)
        results[name] = (curves, recovered, stable)
        print(f"    {name:<24} recovered {recovered:+d}  "
              f"margin {margin:.4f}  stable from {stable if stable else '-'}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, (name, (curves, _, stable)) in zip(axes, results.items()):
        for i, w in enumerate(HYPOTHESES):
            if w == TRUE_WEIGHT:
                ax.plot(counts, curves[i], color="blue", lw=2.2, zorder=10, label=f"true w={w}")
            else:
                ax.plot(counts, curves[i], color="gray", alpha=0.45, lw=0.9)
        if stable:
            ax.axvline(stable, color="green", ls="--", lw=1.2, label=f"stable from {stable}")
        ax.set_title(name, fontweight="bold"); ax.set_xlabel("number of traces")
        ax.grid(alpha=0.3); ax.legend()
    axes[0].set_ylabel("r")
    fig.suptitle(f"Algorithm 3b — stochastic model fitted across {len(POI_WEIGHTS)} POIs",
                 fontweight="bold")
    out = os.path.join(GRAPHS, "alg3b_multipoi.png")
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print(f"    graph -> {out}")


if __name__ == "__main__":
    main()
