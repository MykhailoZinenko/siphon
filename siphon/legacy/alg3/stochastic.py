"""
Algorithm 3 — Stochastic leakage model (textbook section 4.3.2.2).

Instead of ASSUMING the leakage is wt(v) (every bit contributing equally), we
FIT one coefficient per bit on the profiling traces:

    L(v) = a_0*v_0 + a_1*v_1 + ... + a_7*v_7 + noise            (Eq 4.24)

SLM Step a  collect leakages at the POI from every profiling trace
SLM Step b  build M_v, one row per profiling trace = the bits of its true v
SLM Step c  a_hat = (M_v^T M_v)^-1 M_v^T l_pf                    (Eq 4.26)

Then P-DPA Step 12 uses L(v_hat) as the hypothetical signal, and Step 13 is the
same correlation at the POI as in Algorithm 2.

Hamming weight is the special case where every a_s is equal, so this generalises
Algorithm 2 rather than replacing it.
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
    load, snr, correlate_at, hw_float32,
    PROFILING, ATTACK, GRAPHS, M_PROFILING, M_ATTACK,
    INPUT_COLUMN, TRUE_WEIGHT, HYPOTHESES, MAX_TRACES_CURVE,
)

# each legacy folder writes its own figures, not the one it imports helpers from
GRAPHS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")

BITS = 8          # int8 two's complement: the width of the product w*x


def bit_matrix(values, bits=BITS):
    """Rows of bits, LSB first — the book's M_v (Eq 4.25)."""
    as_uint = np.asarray(values, dtype=np.int64) & ((1 << bits) - 1)
    return np.stack([(as_uint >> s) & 1 for s in range(bits)], axis=1).astype(np.float64)


def traces_until_stable(curves, counts, true_idx):
    top = np.argsort(-np.abs(curves), axis=0)[0] == true_idx
    for j in range(len(counts)):
        if top[j:].all():
            return counts[j]
    return None


def main():
    print("[*] profiling ...")
    prof = load(PROFILING, M_PROFILING)
    x_prof = np.loadtxt(os.path.join(PROFILING, "inputs.txt"))[:M_PROFILING, INPUT_COLUMN]
    v_true = (TRUE_WEIGHT * x_prof).astype(np.int64)

    # POI, recomputed here so this script stands alone (same formula as Algorithm 2)
    s, _ = snr(prof, v_true)
    poi = int(np.argmax(s))
    print(f"    POI = {poi}  (SNR = {s[poi]:.2f})")

    # --- SLM Step a: leakages at the POI ------------------------------------
    l_pf = prof[:, poi].astype(np.float64)

    # --- SLM Step b: the design matrix of bits ------------------------------
    M_v = bit_matrix(v_true)
    varying = M_v.std(axis=0) > 0
    dead = np.where(~varying)[0]
    print(f"    M_v shape {M_v.shape}; bits that never vary: {list(dead)}")

    # --- SLM Step c: least squares -----------------------------------------
    # The book inverts (M_v^T M_v) directly. We cannot: v = 4*x is always a
    # multiple of 4, so bits 0 and 1 are identically zero, the matrix is
    # singular, and the inverse does not exist. lstsq gives the minimum-norm
    # solution instead, which sets the coefficients of dead bits to zero.
    alpha, *_ = np.linalg.lstsq(M_v, l_pf, rcond=None)
    print("    fitted coefficients a_s (LSB first):")
    for i, a in enumerate(alpha):
        note = "   <- never varies, coefficient meaningless" if i in dead else ""
        print(f"      a_{i} = {a:+.6f}{note}")

    rank = np.linalg.matrix_rank(M_v)
    print(f"    rank(M_v) = {rank} of {BITS} -> "
          f"{'RANK DEFICIENT' if rank < BITS else 'full rank'}")

    # --- attack --------------------------------------------------------------
    print("[*] attacking ...")
    atk = load(ATTACK, M_ATTACK)
    x_atk = np.loadtxt(os.path.join(ATTACK, "inputs.txt"))[:M_ATTACK, INPUT_COLUMN]
    v_hat = (HYPOTHESES[:, None] * x_atk[None, :]).astype(np.int64)

    # Step 12 with the stochastic model, and with Hamming weight for comparison
    H_stoch = np.stack([bit_matrix(row) @ alpha for row in v_hat])
    H_hw = np.stack([hw_float32(row) for row in v_hat])

    column = atk[:, poi].astype(np.float64)
    counts = np.arange(2, MAX_TRACES_CURVE + 1)
    true_idx = int(np.where(HYPOTHESES == TRUE_WEIGHT)[0][0])

    results = {}
    for name, H in (("stochastic", H_stoch), ("Hamming weight", H_hw)):
        curves = np.zeros((len(HYPOTHESES), len(counts)))
        for j, m in enumerate(counts):
            curves[:, j] = correlate_at(H[:, :m], column[:m])
        full = correlate_at(H, column)
        recovered = HYPOTHESES[np.argmax(np.abs(full))]
        stable = traces_until_stable(curves, counts, true_idx)
        results[name] = (curves, full, recovered, stable)
        print(f"    {name:<16} recovered {recovered:+d}  "
              f"|r| = {abs(full[true_idx]):.4f}  "
              f"stable from {stable if stable else '-'} traces")

    # --- plot ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, (name, (curves, _, _, stable)) in zip(axes, results.items()):
        for i, w in enumerate(HYPOTHESES):
            if w == TRUE_WEIGHT:
                ax.plot(counts, curves[i], color="blue", lw=2.2, zorder=10, label=f"true w={w}")
            else:
                ax.plot(counts, curves[i], color="gray", alpha=0.45, lw=0.9)
        if stable:
            ax.axvline(stable, color="green", ls="--", lw=1.2, label=f"stable from {stable}")
        ax.set_title(f"{name} model", fontweight="bold")
        ax.set_xlabel("number of traces"); ax.grid(alpha=0.3); ax.legend()
    axes[0].set_ylabel("r")
    fig.suptitle(f"Algorithm 3 — stochastic vs Hamming weight at POI={poi}", fontweight="bold")
    os.makedirs(GRAPHS, exist_ok=True)
    out = os.path.join(GRAPHS, "alg3_stochastic.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    graph -> {out}")


if __name__ == "__main__":
    main()
