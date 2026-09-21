"""
Algorithm 2 — Profiled DPA (textbook section 4.3.2.1), adapted to a quantized NN.

Profiling set: traces_profiling/   (analogue of the book's Random dataset)
Attack set:    traces/             (analogue of Random plaintext dataset)

The 13 steps; only 5,6,7-8,9,13 differ from Algorithm 1:
  Step 5      choose target signal: exact v, or wt(v)
  Step 6      group profiling traces into sets A_s by that signal
  Step 7-8    model leakage as signal+noise, compute SNR per time sample
  Step 9      POI = argmax SNR
  Step 13     correlate at the POI only, versus number of attack traces (Eq 4.23)

Note: Step 12 still *chooses* a leakage model by hand. Profiling fixes WHERE to
look, not WHAT to look for. The float32 Hamming weight from Algorithm 1 is kept
because identity remains degenerate for a linear layer.
"""
import os
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import (  # noqa: E402
    DATA, FIRMWARE, STUDENT_ATTACK, STUDENT_PROFILING, REBUILT_ATTACK,
    CAMPAIGN_WEIGHT, CAMPAIGN_COLUMN,
)
import struct

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILING = STUDENT_PROFILING
ATTACK = STUDENT_ATTACK
GRAPHS = os.path.join(HERE, "graphs")

M_PROFILING = 5000
M_ATTACK = 5000
WINDOW = (0, 10000)
INPUT_COLUMN = 1
TRUE_WEIGHT = 4
HYPOTHESES = np.array([w for w in range(-7, 8) if w != 0])
MAX_TRACES_CURVE = 50          # book's Figures 4.39/4.40 use 50


def load(folder, count):
    tag = os.path.basename(folder)
    cache = os.path.join(folder, f"_cache_{tag}_{count}_{WINDOW[0]}_{WINDOW[1]}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    lo, hi = WINDOW
    out = np.empty((count, hi - lo), dtype=np.float32)
    for i in range(count):
        with open(os.path.join(folder, f"trace_{i}.txt")) as fh:
            out[i] = np.array(fh.read().split(), dtype=np.float32)[lo:hi]
        if i % 1000 == 0:
            print(f"    {tag}: {i}/{count}")
    np.save(cache, out)
    return out


def hw_float32(values):
    return np.array(
        [bin(struct.unpack("!I", struct.pack("!f", float(a)))[0]).count("1")
         for a in values], dtype=np.float64)


def snr(traces, labels):
    """P-DPA Step 8. Var of the class means over Var of the residuals."""
    classes = np.unique(labels)
    means = np.zeros((len(classes), traces.shape[1]), dtype=np.float64)
    for i, c in enumerate(classes):
        means[i] = traces[labels == c].mean(axis=0)

    index = {c: i for i, c in enumerate(classes)}
    per_trace_mean = means[[index[c] for c in labels]]      # signal estimate
    signal_var = per_trace_mean.var(axis=0)
    noise_var = (traces - per_trace_mean).var(axis=0)
    out = np.zeros_like(signal_var)
    np.divide(signal_var, noise_var, out=out, where=noise_var > 0)
    return out, len(classes)


def correlate_at(H, column):
    """Equation 4.23 at a single time sample. H:(M_k, n), column:(n,)"""
    Hc = H - H.mean(axis=1, keepdims=True)
    c = column - column.mean()
    num = Hc @ c
    den = np.sqrt((Hc ** 2).sum(1) * (c ** 2).sum())
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den > 0)
    return out


def main():
    print("[*] loading profiling traces ...")
    prof = load(PROFILING, M_PROFILING)
    x_prof = np.loadtxt(os.path.join(PROFILING, "inputs.txt"))[:M_PROFILING, INPUT_COLUMN]

    # Step 5+6: target signal = exact v, and wt(v). Group by it.
    v_true = TRUE_WEIGHT * x_prof
    signals = {
        "exact v": v_true,
        "wt(v)": hw_float32(v_true),
    }

    pois = {}
    plt.figure(figsize=(14, 5))
    for name, labels in signals.items():
        s, n_classes = snr(prof, labels)
        poi = int(np.argmax(s))
        pois[name] = poi
        print(f"    target signal '{name}': {n_classes} classes, "
              f"POI = {poi}, SNR = {s[poi]:.4f}")
        plt.plot(np.arange(*WINDOW), s, lw=0.8, label=f"{name} (POI={poi}, SNR={s[poi]:.2f})")
        plt.axvline(poi, color="red", ls=":", lw=1, alpha=0.6)

    plt.title("Algorithm 2 — SNR per time sample (profiling set)", fontweight="bold")
    plt.xlabel("time sample"); plt.ylabel("SNR"); plt.grid(alpha=0.3); plt.legend()
    os.makedirs(GRAPHS, exist_ok=True)
    plt.savefig(os.path.join(GRAPHS, "alg2_snr.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    graph -> {GRAPHS}/alg2_snr.png")

    # Steps 10-13: attack at the POI
    print("[*] loading attack traces ...")
    atk = load(ATTACK, M_ATTACK)
    x_atk = np.loadtxt(os.path.join(ATTACK, "inputs.txt"))[:M_ATTACK, INPUT_COLUMN]

    v_hat = HYPOTHESES[:, None] * x_atk[None, :]
    H = np.stack([hw_float32(row) for row in v_hat])        # Step 12

    poi = pois["exact v"]
    counts = np.arange(2, MAX_TRACES_CURVE + 1)
    curves = np.zeros((len(HYPOTHESES), len(counts)))
    for j, m in enumerate(counts):
        curves[:, j] = correlate_at(H[:, :m], atk[:m, poi].astype(np.float64))

    ranks = np.argsort(-np.abs(curves), axis=0)
    true_idx = int(np.where(HYPOTHESES == TRUE_WEIGHT)[0][0])
    top = ranks[0] == true_idx
    stable = None
    for j in range(len(counts)):
        if top[j:].all():
            stable = counts[j]
            break

    full = correlate_at(H, atk[:, poi].astype(np.float64))
    recovered = HYPOTHESES[np.argmax(np.abs(full))]
    print(f"\n[*] attack at POI = {poi}")
    print(f"    recovered {recovered:+d}, true {TRUE_WEIGHT:+d}  "
          f"|r| = {abs(full[true_idx]):.4f}")
    print(f"    correct weight stays top from {stable} traces onward"
          if stable else "    correct weight never stabilises at the top")

    plt.figure(figsize=(11, 6))
    for i, w in enumerate(HYPOTHESES):
        if w == TRUE_WEIGHT:
            plt.plot(counts, curves[i], color="blue", lw=2.2, zorder=10, label=f"true w={w}")
        else:
            plt.plot(counts, curves[i], color="gray", alpha=0.45, lw=0.9)
    if stable:
        plt.axvline(stable, color="green", ls="--", lw=1.2, label=f"stable from {stable}")
    plt.title(f"Algorithm 2 — correlation at POI={poi} vs number of attack traces",
              fontweight="bold")
    plt.xlabel("number of traces"); plt.ylabel("r"); plt.grid(alpha=0.3); plt.legend()
    plt.savefig(os.path.join(GRAPHS, "alg2_traces.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    graph -> {GRAPHS}/alg2_traces.png")


if __name__ == "__main__":
    main()
