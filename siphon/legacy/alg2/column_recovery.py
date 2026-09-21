"""
Algorithm 2b — POI selection done properly, and full-column weight recovery.

Why this exists: `POI = argmax SNR` does not identify WHICH weight is being
attacked. The hot-spot zeroes nine INPUTS, but input[1] is read by all six
neurons of layer 1, so each neuron multiplies the same x by its own
L1_W[o][1]. There are therefore ~6 leaking moments, one per neuron, and the
global SNR maximum lands on whichever neuron happens to leak hardest.

So instead of taking the maximum, we enumerate every SNR peak and attack at
each one. That turns the ambiguity into a stronger result: one capture set
yields the entire column L1_W[:][1] rather than a single weight.
"""
import os
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import (  # noqa: E402
    DATA, FIRMWARE, STUDENT_ATTACK, STUDENT_PROFILING, REBUILT_ATTACK,
    CAMPAIGN_WEIGHT, CAMPAIGN_COLUMN,
)
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profiled_dpa import (          # noqa: E402
    load, snr, correlate_at, hw_float32,
    PROFILING, ATTACK, GRAPHS, M_PROFILING, M_ATTACK,
    INPUT_COLUMN, TRUE_WEIGHT, HYPOTHESES, WINDOW,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def ground_truth_column(column=INPUT_COLUMN):
    """Read L1_W[:][column] out of network_config.h rather than hardcoding it."""
    text = open(os.path.join(FIRMWARE, "network_config.h")).read()
    block = re.search(r"L1_W\[NET_L1\]\[NET_INPUTS\]\s*=\s*\{(.*?)\};", text, re.S).group(1)
    rows = re.findall(r"\{([^}]*)\}", block)
    return [int(r.split(",")[column]) for r in rows]


def main():
    truth = ground_truth_column()
    print(f"[*] ground truth L1_W[:][{INPUT_COLUMN}] = {truth}")

    prof = load(PROFILING, M_PROFILING)
    x_prof = np.loadtxt(os.path.join(PROFILING, "inputs.txt"))[:M_PROFILING, INPUT_COLUMN]
    # Grouping by the true v is the same partition as grouping by x, so this SNR
    # highlights every x-dependent moment — i.e. all six neurons at once.
    s, _ = snr(prof, (TRUE_WEIGHT * x_prof).astype(np.int64))

    peaks, _ = find_peaks(s, height=s.max() * 0.20, distance=200)
    peaks = peaks[np.argsort(-s[peaks])][:12]
    peaks = np.sort(peaks)
    print(f"[*] {len(peaks)} SNR peaks above 20% of max")

    atk = load(ATTACK, M_ATTACK)
    x_atk = np.loadtxt(os.path.join(ATTACK, "inputs.txt"))[:M_ATTACK, INPUT_COLUMN]
    v_hat = (HYPOTHESES[:, None] * x_atk[None, :]).astype(np.int64)
    H = np.stack([hw_float32(row) for row in v_hat])

    print(f"\n{'POI':>6} {'SNR':>9} {'weight':>8} {'|r|':>7}   neuron")
    print("-" * 48)
    found = {}
    rows = []
    for p in peaks:
        r = correlate_at(H, atk[:, p].astype(np.float64))
        w = int(HYPOTHESES[np.argmax(np.abs(r))])
        peak_r = float(np.abs(r).max())
        neuron = truth.index(w) if w in truth else None
        label = f"L1_W[{neuron}][{INPUT_COLUMN}]" if neuron is not None else "— не из столбца"
        rows.append((p, s[p], w, peak_r, neuron))
        if neuron is not None and neuron not in found:
            found[neuron] = (p, w, peak_r)
        print(f"{p:6d} {s[p]:9.1f} {w:+8d} {peak_r:7.4f}   {label}")

    print(f"\n[*] recovered {len(found)} of {len(truth)} weights in this column")
    print(f"{'neuron':>7} {'true':>6} {'recovered':>10} {'POI':>7}  ok")
    print("-" * 42)
    correct = 0
    for o, t in enumerate(truth):
        if o in found:
            p, w, _ = found[o]
            ok = (w == t)
            correct += ok
            print(f"{o:7d} {t:+6d} {w:+10d} {p:7d}  {'yes' if ok else 'NO'}")
        else:
            print(f"{o:7d} {t:+6d} {'—':>10} {'—':>7}  not found")
    print(f"\n[*] {correct}/{len(truth)} weights of column {INPUT_COLUMN} recovered correctly")

    # ---- plot -------------------------------------------------------------
    plt.figure(figsize=(15, 5.5))
    t_axis = np.arange(*WINDOW)
    plt.plot(t_axis, s, lw=0.7, color="steelblue")
    for p, sv, w, _, neuron in rows:
        colour = "green" if neuron is not None else "gray"
        plt.axvline(p, color=colour, ls=":", lw=1, alpha=0.7)
        plt.annotate(f"w={w:+d}" + (f"\nn{neuron}" if neuron is not None else ""),
                     xy=(p, sv), xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=8, color=colour, fontweight="bold")
    plt.title("Algorithm 2b — every SNR peak leaks a different neuron's weight",
              fontweight="bold")
    plt.xlabel("time sample"); plt.ylabel("SNR"); plt.grid(alpha=0.3)
    os.makedirs(GRAPHS, exist_ok=True)
    out = os.path.join(GRAPHS, "alg2b_column.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    graph -> {out}")


if __name__ == "__main__":
    main()
