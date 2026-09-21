"""
Algorithm 1 — Non-profiled DPA (textbook section 4.3.1), adapted from a block
cipher to a quantized neural network.

Book target:  v = SBox(k XOR p),  k = key nibble, 16 hypotheses
Our target:   v = w * x,          w = weight,     14 hypotheses (-7..7 \ {0})

Runs DPA Step 6 with BOTH leakage models the book presents:
  * identity  -> H = v          (expected to DEGENERATE here, see note below)
  * Hamming   -> H = wt(v)

Why identity degenerates: every hypothesis is w_hat * x_j, i.e. a scalar multiple
of the same vector x. Pearson correlation is invariant under scaling of one
variable, so all 14 hypotheses yield identical |r|. The book's S-box is
nonlinear, which is exactly what stops this happening for PRESENT.
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
TRACES = STUDENT_ATTACK
GRAPHS = os.path.join(HERE, "graphs")

# ---- DPA Step 2: measured leakages -----------------------------------------
M_P = 5000            # number of attack traces
WINDOW = (0, 10000)   # network execution lies in the first 10k samples

# ---- DPA Step 3: part of the "key" to recover ------------------------------
INPUT_COLUMN = 1      # all other inputs are zero -> isolates one multiply
TRUE_WEIGHT = 4       # ground truth, used only for labelling the plots

# ---- DPA Step 5: hypotheses -------------------------------------------------
HYPOTHESES = np.array([w for w in range(-7, 8) if w != 0])   # M_k = 14

ACC_BITS = 32         # device accumulates int8*int8 into an int32 register


def load_traces():
    """Load traces, caching to .npy — plain text reload takes minutes."""
    cache = os.path.join(TRACES, f"_cache_{M_P}_{WINDOW[0]}_{WINDOW[1]}.npy")
    if os.path.exists(cache):
        return np.load(cache)

    lo, hi = WINDOW
    out = np.empty((M_P, hi - lo), dtype=np.float32)
    for i in range(M_P):
        with open(os.path.join(TRACES, f"trace_{i}.txt")) as fh:
            row = np.array(fh.read().split(), dtype=np.float32)
        out[i] = row[lo:hi]
        if i % 1000 == 0:
            print(f"    {i}/{M_P}")
    np.save(cache, out)
    return out


def hamming_weight_float32(values):
    """wt of the IEEE-754 float32 encoding of v.

    Empirically the only model here that separates the hypotheses. It is NOT
    physically justified — the device does int8 arithmetic, nothing float —
    but the float32 encoding is a strongly nonlinear function of v (exponent
    tracks magnitude, mantissa the significand), which is what breaks the
    collinearity. Guessing a model that happens to work is exactly the
    weakness that the profiled attacks in 4.3.2 remove.
    """
    return np.array(
        [bin(struct.unpack("!I", struct.pack("!f", float(a)))[0]).count("1")
         for a in values], dtype=np.float64)


def hamming_weight(values, bits=ACC_BITS):
    """wt(v) over the two's-complement representation the device actually holds."""
    as_uint = np.asarray(values, dtype=np.int64) & ((1 << bits) - 1)
    return np.array([bin(int(v)).count("1") for v in as_uint], dtype=np.float64)


def correlate(H, T):
    """Textbook Equation 4.21, for every hypothesis and every time sample.

    H : (M_k, M_p) hypothetical signals
    T : (M_p, q)   measured leakages
    -> (M_k, q)    sample correlation coefficients r_{i,t}
    """
    Hc = H - H.mean(axis=1, keepdims=True)
    Tc = T - T.mean(axis=0, keepdims=True)
    num = Hc @ Tc
    den = np.sqrt((Hc ** 2).sum(1)[:, None] * (Tc ** 2).sum(0)[None, :])
    # some time samples are constant -> den = 0; divide only where it is safe
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den > 0)
    return out


def report(name, r):
    peak = np.abs(r).max(axis=1)
    recovered = HYPOTHESES[peak.argmax()]
    spread = peak.max() - peak.min()

    print(f"\n--- {name} leakage model " + "-" * 30)
    for w, p in zip(HYPOTHESES, peak):
        mark = ""
        if w == TRUE_WEIGHT:
            mark = "  <- TRUE"
        if w == recovered and w != TRUE_WEIGHT:
            mark = "  <- recovered (WRONG)"
        print(f"  w = {w:+d}   |r|peak = {p:.6f}{mark}")
    print(f"  recovered {recovered:+d}, true {TRUE_WEIGHT:+d}"
          f"  |  spread across hypotheses = {spread:.2e}")
    if spread < 1e-9:
        print("  !! all hypotheses identical -> distinguisher carries NO information")
    return recovered, peak


def plot(name, r, fname):
    plt.figure(figsize=(14, 6))
    t = np.arange(WINDOW[0], WINDOW[1])
    for i, w in enumerate(HYPOTHESES):
        if w == TRUE_WEIGHT:
            plt.plot(t, np.abs(r[i]), color="red", lw=2.0, zorder=10, label=f"true w={w}")
        else:
            plt.plot(t, np.abs(r[i]), color="gray", alpha=0.35, lw=0.7)
    plt.title(f"Algorithm 1 — non-profiled DPA, {name} leakage model", fontweight="bold")
    plt.xlabel("time sample")
    plt.ylabel("|r|")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    os.makedirs(GRAPHS, exist_ok=True)
    path = os.path.join(GRAPHS, fname)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  graph -> {path}")


def main():
    print(f"[*] loading {M_P} traces, window {WINDOW} ...")
    traces = load_traces()
    inputs = np.loadtxt(os.path.join(TRACES, "inputs.txt"))[:M_P, INPUT_COLUMN]
    print(f"    traces {traces.shape}, inputs {inputs.shape}, "
          f"distinct x values: {len(np.unique(inputs))}")

    # DPA Step 4/5: hypothetical intermediate values  v_hat = w_hat * x
    v_hat = HYPOTHESES[:, None] * inputs[None, :]        # (M_k, M_p)

    # DPA Step 6: leakage models
    H_identity = v_hat.astype(np.float64)
    H_hamming_f32 = np.stack([hamming_weight_float32(row) for row in v_hat])

    # DPA Step 7: statistical analysis
    for name, H, fname in (
        ("identity", H_identity, "alg1_identity.png"),
        ("Hamming weight (float32 enc.)", H_hamming_f32, "alg1_hamming.png"),
    ):
        r = correlate(H, traces)
        report(name, r)
        plot(name, r, fname)


if __name__ == "__main__":
    main()
