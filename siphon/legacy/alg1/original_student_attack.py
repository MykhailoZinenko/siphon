import os
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import (  # noqa: E402
    DATA, FIRMWARE, STUDENT_ATTACK, STUDENT_PROFILING, REBUILT_ATTACK,
    CAMPAIGN_WEIGHT, CAMPAIGN_COLUMN,
)
import numpy as np
import struct
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, "traces")

def hw_float32(f):
    try:
        [packed] = struct.unpack('!I', struct.pack('!f', float(f)))
        return bin(packed).count('1')
    except: return 0

# configuration
N_TRACES = 5000
WINDOW_START = 0
WINDOW_END = 10000 # whole network execution lays in the first 10k samples
INPUT_COLUMN = 1 # which input column whe should use to recover weight with hot spot cpa strategy
# all other inputs except this column mean to be zero
TRUE_WEIGHT = 4 # true value to be recovered, only for prettier comparative analysis

print(f"[*] Loading {N_TRACES} traces...")
inputs_raw = np.loadtxt(os.path.join(FOLDER, 'inputs.txt'))
inputs = inputs_raw[:N_TRACES, INPUT_COLUMN]

window_len = WINDOW_END - WINDOW_START + 1
traces = np.zeros((N_TRACES, window_len))

for i in range(N_TRACES):
    data = np.loadtxt(os.path.join(FOLDER, f'trace_{i}.txt'))
    traces[i, :] = data[WINDOW_START:WINDOW_END+1]

# correlation analysis
weight_candidates = np.arange(-7, 8)
r_traces = np.zeros((len(weight_candidates), window_len))

print("[*] Calculating Absolute Correlations...")
for w_idx, w_hyp in enumerate(weight_candidates):
    if w_hyp == 0: continue
    
    # create hypothetical leakage as product of multiplication candidate weight and input related to trace
    hyp_hw = np.array([hw_float32(float(w_hyp) * float(x)) for x in inputs])
    
    if np.std(hyp_hw) > 0:
        for t in range(window_len):
            # calculate correlation coefficient between hypothetical leakage and actual traces
            corr, _ = pearsonr(hyp_hw, traces[:, t])
            r_traces[w_idx, t] = abs(np.nan_to_num(corr))

# results
max_corr = np.max(r_traces, axis=1)
recovered = weight_candidates[np.argmax(max_corr)]

print("\n" + "="*35)
print(f"{'Weight':^10} | {'Peak Correlation':^15}")
print("-" * 35)
for i, w in enumerate(weight_candidates):
    if w == 0: continue
    
    line = f"{w:^10} | {max_corr[i]:^15.6f}"
    
    if w == TRUE_WEIGHT and w == recovered:
        print(f"\033[92m{line} [CORRECT]\033[0m")
    elif w == TRUE_WEIGHT:
        print(f"\033[91m{line} [TARGET]\033[0m") 
    elif w == recovered:
        print(f"\033[94m{line} [GUESSED]\033[0m")
    else:
        print(line)
print("="*35)

# visualization
plt.figure(figsize=(15, 7))
time_axis = np.arange(WINDOW_START, WINDOW_END + 1)

for i, w in enumerate(weight_candidates):
    if w == 0: continue
    
    if w == TRUE_WEIGHT:
        plt.plot(time_axis, r_traces[i, :], color='red', linewidth=2.5, label=f'TRUE ({w})', zorder=10)
    elif w == recovered:
        plt.plot(time_axis, r_traces[i, :], color='blue', linewidth=2, label=f'RECOVERED ({w})', zorder=5)
    else:
        plt.plot(time_axis, r_traces[i, :], color='gray', alpha=0.2, linewidth=0.8)

plt.title(f"CPA Window {WINDOW_START}-{WINDOW_END} (Weight {INPUT_COLUMN})", fontweight='bold')
plt.xlabel("Sample Index")
plt.ylabel("Absolute Correlation $|r|$")
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
os.makedirs(os.path.join(HERE, "graphs"), exist_ok=True)
out = os.path.join(HERE, "graphs", "cpa_unprotected.png")
plt.savefig(out, dpi=130, bbox_inches="tight")

print(f"\nFinal Result: Recovered {recovered} | Target {TRUE_WEIGHT}")
print(f"graph saved -> {out}")