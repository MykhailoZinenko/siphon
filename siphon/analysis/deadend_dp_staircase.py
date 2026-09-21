"""ТУПИК. Поиск лестницы динамическим программированием со свободными порогами.
Заметка 17: восемь ступеней хорошо приближают ЛЮБУЮ монотонную кривую, поэтому
критерий садится на аккумулятор. Не повторять.
"""
import numpy as np
from pathlib import Path
from siphon.sweep import recovered_weights
from siphon.traces import load_traces

B, K = 120, 8
d = Path("siphon/data/rebuilt_firmware/sweep")
inp = np.loadtxt(d / "inputs.txt").astype(int)
W1 = recovered_weights(sorted(Path("siphon/runs").glob("*layer-1-complete"))[-1] / "results.json")
acc = inp @ W1.T
tr = load_traces(d, 3000, (0, 24430)).astype(np.float64)

def segment_dp(m, K):
    """m: (B, S) binned means. Best K contiguous segments per column."""
    Bn, S = m.shape
    cs = np.vstack([np.zeros(S), np.cumsum(m, 0)])
    cs2 = np.vstack([np.zeros(S), np.cumsum(m**2, 0)])
    cost = np.full((Bn, Bn, S), np.inf)
    for i in range(Bn):
        j = np.arange(i, Bn)
        n = (j - i + 1)[:, None]
        s = cs[j + 1] - cs[i]
        q = cs2[j + 1] - cs2[i]
        cost[i, i:] = q - s**2 / n
    dp = np.full((K + 1, Bn + 1, S), np.inf)
    back = np.zeros((K + 1, Bn + 1, S), dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, K + 1):
        for j in range(k, Bn + 1):
            cand = dp[k - 1, k - 1:j] + cost[k - 1:j, j - 1]
            idx = np.argmin(cand, 0)
            dp[k, j] = cand[idx, np.arange(S)]
            back[k, j] = idx + (k - 1)
    return dp[K, Bn], back

print("поиск лестницы (истина не используется)\n")
print(f"{'нейрон':>7} {'точка':>6} {'R2 ступеней':>12} {'R2 линии':>9} {'пороги (acc)':<44}")
out = {}
for o in range(6):
    order = np.argsort(acc[:, o], kind="stable")
    edges = np.array_split(order, B)
    centres = np.array([acc[e, o].mean() for e in edges])
    m = np.stack([tr[e].mean(0) for e in edges])            # (B, samples)
    sst = ((m - m.mean(0)) ** 2).sum(0)
    # линейная модель по acc, для сравнения
    A = np.vstack([centres, np.ones(B)]).T
    resid_lin = m - A @ np.linalg.lstsq(A, m, rcond=None)[0]
    r2_lin = 1 - (resid_lin ** 2).sum(0) / np.maximum(sst, 1e-15)
    cand = np.where(r2_lin > -np.inf)[0]
    cand = cand[np.argsort(-sst[cand])[:400]]               # где вообще есть сигнал
    sse, back = segment_dp(m[:, cand], K)
    r2 = 1 - sse / np.maximum(sst[cand], 1e-15)
    gain = r2 - r2_lin[cand]
    best = int(np.argmax(gain))
    t = int(cand[best])
    cuts, j = [], B
    for k in range(K, 0, -1):
        i = back[k, j, best]
        cuts.append(i); j = i
    thr = [centres[c] for c in sorted(cuts)[1:]]
    out[o] = thr
    print(f"{o:>7} {t:>6} {r2[best]:>12.3f} {r2_lin[cand][best]:>9.3f} "
          + " ".join(f"{v:7.1f}" for v in thr))
np.save("thr.npy", np.array([out[o] for o in range(6)]))
