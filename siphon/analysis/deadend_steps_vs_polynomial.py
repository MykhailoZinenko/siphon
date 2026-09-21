"""ТУПИК. Ступени против полинома 7-й степени.
Заметка 17: держится на ложном допущении, что утечка аккумулятора гладкая.
Она HW(acc) и меняется на каждой единице. Признак переворачивается между нейронами.
"""
import numpy as np, re
from pathlib import Path
from siphon.sweep import recovered_weights
from siphon.traces import load_traces

NB, K, DEG = 100, 8, 7
POI = [4233, 4813, 5433, 6053, 6717, 7333]
ACT = [7801, 4870, 5401, 6021, 6730, 7350]        # известные, только для валидации

d = Path("siphon/data/rebuilt_firmware/sweep")
inp = np.loadtxt(d / "inputs.txt").astype(int)
W1 = recovered_weights(sorted(Path("siphon/runs").glob("*layer-1-complete"))[-1] / "results.json")
acc = inp @ W1.T
tr = load_traces(d, 3000, (0, 24430)).astype(np.float64)

def dp8(m, K):
    Bn, S = m.shape
    cs = np.vstack([np.zeros(S), np.cumsum(m, 0)])
    cs2 = np.vstack([np.zeros(S), np.cumsum(m ** 2, 0)])
    cost = np.full((Bn + 1, Bn + 1, S), np.inf, dtype=np.float64)
    for i in range(Bn):
        j = np.arange(i + 1, Bn + 1); n = (j - i)[:, None]
        cost[i, i + 1:] = (cs2[j] - cs2[i]) - (cs[j] - cs[i]) ** 2 / n
    dp = np.full((K + 1, Bn + 1, S), np.inf); back = np.zeros((K + 1, Bn + 1, S), np.int16)
    dp[0, 0] = 0
    for k in range(1, K + 1):
        for j in range(k, Bn + 1):
            cand = dp[k - 1, k - 1:j] + cost[k - 1:j, j]
            i = np.argmin(cand, 0); dp[k, j] = cand[i, np.arange(S)]; back[k, j] = i + (k - 1)
    return dp[K, Bn], back

def prep(o, win):
    order = np.argsort(acc[:, o], kind="stable"); gs = np.array_split(order, NB)
    c = np.array([acc[g, o].mean() for g in gs])
    M = np.stack([tr[np.ix_(g, win)].mean(0) for g in gs])
    noise = np.stack([tr[np.ix_(g, win)].var(0) / len(g) for g in gs]).mean(0)
    return c, M, noise

print("валидация признака: SSE(8 ступеней) / SSE(полином 7-й степени)\n")
print(f"{'нейрон':>7} {'точка':>7} {'что это':<13} {'отношение':>10} {'разброс/шум':>12}")
for o in range(6):
    win = np.array([ACT[o], POI[o]])
    c, M, noise = prep(o, win)
    xs = (c - c.mean()) / c.std()
    V = np.vander(xs, DEG + 1)
    resid_poly = M - V @ np.linalg.lstsq(V, M, rcond=None)[0]
    sse_poly = (resid_poly ** 2).sum(0)
    sse_step, _ = dp8(M, K)
    for i, name in enumerate(("активация", "аккумулятор")):
        print(f"{o:>7} {win[i]:>7} {name:<13} {sse_step[i]/max(sse_poly[i],1e-18):>10.3f} "
              f"{M[:, i].var()/noise[i]:>12.0f}")
