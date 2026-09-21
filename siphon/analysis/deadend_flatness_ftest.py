"""ТУПИК (частично). F-тест на плоскость ступеней.
Заметка 17: как ДИАГНОСТИКА работает (1.3-2.2 против 48-285 на истинных метках),
но как слепой оценщик вырожден по масштабу. Диагностическая половина полезна.
"""
import numpy as np, re
from pathlib import Path
from siphon.sweep import recovered_weights
from siphon.traces import load_traces

NB, NCAND = 240, 150
INV = np.arange(12.0, 90.01, 0.25)
BIAS = np.arange(-90.0, 90.01, 1.0)

d = Path("siphon/data/rebuilt_firmware/sweep")
inp = np.loadtxt(d / "inputs.txt").astype(int)
W1 = recovered_weights(sorted(Path("siphon/runs").glob("*layer-1-complete"))[-1] / "results.json")
acc = inp @ W1.T
tr = load_traces(d, 3000, (0, 24430)).astype(np.float64)
src = open("siphon/firmware/source/network_config.h").read()
def grab(x):
    b = re.search(rf"{x}\[[^=]*=\s*\{{(.*?)\}};", src, re.S).group(1)
    rows = re.findall(r"\{([^}]*)\}", b)
    return [[int(v) for v in r.split(",")] for r in rows] if rows else [int(v) for v in b.split(",")]
B1, M01, N1 = map(np.array, (grab("L1_B"), grab("L1_M0"), grab("L1_N")))
S = M01 / 2.0**(31+N1)

print("F = (остаток внутри ступеней) / (шум). Плоская ступень => F ~ 1\n")
print(f"{'нейрон':>7} {'точка':>7} {'что это':<13} {'F':>8}")
prep = {}
for o in range(6):
    order = np.argsort(acc[:, o], kind="stable")
    gs = np.array_split(order, NB)
    c = np.array([acc[g, o].mean() for g in gs])
    M = np.stack([tr[g].mean(0) for g in gs])
    noise = np.stack([tr[g].var(0) / len(g) for g in gs]).mean(0)   # дисперсия среднего корзины
    prep[o] = (c, M, noise)
    a_true = np.clip(np.floor((acc[:, o] + B1[o]) * S[o] + 0.5), 0, 7).astype(int)
    lab = np.array([np.bincount(a_true[g], minlength=8).argmax() for g in gs])
    for t, name in (((7801,4870,5401,6021,6730,7350)[o], "активация"),
                    ((4233,4813,5433,6053,6717,7333)[o], "аккумулятор")):
        sse = sum(((M[lab == g, t] - M[lab == g, t].mean()) ** 2).sum()
                  for g in range(8) if (lab == g).sum() > 1)
        df = NB - len(np.unique(lab))
        print(f"{o:>7} {t:>7} {name:<13} {sse/df/noise[t]:>8.2f}")

# калибровка порога амплитуды по известным точкам
sp = []
for o in range(6):
    c, M, noise = prep[o]
    t = (7801,4870,5401,6021,6730,7350)[o]
    a_true = np.clip(np.floor((acc[:, o] + B1[o]) * S[o] + 0.5), 0, 7).astype(int)
    order = np.argsort(acc[:, o], kind="stable"); gs = np.array_split(order, NB)
    lab = np.array([np.bincount(a_true[g], minlength=8).argmax() for g in gs])
    means = np.array([M[lab == g, t].mean() for g in range(8) if (lab == g).sum()])
    sp.append(means.std() ** 2 / noise[t])
print("\nразброс ступеней / шум в известных точках:",
      " ".join(f"{v:.0f}" for v in sp))
SPREAD_MIN = min(sp) / 3

print("\nпоиск с этим критерием: минимизируем F, среди тех, у кого амплитуда заметна\n")
print(f"{'нейрон':>7} {'точка':>6} {'1/s':>7} {'b':>6} {'F':>6} | {'1/s ист.':>9} {'b ист.':>7}")
for o in range(6):
    c, M, noise = prep[o]
    amp = M.max(0) - M.min(0)
    cand = np.argsort(-amp)[:NCAND]
    Mc, nz = M[:, cand], noise[cand]
    C1 = np.concatenate([[0.0], np.cumsum(np.ones(NB))])
    S1 = np.vstack([np.zeros(len(cand)), np.cumsum(Mc, 0)])
    S2 = np.vstack([np.zeros(len(cand)), np.cumsum(Mc ** 2, 0)])
    best = (1e18, None)
    for inv in INV:
        thr = (np.arange(1, 8) - 0.5) * inv - BIAS[:, None]
        cuts = np.searchsorted(c, thr)
        ok = (cuts[:, 0] >= 6) & (cuts[:, -1] <= NB - 6) & (np.diff(cuts, 1) >= 5).all(1)
        if not ok.any(): continue
        cc = np.concatenate([np.zeros((ok.sum(),1),int), cuts[ok], np.full((ok.sum(),1),NB,int)],1)
        lo, hi = cc[:, :-1], cc[:, 1:]
        cnt = (C1[hi] - C1[lo])[:, :, None]
        s1, s2 = S1[hi] - S1[lo], S2[hi] - S2[lo]
        sse = (s2 - s1 ** 2 / np.maximum(cnt, 1e-9)).sum(1)
        F = sse / (NB - 8) / nz
        # требуем реальную ступенчатость: разброс средних ступеней много больше шума
        spread = (s1 / np.maximum(cnt, 1e-9)).std(1) ** 2 / nz
        F = np.where(spread > SPREAD_MIN, F, 1e18)
        h, t = np.unravel_index(np.argmin(F), F.shape)
        if F[h, t] < best[0]:
            best = (F[h, t], (int(cand[t]), inv, BIAS[ok][h]))
    f, (t, inv, b) = best
    print(f"{o:>7} {t:>6} {inv:>7.2f} {b:>6.0f} {f:>6.2f} | {1/S[o]:>9.2f} {B1[o]:>7d}")
