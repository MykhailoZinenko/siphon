"""Текущая атака на слой 2 (5/6): точки по критерию 'зависит от v, не от веса'.
Заметка 18. Не финал — какой вес теряется, зависит от выбора точек.
"""
import re, numpy as np
from pathlib import Path
from siphon.templates import build, coverage, score
from siphon.traces import load_traces

D = Path("siphon/data/rebuilt_firmware/layer2"); WIN = (7000, 9500)
tags = np.loadtxt(D / "tags.txt").astype(int)
man = {int(l.split()[0]): [int(v) for v in l.split()[1:]]
       for l in (D / "manifest.txt").read_text().splitlines()}
ks = sorted(man)
prof = [load_traces(D / f"k{k}", 1500, WIN) for k in ks]
atk = load_traces(D / "attack", 1500, WIN)
src = open("siphon/firmware/source/network_config.h").read()
W2 = [int(v) for v in re.search(r"L2_W\[NET_L2\]\[NET_L1\]\s*=\s*\{\s*\{([^}]*)\}", src, re.S).group(1).split(",")]
HYP = tuple(w for w in range(-7, 8) if w != 0)

SEP, NPOI = 8, 6
print("выбор точек: сильно зависит от v, слабо — от самого веса\n")
print(f"{'вход':>5} {'POI':>34} {'eta2(v)':>8} {'ост. по w':>10} {'w2':>5} {'истина':>7} {'запас':>11} {'':>3}")
got = []
for j in range(6):
    sel = tags[:, 0] == j
    a = tags[sel, 1]
    v = np.concatenate([man[k][j] * a for k in ks]).astype(np.int64)
    wp = np.concatenate([np.full(sel.sum(), man[k][j]) for k in ks])
    X = np.concatenate([t[sel] for t in prof]).astype(np.float64)

    tot = X.var(0) * len(X)
    ssw = np.zeros(X.shape[1]); resid_w = np.zeros(X.shape[1])
    for u in np.unique(v):
        rows = v == u
        if rows.sum() < 8: continue
        blk = X[rows]; mu = blk.mean(0)
        ssw += ((blk - mu) ** 2).sum(0)
        for ww in np.unique(wp[rows]):            # внутри класса v — разброс по весу
            sub = blk[wp[rows] == ww]
            if len(sub) < 4: continue
            resid_w += len(sub) * (sub.mean(0) - mu) ** 2
    eta = 1 - ssw / np.maximum(tot, 1e-12)
    frac = resid_w / np.maximum(ssw, 1e-12)
    cand = np.where((eta > 0.5) & (frac < 0.05))[0]
    if not len(cand):
        cand = np.array([int(np.argmax(eta))])
    pois = []
    for t in cand[np.argsort(-eta[cand])]:
        if all(abs(int(t) - q) > SEP for q in pois):
            pois.append(int(t))
        if len(pois) == NPOI: break
    poi = pois[0]
    tmpl = build(v, X[:, pois])
    sc = score(tmpl, atk[sel][:, pois].astype(np.float64), a, HYP)
    rank = sorted(sc.items(), key=lambda kv: -kv[1])
    got.append(rank[0][0])
    ok = "ok" if rank[0][0] == W2[j] else "NO"
    print(f"{j:>5} {str([WIN[0]+q for q in pois]):>34} {eta[poi]:>8.3f} {frac[poi]:>10.3f} "
          f"{rank[0][0]:>+5d} {W2[j]:>+7d} {rank[0][1]-rank[1][1]:>11,.0f} {ok:>3}")
print(f"\nвосстановлено {sum(g==t for g,t in zip(got,W2))}/6:  {got}   истина {W2}")
