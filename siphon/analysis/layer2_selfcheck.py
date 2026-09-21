"""ВАЛИДНАЯ самопроверка слоя 2: сплит трасс ВНУТРИ захватов.
Заметка 18: 83/84. Все веса остаются в профиле, поэтому покрытие не рушится.
"""
import numpy as np
from pathlib import Path
from siphon.templates import build, coverage, score
from siphon.traces import load_traces

D = Path("siphon/data/rebuilt_firmware/layer2"); WIN = (7000, 9500)
tags = np.loadtxt(D / "tags.txt").astype(int)
man = {int(l.split()[0]): [int(v) for v in l.split()[1:]]
       for l in (D / "manifest.txt").read_text().splitlines()}
ks = sorted(man)
prof = {k: load_traces(D / f"k{k}", 1500, WIN) for k in ks}
HYP = tuple(w for w in range(-7, 8) if w != 0)
rng = np.random.default_rng(0)
half = rng.random(1500) < 0.5            # обучение / проверка внутри каждого захвата

def analyse(j, fmax=0.05, npoi=6, sep=8):
    sel = tags[:, 0] == j
    a_all = tags[sel, 1]
    tr = half[sel]
    v = np.concatenate([man[k][j] * a_all[tr] for k in ks]).astype(np.int64)
    wp = np.concatenate([np.full(tr.sum(), man[k][j]) for k in ks])
    X = np.concatenate([prof[k][sel][tr] for k in ks]).astype(np.float64)
    tot = X.var(0) * len(X); ssw = np.zeros(X.shape[1]); rw = np.zeros(X.shape[1])
    for u in np.unique(v):
        r = v == u
        if r.sum() < 8: continue
        b = X[r]; mu = b.mean(0); ssw += ((b - mu) ** 2).sum(0)
        for ww in np.unique(wp[r]):
            sub = b[wp[r] == ww]
            if len(sub) < 4: continue
            rw += len(sub) * (sub.mean(0) - mu) ** 2
    eta = 1 - ssw / np.maximum(tot, 1e-12); frac = rw / np.maximum(ssw, 1e-12)
    cand = np.where((eta > 0.5) & (frac < fmax))[0]
    if not len(cand): cand = np.array([int(np.argmax(eta))])
    pts = []
    for t in cand[np.argsort(-eta[cand])]:
        if all(abs(int(t) - q) > sep for q in pts): pts.append(int(t))
        if len(pts) == npoi: break
    return sel, a_all, v, X, pts

print("проверка на отложенных ТРАССАХ (все веса остаются в профиле)\n")
print(f"{'вход':>5}  " + " ".join(f"{w:>3}" for w in
      [man[k][0] for k in ks]).replace(str(man[ks[0]][0]), str(man[ks[0]][0])) + "   верно  покрытие")
for j in range(6):
    sel, a_all, v, X, pts = analyse(j)
    tm = build(v, X[:, pts])
    te = ~half[sel]
    hits, marks = 0, []
    for k in ks:
        q = prof[k][sel][te][:, pts].astype(np.float64)
        sc = score(tm, q, a_all[te], HYP)
        w = max(sc, key=sc.get)
        ok = w == man[k][j]; hits += ok
        marks.append(" . " if ok else " x ")
    print(f"{j:>5}  " + "".join(marks) + f"   {hits}/14   {coverage(tm, HYP, a_all[te]):.0%}")
