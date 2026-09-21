"""НЕВАЛИДНАЯ проверка, оставлена как предупреждение.
leave-one-capture-out: латинский квадрат делает захват единственным источником
своих значений v, истинная гипотеза теряет покрытие и получает -inf.
Давала 1-2 из 14 и означала не то, что кажется. Заметка 18.
"""
import numpy as np
from pathlib import Path
from siphon.templates import build, score
from siphon.traces import load_traces

D = Path("siphon/data/rebuilt_firmware/layer2"); WIN = (7000, 9500)
tags = np.loadtxt(D / "tags.txt").astype(int)
man = {int(l.split()[0]): [int(v) for v in l.split()[1:]]
       for l in (D / "manifest.txt").read_text().splitlines()}
ks = sorted(man)
prof = {k: load_traces(D / f"k{k}", 1500, WIN) for k in ks}
HYP = tuple(w for w in range(-7, 8) if w != 0)

def points(j, keep, fmax=0.05, npoi=6, sep=8):
    sel = tags[:, 0] == j; a = tags[sel, 1]
    v = np.concatenate([man[k][j] * a for k in keep]).astype(np.int64)
    wp = np.concatenate([np.full(sel.sum(), man[k][j]) for k in keep])
    X = np.concatenate([prof[k][sel] for k in keep]).astype(np.float64)
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
    return sel, a, v, X, pts

print("leave-one-capture-out: восстанавливаем ИЗВЕСТНЫЙ прошитый вес\n")
print(f"{'вход':>5} " + " ".join(f"k{k:<2}" for k in ks) + "  верно")
for j in range(6):
    row, hit = [], 0
    for kout in ks:
        keep = [k for k in ks if k != kout]
        sel, a, v, X, pts = points(j, keep)
        tm = build(v, X[:, pts])
        aq = tags[sel, 1]
        sc = score(tm, prof[kout][sel][:, pts].astype(np.float64), aq, HYP)
        w = max(sc, key=sc.get)
        ok = w == man[kout][j]; hit += ok
        row.append(f"{'.' if ok else 'x'}  ")
    print(f"{j:>5} " + "".join(row) + f"  {hit}/14")
