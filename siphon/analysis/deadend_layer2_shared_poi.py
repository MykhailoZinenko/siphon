"""ТУПИК. Первая версия атаки на слой 2 без разноса точек.
Заметка 18: входы 2, 3 и 5 сели на один отсчёт 8189 (финальный аккумулятор).
Та же коллизия, что стоила двух весов в слое 1.
"""
import re, numpy as np
from pathlib import Path
from siphon.templates import build, coverage, score
from siphon.traces import load_traces

D = Path("siphon/data/rebuilt_firmware/layer2")
WIN = (7000, 9500)
tags = np.loadtxt(D / "tags.txt").astype(int)
man = {int(l.split()[0]): [int(v) for v in l.split()[1:]]
       for l in (D / "manifest.txt").read_text().splitlines()}
ks = sorted(man)
prof = [load_traces(D / f"k{k}", 1500, WIN) for k in ks]
atk = load_traces(D / "attack", 1500, WIN)
src = open("siphon/firmware/source/network_config.h").read()
W2 = [int(v) for v in re.search(r"L2_W\[NET_L2\]\[NET_L1\]\s*=\s*\{\s*\{([^}]*)\}", src, re.S).group(1).split(",")]
HYP = tuple(w for w in range(-7, 8) if w != 0)

from siphon.poi import assign_pois

print("шаблонная атака на слой 2 (хотспот по активациям)\n")
data = {}
profiles = {}
for j in range(6):
    sel = tags[:, 0] == j
    a = tags[sel, 1]
    v = np.concatenate([man[k][j] * a for k in ks]).astype(np.int64)
    X = np.concatenate([t[sel] for t in prof]).astype(np.float64)
    m = np.stack([X[v == u].mean(0) for u in np.unique(v)])
    w = np.stack([X[v == u].var(0) for u in np.unique(v)]).mean(0)
    profiles[j] = m.var(0) / np.maximum(w, 1e-12)
    data[j] = (sel, a, v, X)

SEP = 35                       # чтения hidden_q стоят в ~44-60 отсчётах друг от друга
pts = assign_pois(profiles, SEP)          # разнести, отсортировать по времени
print(f"точки в порядке исполнения: {[WIN[0]+p for p in pts]}\n")
print(f"{'вход j':>7} {'POI':>7} {'шабл.':>6} {'покр.':>6} {'w2':>5} {'истина':>7} {'запас':>12} {'':>3}")
got = []
for j in range(6):
    sel, a, v, X = data[j]
    poi = pts[j]
    tmpl = build(v, X[:, [poi]])
    aa = tags[sel, 1]
    sc = score(tmpl, atk[sel][:, [poi]].astype(np.float64), aa, HYP)
    rank = sorted(sc.items(), key=lambda kv: -kv[1])
    got.append(rank[0][0])
    ok = "ok" if rank[0][0] == W2[j] else "NO"
    print(f"{j:>7} {WIN[0]+poi:>7} {len(tmpl):>6} "
          f"{coverage(tmpl, HYP, aa):>6.0%} {rank[0][0]:>+5d} {W2[j]:>+7d} "
          f"{rank[0][1]-rank[1][1]:>12,.0f} {ok:>3}")
print(f"\nвосстановлено {sum(g==t for g,t in zip(got,W2))}/6:  {got}   истина {W2}")
