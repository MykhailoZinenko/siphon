"""Пересборка профиля бутстрэпом: какие ячейки слоя 1 неустойчивы.
Подтверждает заметку 16: 4 хрупкие ячейки из 60, x9/n5 верна лишь в 10 из 30.
"""
import numpy as np
from collections import Counter
from itertools import product
from pathlib import Path
from siphon.config import Config
from siphon.attack import Profile, recover
from siphon.poi import assign_pois, leak_profile
from siphon.templates import build
from siphon.traces import load_inputs, load_manifest, load_traces

CAMPS = (0, 1, 5)
REPS, SUB = 10, 1000
cfgs = [Config.load(Path(f"siphon/configs/layer1/column{c}.yaml")) for c in range(10)]
rng = np.random.default_rng(0)

pool = []
for pc in CAMPS:
    base = Config.load(Path(f"siphon/configs/column{pc}.yaml"))
    man = load_manifest(base.profiling); ks = sorted(man)
    tr = [load_traces(base.profiling / f"k{k}", 1000, base.window) for k in ks]
    xi = [load_inputs(base.profiling / f"k{k}", 1000, base.prof_column) for k in ks]
    for rep in range(REPS):
        idx = [rng.choice(1000, SUB, replace=True) for _ in ks]
        st = [t[i] for t, i in zip(tr, idx)]
        sx = [x[i] for x, i in zip(xi, idx)]
        prof = {n: leak_profile(st, sx, [man[k][n] for k in ks], base.hypotheses)
                for n in range(6)}
        pts = assign_pois(prof, base.poi_separation)
        tmpl = [build(np.concatenate([(man[k][n] * sx[i]).astype(np.int64)
                                      for i, k in enumerate(ks)]),
                      np.concatenate([t[:, [pts[n]]].astype(np.float64) for t in st]))
                for n in range(6)]
        pool.append((pc, Profile(base.profiling, base.prof_column, base.window,
                                 [[p] for p in pts], tmpl)))
        print(f"  x{pc}#{rep}: POI {pts}", flush=True)

print(f"\n{len(pool)} профилей\n", flush=True)
votes, truth = {}, {}
for pc, pr in pool:
    for cfg in cfgs:
        for r in recover(pr, cfg):
            votes.setdefault((cfg.column, r.neuron), []).append(r.recovered)
            truth[(cfg.column, r.neuron)] = r.truth

shaky = {c: v for c, v in votes.items() if len(set(v)) > 1}
print(f"ячеек, где профили расходятся: {len(shaky)} из 60\n")
for cell in sorted(shaky):
    cnt = Counter(votes[cell])
    print(f"  x{cell[0]} нейрон {cell[1]}: истина {truth[cell]:+d}, верно у "
          f"{cnt[truth[cell]]}/{len(pool)}, голоса "
          + ", ".join(f"{k:+d}x{v}" for k, v in cnt.most_common()))

import random
random.seed(0)
byc = {pc: [i for i, (p, _) in enumerate(pool) if p == pc] for pc in CAMPS}
combos = list(product(*(byc[pc] for pc in CAMPS)))
fails, clean = Counter(), 0
for combo in combos:
    bad = [c for c in votes
           if Counter([votes[c][i] for i in combo]).most_common(1)[0][0] != truth[c]]
    for c in bad:
        fails[c] += 1
    clean += not bad
print(f"\nголосование трёх (по профилю из каждой кампании), {len(combos)} комбинаций:")
for cell, n in fails.most_common():
    print(f"  x{cell[0]} нейрон {cell[1]}: промах в {n}/{len(combos)} ({100*n/len(combos):.0f}%)")
print(f"  чистых 60/60: {clean}/{len(combos)} ({100*clean/len(combos):.0f}%)")

print("\nсколько профилей нужно, чтобы большинство было чистым (1000 жеребьёвок):")
print(f"{'N':>3}  {'чистых 60/60':>13}  {'x3 n0':>7}  {'x9 n5':>7}")
allp = list(range(len(pool)))
for N in (1, 3, 5, 7, 9, 15, 21):
    if N > len(allp):
        break
    clean = 0
    per = Counter()
    for _ in range(1000):
        pick = random.sample(allp, N)
        bad = [c for c in votes
               if Counter([votes[c][i] for i in pick]).most_common(1)[0][0] != truth[c]]
        for c in bad:
            per[c] += 1
        clean += not bad
    print(f"{N:>3}  {clean/10:>12.1f}%  {100-per[(3,0)]/10:>6.1f}%  {100-per[(9,5)]/10:>6.1f}%")
