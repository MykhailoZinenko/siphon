"""Разбивка тех же хрупких ячеек по кампаниям.
Подтверждает заметку 16: x3/n0 голосование чинит, x9/n5 — нет (нет ни одной надёжной кампании).
"""
import numpy as np
from collections import Counter
from pathlib import Path
from siphon.config import Config
from siphon.attack import Profile, recover
from siphon.poi import assign_pois, leak_profile
from siphon.templates import build
from siphon.traces import load_inputs, load_manifest, load_traces

CAMPS, REPS = (0, 1, 5), 10
WATCH = [(3, 0), (9, 5)]
cfgs = {c: Config.load(Path(f"siphon/configs/layer1/column{c}.yaml")) for c in (3, 9)}
rng = np.random.default_rng(0)

for pc in CAMPS:
    base = Config.load(Path(f"siphon/configs/column{pc}.yaml"))
    man = load_manifest(base.profiling); ks = sorted(man)
    tr = [load_traces(base.profiling / f"k{k}", 1000, base.window) for k in ks]
    xi = [load_inputs(base.profiling / f"k{k}", 1000, base.prof_column) for k in ks]
    got = {cell: [] for cell in WATCH}
    poi5 = []
    for rep in range(REPS):
        idx = [rng.choice(1000, 1000, replace=True) for _ in ks]
        st = [t[i] for t, i in zip(tr, idx)]
        sx = [x[i] for x, i in zip(xi, idx)]
        prof = {n: leak_profile(st, sx, [man[k][n] for k in ks], base.hypotheses)
                for n in range(6)}
        pts = assign_pois(prof, base.poi_separation)
        poi5.append(pts[5])
        tmpl = [build(np.concatenate([(man[k][n] * sx[i]).astype(np.int64)
                                      for i, k in enumerate(ks)]),
                      np.concatenate([t[:, [pts[n]]].astype(np.float64) for t in st]))
                for n in range(6)]
        P = Profile(base.profiling, base.prof_column, base.window,
                    [[p] for p in pts], tmpl)
        for col, neu in WATCH:
            got[(col, neu)].append(recover(P, cfgs[col])[neu].recovered)
    print(f"кампания x{pc}")
    print(f"  POI нейрона 5: {sorted(Counter(poi5).items())}")
    for cell in WATCH:
        c = Counter(got[cell])
        print(f"  x{cell[0]} нейрон {cell[1]}: "
              + ", ".join(f"{k:+d}x{v}" for k, v in c.most_common()))
    print(flush=True)
