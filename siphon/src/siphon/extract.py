# End-to-end extraction: traces in, whole 10-6-1 network out.
# Four stages, each feeding the next (each needs the inputs of the layer it
# attacks, known only once everything upstream is recovered):
#   1. layer-1 weights        templates over a hot-spot campaign
#   2. layer-1 bias and scale  activation templates, likelihood over (b, s)
#   3. layer-2 weights         hot spot on the activations
#   4. layer-2 bias            templates over a spread of flashed biases
# Nothing here reads the target's parameters; truth, if given, only scores after.
# (The campaign must be captured on the target's own model, note 24.)
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import activation as act
from . import layer2 as l2
from .attack import profile, recover, vote
from .config import Config
from .progress import Reporter
from .sweep import craft, hidden_hotspot
from .traces import load_manifest, load_traces

WINDOW = (0, 10000)                            # layer-1 crop
FULL = (0, 24430)                              # whole trace (activation lives late)
L2_WINDOW = (7000, 9500)                       # layer-2 crop
POI1 = [4233, 4813, 5433, 6053, 6717, 7333]    # per-neuron accumulator points
L2_READS = (7750, 8400)                        # where layer 2 reads hidden_q
CLONE_M0, CLONE_SHIFT = 687194767, 4          # 1/s = 50 on the clone
L2_BIASES = [-24, -12, 0, 12, 24]              # spread of biases the campaign flashes


@dataclass
class Model:
    # The recovered network. No layer-2 scale: a decision does not need it.
    l1_w: list[list[int]]
    l1_bias: list[int]
    l1_inv_scale: list[float]
    l2_w: list[int]
    l2_bias: int

    def to_json(self) -> str:
        return json.dumps({
            "L1_W": self.l1_w, "L1_B": self.l1_bias,
            "L1_inv_scale": self.l1_inv_scale,
            "L2_W": self.l2_w, "L2_B": [self.l2_bias],
            "note": "L2 scale not recovered; it does not affect decisions",
        }, indent=2)

    def activations(self, x: np.ndarray) -> np.ndarray:
        # Hidden layer under the recovered params: clamp(round((acc+b)/(1/s)),0,7).
        w = np.array(self.l1_w)
        return np.stack([np.clip(np.floor((x @ w[o] + b) / inv + 0.5), 0, 7)
                         for o, (inv, b) in enumerate(zip(self.l1_inv_scale, self.l1_bias))],
                        1).astype(int)

    def logit(self, x: np.ndarray) -> np.ndarray:
        # Output up to a positive scale; the decision is its sign.
        return self.activations(x) @ np.array(self.l2_w) + self.l2_bias


def _column_config(campaign: Path, attack: Path, column: int, header: Path,
                   profiling_column: int, traces: int) -> Config:
    # Build a Config for one column without a YAML file.
    return Config(name=f"column {column}", column=column, neurons=6,
                  profiling=campaign, attack=attack, firmware_config=header,
                  profiling_column=profiling_column, window=WINDOW,
                  attack_traces=traces)


def layer1_weights(campaigns: list[tuple[Path, int]], attacks: dict[int, Path],
                   header: Path, traces: int, reporter: Reporter) -> list[list[int]]:
    # Recover every column (voting across campaigns); returns the neurons x inputs matrix.
    # One wrong weight cascades (shifts acc, collapses the activation fit), so the
    # weakest neuron's plateau is cleared by voting: 1 campaign 59/60, 3 give 60/60.
    cfgs = {c: _column_config(campaigns[0][0], p, c, header, campaigns[0][1], traces)
            for c, p in sorted(attacks.items())}
    profiles = [profile(_column_config(path, next(iter(attacks.values())), 0, header,
                                       column, traces), reporter)
                for path, column in campaigns]           # profile each campaign once
    columns = {}
    for c, cfg in cfgs.items():
        reporter.stage(f"column {c}")
        results = (vote(profiles, cfg, reporter) if len(profiles) > 1
                   else recover(profiles[0], cfg, reporter))
        columns[c] = [r.recovered for r in results]
        shaky = [r.neuron for r in results if r.votes and r.agreement < len(r.votes)]
        if shaky:
            reporter.detail(f"campaigns disagreed on neuron(s) {shaky}")
    return [[columns[c][n] for c in sorted(columns)] for n in range(6)]   # transpose to (neuron, input)


def layer1_nonlinearity(profile_dir: Path, target_dir: Path, weights: np.ndarray,
                        traces: int, reporter: Reporter) -> tuple[list[int], list[float]]:
    # Recover (bias, 1/scale) per neuron from the clone profile vs the target sweep.
    labels = np.loadtxt(profile_dir / "activations.txt").astype(int)   # known a on the clone
    inputs = np.loadtxt(profile_dir / "inputs.txt").astype(int)[:traces]
    if not (inputs == np.loadtxt(target_dir / "inputs.txt").astype(int)[:traces]).all():
        raise ValueError("profile and target sweeps use different inputs")  # must match
    acc = (inputs @ weights.T).astype(float)             # accumulators from recovered W1

    reporter.stage("loading sweeps")
    prof = load_traces(profile_dir, traces, FULL, reporter).astype(np.float64)
    tgt = load_traces(target_dir, traces, FULL, reporter).astype(np.float64)

    inv_grid = np.arange(15.0, 80.01, 0.05)              # candidate 1/scale
    bias_grid = np.arange(-40.0, 40.01, 1.0)             # candidate bias
    biases, scales = [], []
    for n in reporter.track(range(weights.shape[0]), "neurons", total=weights.shape[0]):
        allowed = np.zeros(prof.shape[1], bool)          # restrict POI search to
        allowed[POI1[n] + 5:POI1[n] + 600] = True        #   after this neuron's acc
        allowed[L2_READS[0]:L2_READS[1]] = True          #   and layer 2's reads of it
        pois = act.leak_points(prof, labels[:traces, n], acc[:, n], allowed)
        inv, bias = act.fit_neuron(prof[:, pois], labels[:traces, n], tgt[:, pois],
                                   acc[:, n], inv_grid, bias_grid)
        biases.append(int(bias))
        scales.append(float(inv))
    return biases, scales


def layer2_weights(data: Path, traces: int, reporter: Reporter) -> list[int]:
    # Recover the six layer-2 weights from the hidden hot-spot campaign.
    tags = np.loadtxt(data / "tags.txt").astype(int)     # (hot neuron, activation) per trace
    manifest = load_manifest(data)                       # flashed L2_W per capture
    reporter.stage("loading layer-2 campaign")
    prof = {k: load_traces(data / f"k{k}", traces, L2_WINDOW, reporter)
            for k in sorted(manifest)}
    atk = load_traces(data / "attack", traces, L2_WINDOW, reporter)
    reporter.stage("scoring hypotheses")
    results, kept = l2.recover(prof, atk, tags, manifest, tuple(range(-7, 8)))
    dropped = sorted(set(manifest) - set(kept))          # glitched captures thrown out
    if dropped:
        reporter.detail(f"dropped glitched captures: {dropped}")
    return [r.recovered for r in results]


def layer2_bias(data: Path, target: Path, acc2: np.ndarray, traces: int,
                reporter: Reporter) -> int:
    # Recover b2 from the bias campaign (templates of acc2 + b2).
    flashed = {int(l.split()[0]): int(l.split()[1])
               for l in (data / "manifest.txt").read_text().splitlines()}
    reporter.stage("loading bias campaign")
    prof = {i: load_traces(data / f"b{i}", traces, L2_WINDOW, reporter)
            for i in sorted(flashed)}
    atk = load_traces(target, traces, L2_WINDOW, reporter)
    reporter.stage("scoring hypotheses")
    bias, _ = l2.recover_bias(prof, flashed, atk, acc2)
    return bias


def score(model: Model, truth: dict, samples: int = 50000, seed: int = 1) -> dict:
    # Compare the recovered model to a known network (scoring only, never a decision).
    rng = np.random.default_rng(seed)
    x = rng.integers(-128, 128, (samples, 10))           # random int8 test inputs
    w1 = np.array(truth["L1_W"]); b1 = np.array(truth["L1_B"])
    m0 = np.array(truth["L1_M0"]); n1 = np.array(truth["L1_N"])
    w2 = np.array(truth["L2_W"]); b2 = int(truth["L2_B"][0])

    def rq(a, m, n):                                      # the firmware's requant
        return (a.astype(np.int64) * int(m) + (1 << (31 + int(n) - 1))) >> (31 + int(n))

    true_a = np.stack([np.clip(rq(x @ w1[o] + b1[o], m0[o], n1[o]), 0, 7)
                       for o in range(len(w1))], 1)       # true hidden vector
    true_pre = true_a @ w2 + b2                           # true pre-requant logit
    ours_a = model.activations(x)
    ours_pre = model.logit(x)
    return {
        "l1_weights": int((np.array(model.l1_w) == w1).sum()),
        "l1_weights_total": int(w1.size),
        "l1_bias_exact": int(sum(int(a) == int(b) for a, b in zip(model.l1_bias, b1))),
        "l2_weights": int((np.array(model.l2_w) == w2).sum()),
        "l2_weights_total": int(w2.size),
        "l2_bias_exact": bool(model.l2_bias == b2),
        "hidden_vector_exact": float((ours_a == true_a).all(1).mean()),
        "activations_exact": float((ours_a == true_a).mean()),
        "decision_agreement": float((np.sign(ours_pre) == np.sign(true_pre)).mean()),  # the headline
        "samples": samples,
    }
