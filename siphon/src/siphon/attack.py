# Layer-1 recovery: profile a campaign once, then recover any column with it.
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .config import Config
from .progress import Reporter
from .poi import assign_pois, extra_pois, leak_profile
from .templates import Template, build, coverage, score
from .traces import ground_truth, load_inputs, load_manifest, load_traces


@dataclass
class Profile:
    # The reusable, expensive half of the attack: POIs + templates per neuron.
    # Templates are indexed by v = w*x and the leak point depends on the neuron,
    # not the input column, so one Profile recovers any column (one campaign/layer).
    source: Path
    column: int                              # column the campaign was captured on
    window: tuple[int, int]
    pois: list[list[int]]                    # per neuron
    templates: list[dict[int, Template]]     # per neuron

    @property
    def neurons(self) -> int:
        return len(self.templates)


@dataclass
class NeuronResult:
    neuron: int
    poi: list[int]
    truth: int
    recovered: int
    margin: float                            # gap to the runner-up hypothesis
    templates: int
    coverage: float
    votes: list[int] = field(default_factory=list)   # one per campaign, when voting

    @property
    def agreement(self) -> int:
        # How many campaigns agree with the taken answer (1 when not voting).
        return sum(v == self.recovered for v in self.votes) if self.votes else 1

    @property
    def correct(self) -> bool:
        return self.recovered == self.truth


def profile(config: Config, reporter: Reporter | None = None) -> Profile:
    # Build POIs and templates from a profiling campaign (no attack traces yet).
    reporter = reporter or Reporter()
    manifest = load_manifest(config.profiling)           # capture -> flashed weights
    captures = sorted(manifest)

    reporter.stage(f"loading {len(captures)} profiling captures "
                   f"from {config.profiling.name}")
    traces, inputs = [], []
    for k in captures:
        folder = config.profiling / f"k{k}"
        traces.append(load_traces(folder, config.profiling_traces, config.window, reporter))
        inputs.append(load_inputs(folder, config.profiling_traces, config.prof_column))

    reporter.stage("locating each neuron's multiply")
    profiles = {                                          # SNR profile per neuron
        n: leak_profile(traces, inputs, [manifest[k][n] for k in captures], config.hypotheses)
        for n in range(config.neurons)
    }
    primary = assign_pois(profiles, config.poi_separation)   # one POI per neuron, in order
    reporter.detail(f"points of interest, in execution order: {primary}")

    reporter.stage("building templates")
    pois, templates = [], []
    for neuron in reporter.track(range(config.neurons), "neurons", total=config.neurons):
        chosen = (primary[neuron:neuron + 1] if config.pois_per_neuron == 1
                  else extra_pois(profiles[neuron], primary[neuron],
                                  config.pois_per_neuron, config.poi_separation))
        values = np.concatenate([(manifest[k][neuron] * inputs[i]).astype(np.int64)   # v = w*x
                                 for i, k in enumerate(captures)])
        leakage = np.concatenate([t[:, chosen].astype(np.float64) for t in traces])   # trace at POIs
        pois.append(chosen)
        templates.append(build(values, leakage))

    return Profile(source=config.profiling, column=config.prof_column,
                   window=config.window, pois=pois, templates=templates)


def recover(profile: Profile, config: Config,
            reporter: Reporter | None = None) -> list[NeuronResult]:
    # Score one column's attack set against an already-built profile.
    reporter = reporter or Reporter()
    if profile.neurons != config.neurons:
        raise ValueError(f"profile covers {profile.neurons} neurons, "
                         f"config asks for {config.neurons}")
    if profile.window != config.window:                  # POIs are indices into the window
        raise ValueError(f"profile was built on window {profile.window}, "
                         f"config uses {config.window}")

    reporter.stage(f"loading attack traces from {config.attack.name}")
    traces = load_traces(config.attack, config.attack_traces, config.window, reporter)
    inputs = load_inputs(config.attack, config.attack_traces, config.column)
    truth = ground_truth(config.firmware_config, config.column)     # scoring only

    reporter.stage("scoring hypotheses")
    results = []
    for neuron in range(config.neurons):
        pois = profile.pois[neuron]
        templates = profile.templates[neuron]
        scores = score(templates, traces[:, pois].astype(np.float64),
                       inputs, config.hypotheses)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])      # best likelihood first
        results.append(NeuronResult(
            neuron=neuron,
            poi=pois,
            truth=truth[neuron],
            recovered=ranked[0][0],
            margin=ranked[0][1] - ranked[1][1],
            templates=len(templates),
            coverage=coverage(templates, config.hypotheses, inputs),
        ))
    return results


def run(config: Config, reporter: Reporter | None = None) -> list[NeuronResult]:
    # Profile then recover in one call.
    return recover(profile(config, reporter), config, reporter)


def vote(profiles: list[Profile], config: Config,
         reporter: Reporter | None = None) -> list[NeuronResult]:
    # Recover with several campaigns and take the majority per neuron.
    # The weakest neuron's SNR profile is a plateau, so one campaign's argmax can
    # flip a weight (and the margin does not reveal which). Independent campaigns
    # fail on different cells, so a constant number of them clears it.
    per = [recover(p, config, reporter) for p in profiles]
    out = []
    for group in zip(*per):                              # same neuron across campaigns
        votes = [r.recovered for r in group]
        winner = Counter(votes).most_common(1)[0][0]
        best = max((r for r in group if r.recovered == winner), key=lambda r: r.margin)
        out.append(replace(best, votes=votes))
    return out
