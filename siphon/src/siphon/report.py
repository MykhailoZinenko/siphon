# Writing results to runs/<run>/report.md + results.json (survives scrollback).
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from .config import Config
from .runs import Run




def write_layer(run: Run, configs: list, results: list[list], panel: list | None = None) -> None:
    # Whole-layer report: one row per neuron, one column per input. `panel` is
    # the set of campaigns voted across (when voting was used).
    columns = [c.column for c in configs]
    per_column = [sum(r.correct for r in rs) for rs in results]
    recovered = sum(per_column)
    total = sum(len(rs) for rs in results)
    neurons = len(results[0]) if results else 0

    run.results.write_text(json.dumps({          # machine-readable results.json
        "generated": datetime.now().isoformat(timespec="seconds"),
        "columns": columns,
        "recovered": recovered,
        "total": total,
        "per_column": dict(zip(map(str, columns), per_column)),
        "profiling": ([str(c.profiling) for c in panel] if panel
                      else {str(c.column): str(c.profiling) for c in configs}),
        "votes": {f"{c.column}.{r.neuron}": r.votes
                  for c, rs in zip(configs, results) for r in rs if r.votes},
        "weights": {
            str(c.column): [r.recovered for r in rs] for c, rs in zip(configs, results)
        },
    }, indent=2))

    sources = panel or [c for c in configs]
    campaigns = sorted({c.profiling.name for c in sources})
    split = [(c.column, r.neuron, r.votes)       # cells where campaigns disagreed
             for c, rs in zip(configs, results) for r in rs
             if r.votes and r.agreement < len(r.votes)]
    lines = [
        "# Layer 1, complete",
        "",
        f"Recovered **{recovered}/{total}** weights across {len(columns)} input columns.",
        "",
        f"- profiling campaigns used: {len(campaigns)} (`" + "`, `".join(campaigns) + "`)"
        + (", majority vote" if panel and len(panel) > 1 else ""),
        f"- attack sets: {len(configs)}, one per input column",
        "",
        "## Recovered weight matrix",
        "",
        "Cells show the recovered weight; a mismatch with the true weight is marked.",
        "",
        "| neuron | " + " | ".join(f"x{c}" for c in columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for n in range(neurons):                      # one table row per neuron
        cells = []
        for rs in results:
            r = rs[n]
            cells.append(f"{r.recovered:+d}" if r.correct
                         else f"**{r.recovered:+d}** (true {r.truth:+d})")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")

    lines += ["", "## Per column", "", "| column | recovered |", "|---|---|"]
    for cfg, ok in zip(configs, per_column):
        lines.append(f"| {cfg.column} | {ok}/{neurons} |")

    if split:                                     # only when voting cleared a disagreement
        lines += ["", "## Where the campaigns disagreed", "",
                  "One campaign is not quite enough: the weakest neuron's leak profile is",
                  "a plateau, so `argmax` lands arbitrarily and occasionally flips a weight.",
                  "Independent campaigns fail on different cells, so the majority clears it.",
                  "", "| cell | votes | taken |", "|---|---|---|"]
        for col, n, votes in split:
            taken = max(set(votes), key=votes.count)
            lines.append(f"| x{col} neuron {n} | "
                         + ", ".join(f"{v:+d}" for v in votes) + f" | {taken:+d} |")

    if recovered < total:                         # list any weight not recovered
        lines += ["", "## Not recovered", ""]
        for cfg, rs in zip(configs, results):
            for r in rs:
                if not r.correct:
                    lines.append(f"- column {cfg.column}, neuron {r.neuron}: "
                                 f"expected {r.truth:+d}, got {r.recovered:+d} "
                                 f"(coverage {r.coverage:.0%}, margin {r.margin:,.0f})")

    run.report.write_text("\n".join(lines) + "\n")


def write_transfer(run: Run, columns: list[int], grid: dict) -> None:
    # Cross-profiling matrix: every campaign scored against every attack set.
    run.results.write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "columns": columns,
        "grid": {f"{p}->{a}": ok for (p, a), (ok, _) in grid.items()},
    }, indent=2))

    lines = [
        "# Template transfer between input columns",
        "",
        "Each campaign profiles on one input column; each attack set is captured on",
        "one input column. A cell is the number of weights recovered when the row's",
        "templates score the column's attack traces. Off-diagonal cells are transfers.",
        "",
        "| profiling \\ attack | " + " | ".join(f"x{c}" for c in columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for p in columns:                             # one row per profiling column
        cells = [f"{grid[(p, a)][0]}/6" for a in columns]
        lines.append(f"| x{p} | " + " | ".join(cells) + " |")

    lines += ["", "## Margins", "",
              "Distance between the winning hypothesis and the runner-up, per neuron.",
              "", "| profiling | attack | margins |", "|---|---|---|"]
    for (p, a), (_, res) in grid.items():
        lines.append(f"| x{p} | x{a} | " + ", ".join(f"{r.margin:,.0f}" for r in res) + " |")
    run.report.write_text("\n".join(lines) + "\n")


def write_activation(run: Run, fits: list, profile_dir, target_dir) -> None:
    # Layer-1 bias and requant scale, one row per neuron.
    run.results.write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "profile": str(profile_dir),
        "target": str(target_dir),
        "neurons": [
            {"neuron": f.neuron, "bias": f.bias, "inv_scale": f.inv_scale,
             "scale": f.scale, "pois": f.pois}
            for f in fits
        ],
    }, indent=2))

    lines = [
        "# Layer 1 bias and requantisation scale",
        "",
        "`a = clamp(floor((acc + b) * s + 1/2), 0, 7)` - two numbers per neuron.",
        "",
        f"- profile: `{Path(profile_dir).name}` (clone, chosen bias and scale)",
        f"- target: `{Path(target_dir).name}`",
        "- profile and target share one input set; the saturated activation",
        "  classes do not transfer between different sweeps.",
        "",
        "| neuron | POIs | 1/s | scale | bias |",
        "|---|---|---|---|---|",
    ]
    for f in fits:
        lines.append(f"| {f.neuron} | {len(f.pois)} | {f.inv_scale:.2f} | "
                     f"{f.scale:.5f} | {f.bias:+d} |")
    run.report.write_text("\n".join(lines) + "\n")


def write_layer2(run: Run, results: list, data, dropped: list) -> None:
    # Layer-2 weights, one row per input.
    run.results.write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "campaign": str(data),
        "dropped_captures": dropped,
        "weights": [r.recovered for r in results],
        "inputs": [{"input": r.index, "recovered": r.recovered, "true": r.truth,
                    "margin": r.margin, "pois": r.pois, "coverage": r.coverage}
                   for r in results],
    }, indent=2))
    ok = sum(bool(r.correct) for r in results)
    lines = [
        "# Layer 2 weights",
        "",
        f"Recovered **{ok}/{len(results)}** weights.",
        "",
        f"- campaign: `{Path(data).name}`",
        f"- glitched captures dropped: {dropped if dropped else 'none'}",
        "- hot spot on the hidden activations; templates keyed on `v = w2 * a`",
        "- correlation is unusable here: `HW(2x) = HW(x)` makes `+3` and `+6` collinear",
        "",
        "| input | POIs | templates | coverage | recovered | true | margin |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r.index} | {len(r.pois)} | {r.templates} | {r.coverage:.0%} | "
                     f"{r.recovered:+d} | {'-' if r.truth is None else f'{r.truth:+d}'} | "
                     f"{r.margin:,.0f} |")
    run.report.write_text("\n".join(lines) + "\n")


def write_extraction(run: Run, model, card: dict | None, truth) -> None:
    # The whole recovered network, plus the comparison to truth if one is supplied.
    payload = {"generated": datetime.now().isoformat(timespec="seconds"),
               "model": json.loads(model.to_json())}
    if card:
        payload["score"] = card
        payload["truth"] = str(truth)
    run.results.write_text(json.dumps(payload, indent=2))

    lines = ["# Extracted network", "",
             "```json", model.to_json(), "```", ""]
    if card:
        lines += [
            "## Against the known network",
            "",
            f"- layer-1 weights: **{card['l1_weights']}/{card['l1_weights_total']}**",
            f"- layer-1 biases exact: {card['l1_bias_exact']}/6",
            f"- layer-2 weights: **{card['l2_weights']}/{card['l2_weights_total']}**",
            f"- layer-2 bias exact: {'yes' if card['l2_bias_exact'] else 'no'}",
            "",
            f"On {card['samples']} random int8 inputs:",
            "",
            f"- hidden vector exact: **{card['hidden_vector_exact']:.3%}**",
            f"- individual activations: {card['activations_exact']:.3%}",
            f"- **decision agreement: {card['decision_agreement']:.3%}**",
            "",
            "The layer-2 scale is not recovered and is not needed: requantisation",
            "is monotone, so the decision follows the sign of `acc2 + b2`.",
        ]
    run.report.write_text("\n".join(lines) + "\n")
