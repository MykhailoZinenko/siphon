# Command-line interface. capture-* touches the device; everything else reads disk.
# Every command: header -> numbered stages -> results table -> artefact line (ui.Session).
from __future__ import annotations

import json
from pathlib import Path

import typer

from . import paths, report, ui
from .config import Config

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Template-based extraction of quantized neural-network weights")

HYPOTHESES = tuple(range(-7, 8))            # zero is a legal weight
WINDOW_FULL = (0, 24430)
WINDOW_L1 = (0, 10000)
WINDOW_L2 = (7000, 9500)
CLONE_M0, CLONE_SHIFT = 687194767, 4        # 1/s = 50 while profiling activations
L2_BIASES = [-24, -12, 0, 12, 24]
# zero included: without a template for v = 0 a zero weight cannot be recovered
L2_WEIGHTS = [0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7]


# ---------------------------------------------------------------- helpers ---

def _configs(given: list[Path]) -> list[tuple[Path, Config]]:
    """Expand directories to the YAML files inside, ordered by input column."""
    files: list[Path] = []
    for item in given:
        p = paths.existing(item, "config")
        files.extend(sorted(p.glob("*.yaml")) if p.is_dir() else [p])
    if not files:
        raise typer.BadParameter("no configs found")
    return sorted(((f, Config.load(f)) for f in files), key=lambda pair: pair[1].column)


def _recovered(results: Path):
    from .sweep import recovered_weights
    return recovered_weights(paths.existing(results, "layer-1 results"))


def _nonlinearity(params: Path):
    fit = json.loads(paths.existing(params, "activation results").read_text())["neurons"]
    ordered = sorted(fit, key=lambda n: n["neuron"])
    return [(n["inv_scale"], n["bias"]) for n in ordered]


def _layer2_weights(results: Path):
    import numpy as np
    return np.array(json.loads(paths.existing(results, "layer-2 results").read_text())["weights"])


def _header() -> Path:
    from .config import PACKAGE_ROOT
    return PACKAGE_ROOT / "firmware" / "source" / "network_config.h"


# --------------------------------------------------------------- capture ----

@app.command("capture-campaign")
def capture_campaign(config: Path = typer.Option(..., "--config", "-c"),
                     out: Path = typer.Option(None, "--out",
                                              help="override the config's campaign directory"),
                     count: int = typer.Option(1000, "--count", "-n")):
    """Layer-1 profiling campaign: 8 flashes, the whole column varied each time (so v=w*x is covered)."""
    from .campaign import column_for
    from .capture import Step, execute

    from .campaign import CAPTURES

    cfg = Config.load(paths.existing(config, "config"))
    where = paths.out(out) if out else cfg.profiling
    plan, manifest = [], {}
    for k in range(CAPTURES):
        values = column_for(k, cfg.neurons)
        manifest[k] = values
        rows = tuple(("L1_W", o, _row_with(cfg, o, cfg.column, values[o]))
                     for o in range(cfg.neurons))
        plan.append(Step(label=f"capture {k}", out=where / f"k{k}",
                         count=count, column=cfg.column, rows=rows,
                         note=f"column {cfg.column} = {values}"))

    with ui.Session("layer-1 profiling campaign",
                    subtitle=[f"column {cfg.column}, {CAPTURES} flashes x {count} traces",
                              f"into {where}"],
                    sources=paths.existing(config), hardware=True,
                    total_stages=CAPTURES) as s:
        execute(plan, s)
        (where / "manifest.txt").write_text(
            "".join(f"{k} {' '.join(map(str, v))}\n" for k, v in manifest.items()))
        s.done("campaign complete", artefact=where)


def _row_with(cfg: Config, neuron: int, column: int, value: int) -> tuple[int, ...]:
    """One row of L1_W with a single entry replaced, read from the header."""
    import re
    text = cfg.firmware_config.read_text()
    block = re.search(r"L1_W\[NET_L1\]\[NET_INPUTS\]\s*=\s*\{(.*?)\};", text, re.S).group(1)
    row = [int(v) for v in re.findall(r"\{([^}]*)\}", block)[neuron].split(",")]
    row[column] = value
    return tuple(row)


@app.command("capture-attack")
def capture_attack(config: Path = typer.Option(..., "--config", "-c"),
                   count: int = typer.Option(1000, "--count", "-n")):
    """Layer-1 attack sets: real weights in place, one flash, only the input column changes."""
    from .capture import Step, execute

    loaded = _configs([config])
    pending = [c for _, c in loaded if not paths.has_traces(c.attack)]
    skipped = [c for _, c in loaded if paths.has_traces(c.attack)]

    with ui.Session("layer-1 attack sets",
                    subtitle=[f"{len(pending)} columns x {count} traces, one flash"],
                    sources=[p for p, _ in loaded], hardware=bool(pending),
                    total_stages=1 if pending else None) as s:
        if skipped:
            s.detail("already captured: " + ", ".join(f"column {c.column}" for c in skipped))
        if not pending:
            s.done("nothing to capture", ok=True)
            return
        execute([Step(label=f"column {c.column}", out=c.attack, count=count,
                      column=c.column) for c in pending], s)
        s.done(f"{len(pending)} attack sets captured")


@app.command("capture-sweep")
def capture_sweep(results: Path = typer.Option(..., "--from", help="layer-1 results.json"),
                  out: Path = typer.Option(..., "--out"),
                  count: int = typer.Option(3000, "--count", "-n"),
                  span: str = typer.Option("-60,420", "--span")):
    """Accumulator sweep (real params): inputs that drive every neuron across all 8 activations."""
    from .capture import Step, execute
    from .sweep import craft

    out = paths.out(out)
    lo, hi = (int(v) for v in span.split(","))
    vectors, accs = craft(_recovered(results), count, (lo, hi))
    reach = ", ".join(f"n{o}:{accs[:, o].min()}..{accs[:, o].max()}"
                      for o in range(accs.shape[1]))

    with ui.Session("accumulator sweep",
                    subtitle=[f"{count} traces into {out}", f"acc {reach}"],
                    sources=paths.existing(results), hardware=True,
                    total_stages=1) as s:
        execute([Step(label="sweep", out=out, count=count, vectors=vectors)], s)
        s.done("sweep captured", artefact=out)


@app.command("capture-activation")
def capture_activation(results: Path = typer.Option(..., "--from"),
                       out: Path = typer.Option(..., "--out"),
                       count: int = typer.Option(3000, "--count", "-n"),
                       m0: int = typer.Option(CLONE_M0, "--m0"),
                       shift: int = typer.Option(CLONE_SHIFT, "--shift")):
    """Activation profile on the clone with known (b=0, s): labels a on every trace. Same inputs as capture-sweep."""
    import numpy as np

    from .capture import Step, execute
    from .sweep import craft

    out = paths.out(out)
    vectors, accs = craft(_recovered(results), count)
    neurons = accs.shape[1]
    patches = (("L1_B", (0,) * neurons), ("L1_M0", (m0,) * neurons),
               ("L1_N", (shift,) * neurons))

    with ui.Session("activation profile (clone)",
                    subtitle=[f"{count} traces into {out}",
                              f"b = 0, 1/s = {2.0 ** (31 + shift) / m0:.2f}"],
                    sources=paths.existing(results), hardware=True,
                    total_stages=1) as s:
        execute([Step(label="profile", out=out, count=count, vectors=vectors,
                      arrays=patches,
                      note=f"clone: b = 0, 1/s = {2.0 ** (31 + shift) / m0:.2f}")], s)
        labels = np.clip((accs.astype(np.int64) * m0 + (1 << (31 + shift - 1)))
                         >> (31 + shift), 0, 7)
        np.savetxt(out / "activations.txt", labels, fmt="%d")
        (out / "profile.json").write_text(json.dumps(
            {"bias": 0, "m0": m0, "shift": shift,
             "inv_scale": 2.0 ** (31 + shift) / m0}, indent=2))
        s.done("activation profile captured", artefact=out)


@app.command("capture-layer2")
def capture_layer2(results: Path = typer.Option(..., "--from"),
                   params: Path = typer.Option(..., "--params"),
                   out: Path = typer.Option(..., "--out"),
                   count: int = typer.Option(1000, "--count", "-n"),
                   levels: int = typer.Option(4, "--levels"),
                   repeats: int = typer.Option(2, "--repeats"),
                   seed: int = typer.Option(0, "--seed")):
    """Layer-2 campaign: hidden hot spot, 14 weight values per input in random orders, then the attack set."""
    import numpy as np

    from .capture import Step, execute
    from .sweep import hidden_hotspot

    out = paths.out(out)
    weights = _recovered(results)
    vectors, tags = hidden_hotspot(weights, _nonlinearity(params), count, levels=levels)
    rng = np.random.default_rng(seed)
    schedule = np.stack([rng.permutation(np.tile(L2_WEIGHTS, repeats))
                         for _ in range(weights.shape[0])], axis=1)
    pairs = sorted({tuple(t) for t in tags})

    plan = [Step(label=f"capture {k}", out=out / f"k{k}", count=count, vectors=vectors,
                 rows=(("L2_W", 0, tuple(int(v) for v in row)),),
                 note=f"L2_W = {[int(v) for v in row]}")
            for k, row in enumerate(schedule)]
    plan.append(Step(label="attack set", out=out / "attack", count=count, vectors=vectors))

    with ui.Session("layer-2 campaign",
                    subtitle=[f"{len(schedule)} flashes + attack set, {count} traces each",
                              f"{len(pairs)} reachable (input, activation) pairs"],
                    sources=paths.existing(results), hardware=True,
                    total_stages=len(schedule) + 1) as s:
        out.mkdir(parents=True, exist_ok=True)
        np.savetxt(out / "tags.txt", tags, fmt="%d")
        execute(plan, s)
        (out / "manifest.txt").write_text("".join(
            f"{k} {' '.join(str(int(v)) for v in row)}\n" for k, row in enumerate(schedule)))
        s.done("layer-2 campaign complete", artefact=out)


@app.command("capture-l2bias")
def capture_l2bias(results: Path = typer.Option(..., "--from"),
                   out: Path = typer.Option(..., "--out"),
                   count: int = typer.Option(3000, "--count", "-n"),
                   values: str = typer.Option(",".join(map(str, L2_BIASES)), "--values")):
    """Layer-2 bias campaign: a spread of known biases flashed on the clone (one bias would not tile acc2+b2)."""
    from .capture import Step, execute
    from .sweep import craft

    out = paths.out(out)
    biases = [int(v) for v in values.split(",")]
    vectors, _ = craft(_recovered(results), count)
    plan = [Step(label=f"bias {b:+d}", out=out / f"b{i}", count=count, vectors=vectors,
                 arrays=(("L2_B", (b,)),), note=f"L2_B = {b:+d}")
            for i, b in enumerate(biases)]

    with ui.Session("layer-2 bias campaign",
                    subtitle=[f"{len(biases)} flashes x {count} traces, biases {biases}"],
                    sources=paths.existing(results), hardware=True,
                    total_stages=len(biases)) as s:
        out.mkdir(parents=True, exist_ok=True)
        execute(plan, s)
        (out / "manifest.txt").write_text(
            "".join(f"{i} {b}\n" for i, b in enumerate(biases)))
        s.done("bias campaign complete", artefact=out)


@app.command("new-target")
def new_target(seed: int = typer.Option(..., "--seed"),
               truth: Path = typer.Option(..., "--truth"),
               flash_device: bool = typer.Option(True, "--flash/--no-flash")):
    """Write a fresh random network into the device (becomes the new pristine); truth saved for scoring only."""
    from .capture import session
    from .target import flash, generate, save_truth

    truth = paths.out(truth)
    model = generate(seed)
    with ui.Session(f"new target network (seed {seed})",
                    subtitle=[f"replaces the network on the device",
                              f"truth -> {truth}"],
                    sources=_header(), hardware=flash_device, total_stages=2) as s:
        s.stage("writing the header and rebuilding")
        hex_path = flash(model, s.detail)
        truth.parent.mkdir(parents=True, exist_ok=True)
        save_truth(model, truth)
        if flash_device:
            s.stage("flashing")
            with session(hex_path, s.detail):
                pass
        s.table(["parameter", "value"], [
            ["L1_W row 0", model["L1_W"][0]],
            ["L1_B", model["L1_B"]],
            ["L2_W", model["L2_W"]],
            ["L2_B", model["L2_B"]],
            ["1/scale", [round(v, 1) for v in model["inv_scale"]]],
        ])
        s.done("target written", artefact=truth)


@app.command("train-target")
def train_target(truth: Path = typer.Option(..., "--truth"),
                 seeds: int = typer.Option(10, "--seeds",
                                           help="train this many, flash the most accurate"),
                 flash_device: bool = typer.Option(True, "--flash/--no-flash")):
    """Train a target that USES its hidden layer (XOR of signs), then flash the most accurate. See notes/25-phase2.md."""
    from . import train as tr
    from .capture import session
    from .target import flash, save_truth

    truth = paths.out(truth)
    with ui.Session("train and flash a Phase-2 target",
                    subtitle=[f"XOR-of-signs, best of {seeds} seeds",
                              f"truth -> {truth}"],
                    sources=_header(), hardware=flash_device, total_stages=3) as s:
        s.stage(f"training {seeds} networks in the quantised forward model")
        best = None
        for seed in range(seeds):
            model = tr.train(seed=seed)
            ev = tr.evaluate(model, seed=seed)
            s.detail(f"seed {seed}: task {ev['task_accuracy']:.3f}, "
                     f"linear {ev['linear_accuracy']:.3f}, "
                     f"hidden usage {ev['hidden_usage']:.2f}")
            if best is None or ev["task_accuracy"] > best[1]["task_accuracy"]:
                best = (model, ev)
        model, ev = best

        s.stage("writing the header and rebuilding")
        hex_path = flash(model, s.detail)
        truth.parent.mkdir(parents=True, exist_ok=True)
        save_truth(model, truth)
        if flash_device:
            s.stage("flashing")
            with session(hex_path, s.detail):
                pass
        s.table(["property", "value"], [
            ["task accuracy (trained net)", f"{ev['task_accuracy']:.3f}"],
            ["linear-model baseline", f"{ev['linear_accuracy']:.3f}"],
            ["hidden activations off the rails", f"{ev['hidden_usage']:.3f}"],
            ["activation histogram 0..7", ev["activation_hist"]],
            ["L2_W", model["L2_W"]],
        ])
        s.done("Phase-2 target flashed", artefact=truth)


@app.command("phase2")
def phase2(recovered: Path = typer.Option(..., "--recovered", help="extract model.json"),
           truth: Path = typer.Option(..., "--truth", help="the trained target")):
    """Score a recovered Phase-2 model on the task's own inputs (where the hidden units sit on their thresholds)."""
    from . import train as tr

    rec = json.loads(paths.existing(recovered, "recovered model").read_text())
    tru = json.loads(paths.existing(truth, "truth").read_text())
    with ui.Session("Phase-2 evaluation on the task distribution",
                    subtitle=["recovered vs flashed network, XOR-of-signs inputs"],
                    hardware=False, total_stages=1) as s:
        s.stage("scoring on task inputs")
        rep = tr.task_report(tru, rec)
        s.table(["against the trained network", "result"], [
            ["trained net task accuracy", f"{rep['true_task_accuracy']:.3%}"],
            ["recovered net task accuracy", f"{rep['recovered_task_accuracy']:.3%}"],
            ["decision agreement on task", f"{rep['decision_agreement']:.3%}"],
            ["samples", rep["samples"]],
        ])
        s.done(f"decisions match on {rep['decision_agreement']:.3%} of task inputs")


# -------------------------------------------------------------- analysis ----

@app.command()
def layer(config: list[Path] = typer.Option(..., "--config", "-c"),
          profile_with: list[Path] = typer.Option([], "--profile", "-p",
                                                  help="vote across these campaigns")):
    """Recover layer-1 weights for every configured column; pass several campaigns with -p to vote."""
    from .attack import profile, recover, vote

    loaded = _configs(config)
    panel = [Config.load(paths.existing(p, "campaign config")) for p in profile_with]
    campaigns = {(c.profiling, c.prof_column) for c in (panel or [c for _, c in loaded])}

    with ui.Session("layer-1 weights",
                    subtitle=[f"{len(loaded)} columns, {len(campaigns)} campaign(s)"
                              + (", majority vote" if len(panel) > 1 else "")],
                    sources=[p for p, _ in loaded],
                    total_stages=len(loaded) + 1) as s:
        s.stage("building templates")
        cache: dict[tuple, object] = {}
        panel_profiles = [profile(c, s.sub) for c in panel]
        results, configs = [], []
        for _, cfg in loaded:
            key = (cfg.profiling, cfg.prof_column)
            if not panel and key not in cache:
                cache[key] = profile(cfg, s.sub)
            s.stage(f"column {cfg.column}")
            res = vote(panel_profiles, cfg, s.sub) if panel else recover(cache[key], cfg, s.sub)
            results.append(res)
            configs.append(cfg)

    neurons = configs[0].neurons
    s.table(["neuron"] + [f"x{c.column}" for c in configs],
            [[str(n)] + [_cell(res[n]) for res in results] for n in range(neurons)])
    split = [(c.column, r.neuron, r.votes) for c, res in zip(configs, results)
             for r in res if r.votes and r.agreement < len(r.votes)]
    for col, n, votes in split:
        ui.console.print(f"[dim]  * x{col} neuron {n}: votes "
                         + ", ".join(f"{v:+d}" for v in votes) + "[/dim]")

    report.write_layer(s.run, configs, results, panel or None)
    ok = sum(r.correct for res in results for r in res)
    total = sum(len(res) for res in results)
    s.done(f"{ok}/{total} weights recovered", ok=ok == total,
           artefact=s.run.report)
    raise typer.Exit(0 if ok == total else 1)


def _cell(r) -> str:
    text = f"{r.recovered:+d}" + ("*" if r.votes and r.agreement < len(r.votes) else "")
    if r.truth is None:
        return text
    return f"[green]{text}[/green]" if r.correct else f"[red]{text}[/red]"


@app.command()
def activation(results: Path = typer.Option(..., "--from"),
               profile_dir: Path = typer.Option(..., "--profile"),
               target_dir: Path = typer.Option(..., "--target"),
               count: int = typer.Option(3000, "--count", "-n")):
    """Recover per-neuron bias and scale by max-likelihood over (b, s) against the 8 activation templates."""
    import numpy as np

    from .activation import NeuronFit, fit_neuron, leak_points
    from .extract import L2_READS, POI1
    from .traces import load_traces

    profile_dir = paths.existing(profile_dir, "activation profile")
    target_dir = paths.existing(target_dir, "target sweep")
    weights = _recovered(results)
    labels = np.loadtxt(profile_dir / "activations.txt").astype(int)
    pin = np.loadtxt(profile_dir / "inputs.txt").astype(int)[:count]
    tin = np.loadtxt(target_dir / "inputs.txt").astype(int)[:count]
    if not (pin == tin).all():
        raise typer.BadParameter(
            "profile and target were captured with different inputs; the "
            "saturated activation classes will not match")

    neurons = weights.shape[0]
    with ui.Session("layer-1 bias and scale",
                    subtitle=[f"profile {profile_dir.name}, target {target_dir.name}",
                              f"{count} traces"],
                    sources=paths.existing(results), total_stages=2) as s:
        s.stage("loading sweeps")
        prof = load_traces(profile_dir, count, WINDOW_FULL, s.sub).astype(np.float64)
        tgt = load_traces(target_dir, count, WINDOW_FULL, s.sub).astype(np.float64)
        acc = (pin @ weights.T).astype(float)
        s.stage("fitting the activation model")
        inv_grid = np.arange(15.0, 80.01, 0.05)
        bias_grid = np.arange(-80.0, 80.01, 1.0)         # phase-2 biases reach -63
        fits = []
        for n in s.track(range(neurons), "neurons", total=neurons):
            allowed = np.zeros(prof.shape[1], bool)      # layer 2's reads of hidden_q only;
            allowed[L2_READS[0]:L2_READS[1]] = True       #   the acc region contaminates (b, s)
            pois = leak_points(prof, labels[:count, n], acc[:, n], allowed)
            inv, bias = fit_neuron(prof[:, pois], labels[:count, n], tgt[:, pois],
                                   acc[:, n], inv_grid, bias_grid)
            fits.append(NeuronFit(n, pois, inv, bias))

    s.table(["neuron", "POIs", "1/s", "scale", "bias"],
            [[f.neuron, len(f.pois), f"{f.inv_scale:.2f}", f"{f.scale:.5f}",
              f"{f.bias:+d}"] for f in fits])
    report.write_activation(s.run, fits, profile_dir, target_dir)
    s.done(f"{len(fits)} neurons fitted", artefact=s.run.report)


@app.command()
def layer2(data: Path = typer.Option(..., "--data"),
           count: int = typer.Option(1000, "--count", "-n")):
    """Recover layer-2 weights from a hidden-activation campaign (templates keyed on v; correlation cannot)."""
    import numpy as np

    from .layer2 import recover
    from .traces import ground_truth_l2, load_manifest, load_traces

    data = paths.existing(data, "layer-2 campaign")
    tags = np.loadtxt(data / "tags.txt").astype(int)
    manifest = load_manifest(data)
    truth = ground_truth_l2(_header()) if _header().exists() else None

    with ui.Session("layer-2 weights",
                    subtitle=[f"{len(manifest)} captures from {data.name}"],
                    sources=data / "manifest.txt", total_stages=2) as s:
        s.stage("loading the campaign")
        prof = {k: load_traces(data / f"k{k}", count, WINDOW_L2, s.sub)
                for k in s.track(sorted(manifest), "captures", total=len(manifest))}
        atk = load_traces(data / "attack", count, WINDOW_L2, s.sub)
        s.stage("scoring hypotheses")
        results, kept = recover(prof, atk, tags, manifest, HYPOTHESES, truth=truth)

    dropped = sorted(set(manifest) - set(kept))
    if dropped:
        ui.console.print(f"[yellow]  dropped glitched captures: {dropped}[/yellow]")
    s.table(["input", "POIs", "templates", "coverage", "recovered", "true", "margin", ""],
            [[r.index, len(r.pois), r.templates, f"{r.coverage:.0%}",
              f"{r.recovered:+d}", "-" if r.truth is None else f"{r.truth:+d}",
              f"{r.margin:,.0f}", "" if r.correct is None else ui.mark(r.correct)]
             for r in results])
    report.write_layer2(s.run, results, data, dropped)
    ok = sum(bool(r.correct) for r in results)
    s.done(f"{ok}/{len(results)} layer-2 weights recovered",
           ok=ok == len(results), artefact=s.run.report)
    raise typer.Exit(0 if ok == len(results) else 1)


@app.command()
def l2bias(results: Path = typer.Option(..., "--from"),
           params: Path = typer.Option(..., "--params"),
           weights2: Path = typer.Option(..., "--weights2"),
           data: Path = typer.Option(..., "--data"),
           target: Path = typer.Option(..., "--target"),
           count: int = typer.Option(3000, "--count", "-n")):
    """Recover the second layer's bias from a `capture-l2bias` profile."""
    import numpy as np

    from .layer2 import recover_bias
    from .sweep import craft
    from .traces import load_traces

    data = paths.existing(data, "bias campaign")
    target = paths.existing(target, "target sweep")
    flashed = {int(l.split()[0]): int(l.split()[1])
               for l in (data / "manifest.txt").read_text().splitlines()}
    rec = _nonlinearity(params)
    _, accs = craft(_recovered(results), count)
    A = np.stack([np.clip(np.floor((accs[:, o] + b) / inv + 0.5), 0, 7)
                  for o, (inv, b) in enumerate(rec)], 1).astype(int)
    acc2 = (A @ _layer2_weights(weights2)).astype(np.int64)

    with ui.Session("layer-2 bias",
                    subtitle=[f"profiling biases {sorted(flashed.values())}"],
                    sources=data / "manifest.txt", total_stages=2) as s:
        s.stage("loading the campaign")
        prof = {i: load_traces(data / f"b{i}", count, WINDOW_L2, s.sub)
                for i in s.track(sorted(flashed), "captures", total=len(flashed))}
        atk = load_traces(target, count, WINDOW_L2, s.sub)
        s.stage("scoring hypotheses")
        bias, _ = recover_bias(prof, flashed, atk, acc2)

    s.table(["parameter", "value"], [["layer-2 bias", f"{bias:+d}"]])
    s.run.results.write_text(json.dumps({"bias": bias}, indent=2))
    s.run.report.write_text(
        f"# Layer 2 bias\n\nRecovered **{bias:+d}**.\n\n"
        f"- profiling biases: {sorted(flashed.values())}\n"
        f"- the intermediate is `acc2 + b2`; one profiling bias is not enough,\n"
        f"  a hypothesis shifts the value out of the profiled set.\n")
    s.done(f"layer-2 bias {bias:+d}", artefact=s.run.report)


@app.command()
def transfer(config: list[Path] = typer.Option(..., "--config", "-c")):
    """Cross-profile every campaign against every attack set: does one column's templates recover another?"""
    from dataclasses import replace

    from .attack import profile, recover

    loaded = _configs(config)
    columns = [c.column for _, c in loaded]
    with ui.Session("template transfer",
                    subtitle=[f"{len(columns)}x{len(columns)} matrix over columns {columns}"],
                    sources=[p for p, _ in loaded],
                    total_stages=len(columns) + 1) as s:
        s.stage("building one profile per campaign")
        profiles = {c.column: profile(c, s.sub) for _, c in loaded}
        by_column = {c.column: c for _, c in loaded}
        grid = {}
        for p in columns:
            s.stage(f"campaign x{p} against every attack set")
            for a in columns:
                res = recover(profiles[p], replace(by_column[a], profiling_column=p), s.sub)
                grid[(p, a)] = (sum(r.correct for r in res), res)

    s.table(["profiling \\ attack"] + [f"x{a}" for a in columns],
            [[f"x{p}"] + [f"{grid[(p, a)][0]}/{len(grid[(p, a)][1])}" for a in columns]
             for p in columns])
    report.write_transfer(s.run, columns, grid)
    off = [v for (p, a), v in grid.items() if p != a]
    perfect = sum(ok == len(res) for ok, res in off)
    s.done(f"{perfect}/{len(off)} transfers recovered every weight",
           ok=perfect == len(off), artefact=s.run.report)
    raise typer.Exit(0 if perfect == len(off) else 1)


@app.command()
def extract(campaign: list[str] = typer.Option(..., "--campaign",
                                               help="DIR:COLUMN, repeatable; several are voted"),
            work: Path = typer.Option(..., "--work"),
            truth: Path = typer.Option(None, "--truth", help="known network, scoring only"),
            attack_traces: int = typer.Option(1000, "--attack-traces"),
            sweep_traces: int = typer.Option(3000, "--sweep-traces"),
            l2_traces: int = typer.Option(1000, "--l2-traces"),
            l2_repeats: int = typer.Option(2, "--l2-repeats"),
            span: str = typer.Option("-60,420", "--span",
                                     help="accumulator sweep window; use a symmetric "
                                          "span (e.g. -350,350) when W1 rows are "
                                          "sign-flipped pairs, or the shared window "
                                          "collapses to saturation"),
            offline: bool = typer.Option(False, "--offline",
                                         help="refuse to touch the device")):
    """Extract a whole 10-6-1 network: captures what is missing, reuses what is not."""
    import numpy as np

    from . import extract as ex
    from .capture import Step, execute
    from .sweep import craft, hidden_hotspot

    campaigns = []
    for item in campaign:
        directory, _, column = item.partition(":")
        if not column:
            raise typer.BadParameter(f"expected DIR:COLUMN, got {item!r}")
        campaigns.append((paths.existing(directory, "layer-1 campaign"), int(column)))
    work = paths.out(work)
    work.mkdir(parents=True, exist_ok=True)
    truth = paths.existing(truth, "truth") if truth else None

    def need(path: Path) -> bool:
        if paths.has_traces(path):
            return False
        if offline:
            raise typer.BadParameter(f"--offline but {path} has no traces")
        return True

    with ui.Session("full extraction",
                    subtitle=[f"{len(campaigns)} campaign(s): "
                              + ", ".join(f"{d.name}:x{c}" for d, c in campaigns)
                              + (", majority vote" if len(campaigns) > 1 else ""),
                              f"work {work}"],
                    sources=[d / "manifest.txt" for d, _ in campaigns],
                    total_stages=8) as s:
        cols = {c: work / f"attack_c{c}" for c in range(10)}
        todo = [c for c in cols if need(cols[c])]
        s.stage(f"layer-1 attack sets ({len(todo)} to capture, {10 - len(todo)} on disk)")
        if todo:
            execute([Step(label=f"column {c}", out=cols[c], count=attack_traces, column=c)
                     for c in todo], s.sub)

        s.stage("recovering layer-1 weights")
        W1 = ex.layer1_weights(campaigns, cols, _header(),
                               min(300, attack_traces), s.sub)
        W1a = np.array(W1)
        s.detail(f"W1 = {W1}")

        sweep_lo, sweep_hi = (int(v) for v in span.split(","))
        vectors, accs = craft(W1a, sweep_traces, (sweep_lo, sweep_hi))
        sweep, actdir = work / "sweep", work / "act_profile"
        s.stage("accumulator sweep and activation profile")
        if need(sweep):
            execute([Step(label="sweep", out=sweep, count=sweep_traces,
                          vectors=vectors)], s.sub)
        if need(actdir):
            execute([Step(label="clone profile", out=actdir, count=sweep_traces,
                          vectors=vectors,
                          arrays=(("L1_B", (0,) * 6), ("L1_M0", (ex.CLONE_M0,) * 6),
                                  ("L1_N", (ex.CLONE_SHIFT,) * 6)),
                          note="clone: b = 0, 1/s = 50")], s.sub)
            labels = np.clip((accs.astype(np.int64) * ex.CLONE_M0
                              + (1 << (31 + ex.CLONE_SHIFT - 1))) >> (31 + ex.CLONE_SHIFT), 0, 7)
            np.savetxt(actdir / "activations.txt", labels, fmt="%d")

        s.stage("recovering layer-1 bias and scale")
        b1, s1 = ex.layer1_nonlinearity(actdir, sweep, W1a, sweep_traces, s.sub)
        s.detail(f"b1 = {b1}, 1/s = {[round(v, 2) for v in s1]}")

        l2dir = work / "layer2"
        s.stage("layer-2 campaign")
        if need(l2dir / "attack"):
            hot, tags = hidden_hotspot(W1a, list(zip(s1, b1)), l2_traces)
            rng = np.random.default_rng(0)
            schedule = np.stack([rng.permutation(np.tile(L2_WEIGHTS, l2_repeats))
                                 for _ in range(6)], axis=1)
            l2dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(l2dir / "tags.txt", tags, fmt="%d")
            plan = [Step(label=f"capture {k}", out=l2dir / f"k{k}", count=l2_traces,
                         vectors=hot, rows=(("L2_W", 0, tuple(int(v) for v in row)),),
                         note=f"L2_W = {[int(v) for v in row]}")
                    for k, row in enumerate(schedule)]
            plan.append(Step(label="attack set", out=l2dir / "attack",
                             count=l2_traces, vectors=hot))
            execute(plan, s.sub)
            (l2dir / "manifest.txt").write_text("".join(
                f"{k} {' '.join(str(int(v)) for v in row)}\n"
                for k, row in enumerate(schedule)))

        s.stage("recovering layer-2 weights")
        W2 = ex.layer2_weights(l2dir, l2_traces, s.sub)
        s.detail(f"W2 = {W2}")

        bdir = work / "l2bias"
        s.stage("layer-2 bias campaign")
        if need(bdir / "b0"):
            bdir.mkdir(parents=True, exist_ok=True)
            execute([Step(label=f"bias {b:+d}", out=bdir / f"b{i}", count=sweep_traces,
                          vectors=vectors, arrays=(("L2_B", (b,)),),
                          note=f"L2_B = {b:+d}")
                     for i, b in enumerate(ex.L2_BIASES)], s.sub)
            (bdir / "manifest.txt").write_text(
                "".join(f"{i} {b}\n" for i, b in enumerate(ex.L2_BIASES)))

        s.stage("recovering layer-2 bias")
        A = np.stack([np.clip(np.floor((accs[:, o] + b) / inv + 0.5), 0, 7)
                      for o, (inv, b) in enumerate(zip(s1, b1))], 1).astype(int)
        b2 = ex.layer2_bias(bdir, sweep, (A @ np.array(W2)).astype(np.int64),
                            sweep_traces, s.sub)
        s.detail(f"b2 = {b2}")

    model = ex.Model(l1_w=W1, l1_bias=b1, l1_inv_scale=s1, l2_w=W2, l2_bias=b2)
    (s.run.root / "model.json").write_text(model.to_json())
    s.table(["neuron"] + [f"x{c}" for c in range(len(W1[0]))] + ["bias", "1/scale"],
            [[n] + [f"{v:+d}" for v in W1[n]] + [f"{b1[n]:+d}", f"{s1[n]:.2f}"]
             for n in range(len(W1))])
    s.table(["layer 2"] + [f"a{n}" for n in range(len(W2))] + ["bias"],
            [["weights"] + [f"{v:+d}" for v in W2] + [f"{b2:+d}"]])

    card = ex.score(model, json.loads(truth.read_text())) if truth else None
    report.write_extraction(s.run, model, card, truth)
    if card:
        s.table(["against the known network", "result"], [
            ["layer-1 weights", f"{card['l1_weights']}/{card['l1_weights_total']}"],
            ["layer-1 biases exact", f"{card['l1_bias_exact']}/6"],
            ["layer-2 weights", f"{card['l2_weights']}/{card['l2_weights_total']}"],
            ["layer-2 bias exact", "yes" if card["l2_bias_exact"] else "no"],
            ["hidden vector exact", f"{card['hidden_vector_exact']:.3%}"],
            ["decision agreement", f"{card['decision_agreement']:.3%}"]])
        s.done(f"decisions match on {card['decision_agreement']:.3%} of inputs",
               artefact=s.run.root / "model.json")
    else:
        s.done("extraction complete", artefact=s.run.root / "model.json")


# ------------------------------------------------------------------ info ----

@app.command()
def status(config: list[Path] = typer.Option([], "--config", "-c"),
           work: Path = typer.Option(None, "--work")):
    """What data exists, and what the last run of each stage said."""
    from . import runs
    from .traces import ground_truth

    ui.banner("pipeline status")
    if config:
        rows = []
        for _, cfg in _configs(config):
            truth = (", ".join(f"{w:+d}" for w in ground_truth(cfg.firmware_config, cfg.column))
                     if cfg.firmware_config.exists() else "-")
            rows.append([f"column {cfg.column}", paths.traces_in(cfg.attack),
                         len(list(cfg.profiling.glob("k*"))), truth])
        ui.console.print("")
        ui.table(["target", "attack traces", "campaign captures", "true weights"], rows)
    if work:
        work = paths.out(work)
        rows = [[f"attack_c{c}", paths.traces_in(work / f"attack_c{c}") or "-"]
                for c in range(10)]
        rows += [[name, paths.traces_in(work / name) or "-"]
                 for name in ("sweep", "act_profile", "layer2/attack", "l2bias/b0")]
        ui.console.print(f"\n[bold]{work}[/bold]")
        ui.table(["stage", "traces"], rows)

    ui.console.print("\n[bold]recent runs[/bold]")
    recent = sorted((runs.PACKAGE_ROOT / "runs").glob("*/"), reverse=True)[:8]
    rows = []
    for r in recent:
        res = r / "results.json"
        summary = "-"
        if res.exists():
            try:
                data = json.loads(res.read_text())
                summary = (f"{data['recovered']}/{data['total']}" if "recovered" in data
                           else next((f"{k} = {v}" for k, v in data.items()
                                      if isinstance(v, (int, float))), "done"))
            except Exception:
                summary = "unreadable"
        rows.append([r.name, summary])
    ui.table(["run", "result"], rows)


if __name__ == "__main__":
    app()
