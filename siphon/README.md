# siphon

Template-based extraction of quantized neural-network weights from power traces.
Target: a 10 -> 6 -> 1 int8 network on a ChipWhisperer-Lite (STM32F303).

## Two families of command

Names say whether the device is involved:

    capture-*    patches the firmware, flashes, records traces
    everything else reads what is already on disk

Every command prints the same shape: a header saying what it is about to do,
numbered stages while it runs, a table of results, and the artefact path.

Paths on the command line mean what the shell means by them - relative is
relative to where you stand. Paths written inside YAML configs resolve against
the package root, because a config has to be portable.

## Whole network, one command

```bash
siphon extract --campaign data/rebuilt_firmware/campaign_column_c0:0 \
               --campaign data/rebuilt_firmware/campaign_column:1 \
               --campaign data/rebuilt_firmware/campaign_column_c5:5 \
               --work data/rebuilt_firmware/target \
               --truth data/targets/student.json
```

Captures what is missing, reuses what is not. `--offline` refuses to touch the
device, which is the safe way to re-run the analysis.

Give it more than one campaign. With a single one the weakest neuron can flip a
weight, and the error does not stay local: the wrong weight shifts that neuron's
accumulator, its activation fit collapses, and the layer-2 bias follows. One
campaign measured 59/60 weights and 88.7% agreement; three give 60/60 and 99.98%.

## Stage by stage

| stage | capture | analyse |
|---|---|---|
| layer-1 weights | `capture-campaign`, `capture-attack` | `layer` |
| layer-1 bias and scale | `capture-sweep`, `capture-activation` | `activation` |
| layer-2 weights | `capture-layer2` | `layer2` |
| layer-2 bias | `capture-l2bias` | `l2bias` |

Plus `transfer` (does a campaign on one column recover another - it does),
`status` (what data exists) and `new-target` (write a known random network into
the device, for testing whether the attack generalises).

The layer-1 campaign is the one thing reusable between targets: its inputs are
hot spots with a random value and the weights in it are ours.

## Layout

```
src/siphon/     the attack
  paths.py      one rule for command-line paths
  ui.py         one presentation layer
  capture.py    device access; `execute` runs a plan, reflashing only on change
  extract.py    the four stages chained
analysis/       one-off analyses behind numbers quoted in notes/, dead ends kept
legacy/         algorithms 1-3, still runnable
configs/        one YAML per attacked column
runs/           per-run config snapshot, log, report.md, results.json
```

Notes are in `../notes` (Russian); start with `00-journey.md` for how this was
arrived at, or `19-runbook.md` for the order to run things in.
