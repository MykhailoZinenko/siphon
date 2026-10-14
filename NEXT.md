# Next task: make the joint (weight, bias) attack work on the board

Read `notes/27-profiling-context.md` first, then `siphon/src/siphon/joint.py`.
Everything below was measured on 2026-10-14, not assumed. Commit 933ca30.

## Where it stands

`siphon joint -c siphon/configs/context/joint.yaml` recovers (w, b) per neuron
with templates keyed on the accumulator instead of the product. On synthetic
traces it gets weights and biases exactly right. On the board: **2/6 weights,
0/6 biases**, and with per-hypothesis offset calibration 1/6 weights but 2/6
biases (neuron 2 exact).

## The measured obstacle

The victim's leakage sits **1-6 class widths** away from the templates at the
same accumulator value, because the profiling device flashes the attacker's own
requantisation. Per neuron: +0.0151, +0.0028, +0.0038, +0.0058, -0.0181,
+0.0030, against a class gap of ~0.0029.

`firmware/source/network.c` shows why no nearby point escapes it:

```c
acc += bias;                        // bias enters here
int64_t temp = (int64_t)acc * m0;   // requant, next instruction
```

Bias-sensitive and requant-clean are mutually exclusive at the accumulator.

Already ruled out, with numbers - do not redo:
- flash-to-flash drift: 0.1 class widths (quiet region, 30 captures)
- requant-invariant POIs 25-45 samples away: 1/6, worse than 2/6 (SNR drops to
  ~30% and buys only a 2-5x sensitivity reduction)
- per-hypothesis offset calibration: trades weights for biases, see above

## The plan

**Attack the per-multiply leak, which is bias-independent by construction** -
it happens before `acc += bias` and ~536 samples from the requant. The problem
is finding it: the Latin square moves every neuron's weight together
(`w1 = w0 + 1 mod 8`), so each neuron's accumulator is a near-linear function of
its neighbour's product and every neuron shows high SNR at every other neuron's
point. A per-neuron window search just returns the neighbouring accumulate
point - verified, that is what happened.

So: **recapture with a decoupled campaign**, each neuron's weight drawn
independently, exactly the fix this project already used for layer 2 (see
CLAUDE.md, "The Latin square blocks the usual discriminator"). Then:

1. Find per-neuron POIs on the clone campaign alone (the attacker has no other
   device). Criterion: high SNR over v, pooled across captures - pooling is what
   rejects samples that leak only the input x. A sample that leaks x shows huge
   within-capture SNR and recovers `w = 0` for most neurons; that is how the
   first attempt failed.
2. Check the candidate is pre-bias: its leakage must not move when the flashed
   bias changes. `data/context/camp_joint` has 15 bias levels, so this is a free
   test on data already captured.
3. Recover w at that point with the existing v-keyed template attack, profiling
   on the clone (zero neighbours, clone bias and requant). Target: 6/6, which
   would mean the attack needs nothing from the victim.
4. Then recover b at the accumulator point with w fixed - a 1-D search over the
   bias, where calibration already showed the bias information survives.

## Rules

- Hardware is real. Campaigns reflash the board; `capture-*` touches the device,
  everything else reads disk. A capture holds `firmware/.header-patched` and
  ground-truth reads refuse while it is held - do not work around that, it
  exists because an overlapping analysis run silently scored against the
  campaign's own weights.
- Never write into an existing trace directory; delete it first if retrying.
- Do not claim a result without running it. Several confident conclusions in
  this project turned out wrong when measured, including one corrected today.
- Notes in Russian, code and commits in English.

## Useful state

- Target: seed-4242, dense. Column 0 truth `-6, +1, -1, -3, +3, -3`,
  biases `[4, -9, 2, 9, -16, 17]`.
- Attack set: `data/context/attack`, 1000 traces, pristine network, reusable.
- Campaigns on disk: `camp_target`, `camp_zero`, `camp_random`, `camp_c`
  (zero + clone b/s), `camp_d` (target + clone b/s), `camp_e` (clone bias only),
  `camp_f` (clone requant only), `camp_joint` (30 captures, biases -56..+56).
- Accumulator POIs for this build: 4237, 4857, 5477, 6097, 6717, 7337.
- `venv/bin/python -m pytest siphon/tests -q` - 20 passing, 2 skipped.
