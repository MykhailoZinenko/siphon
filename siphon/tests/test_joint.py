"""Joint (weight, bias) recovery: the attack that assumes nothing of the victim.

The leak here is a function of acc = w*x + b, which is what the accumulator
actually holds. Profiling sweeps the bias on a device the attacker owns; the
victim's bias is never flashed, so the test fails if the method quietly needs it.
"""
import numpy as np
import pytest

from siphon import joint as jt
from siphon.campaign import joint_schedule

NEURONS = 2
WIDTH = 60
LEAK = [10, 40]
PER_CAPTURE = 120
INPUTS = np.array([-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7])  # the device's set
VICTIM_W = [-6, 3]
VICTIM_B = [4, -9]          # never used while profiling


# Per-bit weights: a real device does not leak every bit equally, and the
# templates are measured, not modelled. An exactly-HW leak is degenerate -
# see test_pure_hamming_weight_is_nearly_degenerate_in_pairs.
BIT_WEIGHT = np.random.default_rng(99).normal(1.0, 0.25, 32)


def _leak(acc: np.ndarray, rng, weight=0, pure_hw=False) -> np.ndarray:
    # What the accumulator point leaks, plus a small dependence on the
    # multiplier itself - the residual measured on the board, 11-40% of a class
    # gap. It is modelled on |w| because profiling only ever flashes positive
    # weights while the victim's may be negative; the sign arrives through the
    # input, exactly as on the device.
    bits = np.array([[(int(a) >> i) & 1 for i in range(32)] for a in acc], dtype=np.float64)
    weights = np.ones(32) if pure_hw else BIT_WEIGHT
    return 0.004 * (bits @ weights) + 0.0004 * abs(weight) + rng.normal(0, 0.0015, len(acc))


@pytest.fixture(scope="module")
def campaign():
    rng = np.random.default_rng(5)
    schedule = joint_schedule(NEURONS)
    traces, inputs, weights, biases = [], [], [], []
    for bias, values in schedule:
        x = rng.choice(INPUTS, PER_CAPTURE)
        block = rng.normal(0, 0.0015, (PER_CAPTURE, WIDTH))
        for n in range(NEURONS):
            block[:, LEAK[n]] = _leak(values[n] * x + bias, rng, weight=values[n])
        traces.append(block)
        inputs.append(x.astype(np.float64))
        weights.append([values[n] for n in range(NEURONS)])
        biases.append(bias)
    return schedule, traces, inputs, weights, biases


@pytest.fixture(scope="module")
def attack():
    rng = np.random.default_rng(6)
    x = rng.choice(INPUTS, 400)
    block = rng.normal(0, 0.0015, (400, WIDTH))
    for n in range(NEURONS):
        block[:, LEAK[n]] = _leak(VICTIM_W[n] * x + VICTIM_B[n], rng, weight=VICTIM_W[n])
    return block, x.astype(np.float64)


def _run(campaign, attack, bias_range=64):
    schedule, traces, inputs, weights, biases = campaign
    attack_traces, attack_inputs = attack
    accs = {n: jt.accumulators(inputs, [w[n] for w in weights], biases)
            for n in range(NEURONS)}
    templates, leak = [], []
    for n in range(NEURONS):
        templates.append(jt.build(np.concatenate(accs[n]),
                                  np.concatenate([t[:, LEAK[n]] for t in traces])))
        leak.append(attack_traces[:, LEAK[n]])
    return jt.recover(templates, leak, LEAK, attack_inputs,
                      range(-7, 8), range(-bias_range, bias_range + 1))


def test_recovers_weight_and_bias_without_knowing_either(campaign, attack):
    results = _run(campaign, attack)
    assert [r.weight for r in results] == VICTIM_W
    assert [r.bias for r in results] == VICTIM_B


def test_the_accumulator_poi_is_found_without_a_model(campaign):
    # Every neuron reads the same input, so all of them leak at all the points;
    # as in the weight attack, the assignment comes from separating the peaks
    # and taking them in execution order, not from each neuron's own argmax.
    from siphon.poi import assign_pois
    schedule, traces, inputs, weights, biases = campaign
    profiles = {n: jt.acc_profile(traces, jt.accumulators(
        inputs, [w[n] for w in weights], biases)) for n in range(NEURONS)}
    assert assign_pois(profiles, separation=5) == LEAK


def test_thinly_covered_hypotheses_are_dropped_and_scoring_is_per_trace(campaign):
    # Coverage is the real failure mode, in both directions. Demanding every
    # accumulator value eliminates the TRUE pair over one missing class (that is
    # what happened while building this); allowing any coverage lets a guess win
    # by predicting less. So: keep a hypothesis that explains 90% of the attack
    # traces, and score it per covered trace.
    schedule, traces, inputs, weights, biases = campaign
    accs = jt.accumulators(inputs, [w[0] for w in weights], biases)
    templates = jt.build(np.concatenate(accs),
                         np.concatenate([t[:, LEAK[0]] for t in traces]))
    rng = np.random.default_rng(7)
    x = rng.choice(INPUTS, 200)
    leak = _leak(VICTIM_W[0] * x + VICTIM_B[0], rng, weight=VICTIM_W[0])

    levels, counts = np.unique(x, return_counts=True)
    scores = jt.score(templates, leak, x.astype(np.float64), range(-7, 8),
                      range(-500, 501))
    for w, b in scores:
        covered = counts[[int(w * xi + b) in templates for xi in levels]].sum()
        assert covered >= 0.9 * counts.sum()
    assert all(abs(b) < 200 for _, b in scores), "a bias far outside the sweep was scored"


def test_pure_hamming_weight_is_nearly_degenerate_in_pairs():
    # Worth recording: if the point leaked EXACTLY HW(acc), then (w, b) and
    # (2w, 2b) would produce accumulator sequences related by doubling, and
    # HW(2y) = HW(y) for y > 0. Only the negative half, which loses a one bit
    # per doubling in two's complement, separates them - weak evidence. Real
    # devices leak bits unequally, which is what saves the attack; a target
    # whose accumulator leaked flat HW would need another distinguisher.
    rng = np.random.default_rng(11)
    truth_w, truth_b = -6, 4
    acc_true = truth_w * INPUTS + truth_b
    acc_half = (truth_w // 2) * INPUTS + truth_b // 2
    hw = lambda a: np.array([bin(int(v) & 0xFFFFFFFF).count("1") for v in a])
    positives = acc_true > 0
    assert (hw(acc_true)[positives] == hw(acc_half)[positives]).all()
    assert (hw(acc_true)[~positives] != hw(acc_half)[~positives]).any()


def test_weight_and_bias_are_not_confusable(campaign, attack):
    # A wrong bias shifts acc by a constant, a wrong weight tilts it in x, so
    # inputs of both signs separate them. The margin to the next weight should
    # be real, not a coin flip.
    results = _run(campaign, attack)
    assert all(r.margin > 0 for r in results)
