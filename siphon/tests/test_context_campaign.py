"""The profiling-context experiment: patcher safety, TVLA POIs, and HW-keyed templates.

Hardware time is expensive and a bad patch burns a whole campaign, so everything
that can be checked without the board is checked here: that a zero/random context
touches nothing but L1_W, that SNR and TVLA agree on a known leak point, and that
keying templates on HW(v) really does lose the doubling pairs (+3 vs +6).
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from siphon import firmware
from siphon.attack import profile, recover
from siphon.campaign import CAPTURES, column_for
from siphon.config import Config
from siphon.leakage import hamming_int32
from siphon.poi import assign_pois, leak_profile, tvla_profile
from siphon.progress import Reporter

ROOT = Path(__file__).resolve().parents[1]
NEURONS = 2
WIDTH = 400
TRACES = 200
INPUTS = [-7, -5, -3, -1, 1, 3, 5, 7]
TRUTH = [3, -6]                       # the weights the synthetic attack set carries
LEAK = [100, 200]                     # one sample per neuron, in execution order


def _header(path: Path, column: int, weights: list[int]) -> Path:
    rows = []
    for neuron in range(NEURONS):
        row = [0] * 10
        row[column] = weights[neuron]
        rows.append("    { " + ", ".join(map(str, row)) + " },")
    path.write_text("static const int8_t L1_W[NET_L1][NET_INPUTS] = {\n"
                    + "\n".join(rows).rstrip(",") + "\n};\n")
    return path


def _capture(folder: Path, weights: list[int], column: int, rng) -> None:
    # One capture's worth of traces: sample LEAK[n] is graded in that neuron's v.
    folder.mkdir(parents=True, exist_ok=True)
    vectors = []
    for i in range(TRACES):
        x = int(rng.choice(INPUTS))
        vector = [0] * 10
        vector[column] = x
        vectors.append(vector)
        trace = rng.normal(0.0, 0.05, WIDTH)
        for neuron in range(NEURONS):
            trace[LEAK[neuron]] += 0.02 * weights[neuron] * x     # leaks v = w*x
        (folder / f"trace_{i}.txt").write_text("\n".join(f"{v:.6f}" for v in trace) + "\n")
    (folder / "inputs.txt").write_text(
        "".join(" ".join(map(str, v)) + "\n" for v in vectors))


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    # A whole campaign plus a matching attack set, generated from a known leak.
    rng = np.random.default_rng(7)
    base = tmp_path_factory.mktemp("context")
    campaign, attack = base / "campaign", base / "attack"
    manifest = {}
    for k in range(CAPTURES):
        values = column_for(k, NEURONS)
        manifest[k] = values
        _capture(campaign / f"k{k}", values, 0, rng)
    (campaign / "manifest.txt").write_text(
        "".join(f"{k} {' '.join(map(str, v))}\n" for k, v in manifest.items()))
    _capture(attack, TRUTH, 0, rng)

    return Config(name="synthetic", column=0, neurons=NEURONS,
                  profiling=campaign, attack=attack,
                  firmware_config=_header(base / "network_config.h", 0, TRUTH),
                  window=(0, WIDTH), profiling_traces=TRACES, attack_traces=TRACES,
                  poi_separation=20)


def test_templates_on_v_recover_the_weights(synthetic):
    results = recover(profile(synthetic, Reporter()), synthetic, Reporter())
    assert [r.recovered for r in results] == TRUTH


def test_hw_keyed_templates_lose_a_weight(synthetic):
    # Merging v into HW(v) costs at least one weight here. The mechanism is class
    # merging in general, NOT an exact tie: with inputs of both signs the doubling
    # pair is still separated by the negative half (see the two tests below).
    from dataclasses import replace
    config = replace(synthetic, template_key="hw")
    results = recover(profile(config, Reporter()), config, Reporter())
    assert any(not r.correct for r in results), \
        "HW keying unexpectedly recovered every weight; check the synthetic leak"


def _hw_scores(inputs: np.ndarray, seed: int) -> dict[int, float]:
    # A two-capture campaign over weights 3 and 6, attacked with traces from 3.
    # Both hypotheses are fully covered, so a difference in score can only come
    # from the templates, never from a missing class.
    from siphon.templates import build, score
    rng = np.random.default_rng(seed)
    profiled_v = np.concatenate([3 * inputs, 6 * inputs])
    leakage = (0.02 * profiled_v + rng.normal(0, 0.01, len(profiled_v)))[:, None]
    templates = build(hamming_int32(profiled_v), leakage)
    attack = (0.02 * 3 * inputs + rng.normal(0, 0.01, len(inputs)))[:, None]
    return score(templates, attack, inputs, [3, 6], key=hamming_int32)


def test_hw_key_ties_the_doubling_pair_on_positive_inputs():
    # The claim the HW model is criticised for, pinned where it holds: with v > 0
    # only, HW(6x) = HW(3x) for every x, so the two hypotheses read the same
    # templates in the same order and the attack cannot prefer either.
    scores = _hw_scores(np.tile(np.arange(1, 8), 40), seed=3)
    assert np.isfinite(scores[3]) and scores[3] == scores[6]


def test_hw_key_separates_the_doubling_pair_once_inputs_go_negative():
    # And where it does not hold: a negative v loses a one bit on doubling, so
    # the tie breaks and the true weight wins. Any claim that HW cannot tell +3
    # from +6 has to say "for v > 0".
    scores = _hw_scores(np.tile(np.array([-7, -5, -3, -1, 1, 3, 5, 7]), 40), seed=4)
    assert np.isfinite(scores[6]) and scores[3] > scores[6]


def test_snr_and_tvla_find_the_same_samples(synthetic):
    from siphon.attack import load_campaign
    manifest, captures, traces, inputs = load_campaign(synthetic, Reporter())
    picks = {}
    for name, locate in (("snr", leak_profile), ("tvla", tvla_profile)):
        profiles = {n: locate(traces, inputs, [manifest[k][n] for k in captures])
                    for n in range(NEURONS)}
        picks[name] = assign_pois(profiles, synthetic.poi_separation)
    assert picks["snr"] == LEAK
    assert all(abs(a - b) <= 2 for a, b in zip(picks["tvla"], LEAK)), picks


def test_hamming_int32_doubling_holds_only_for_positive_v():
    assert hamming_int32(np.array([3, 6, 12])).tolist() == [2, 2, 2]
    assert hamming_int32(np.array([-3, -6, -12])).tolist() == [31, 30, 29]
    assert hamming_int32(np.array([-1]))[0] == 32


def _naive_tvla(traces: np.ndarray, values: np.ndarray, min_class: int) -> np.ndarray:
    # The straightforward class-against-the-rest form, kept as the reference the
    # one-pass version must reproduce.
    best = np.zeros(traces.shape[1])
    for value in np.unique(values):
        mask = values == value
        if mask.sum() < min_class or (~mask).sum() < 2:
            continue
        a, b = traces[mask].astype(np.float64), traces[~mask].astype(np.float64)
        denom = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
        best = np.maximum(best, np.abs(a.mean(0) - b.mean(0)) / np.maximum(denom, 1e-12))
    return best


def _fake_campaign(rng, offsets=None):
    # Eight captures of a two-neuron layer, leaking v at LEAK[n]. `offsets` adds a
    # per-capture constant at sample 300, i.e. leakage of the OTHER weights that
    # change between flashes.
    traces, inputs, weights = [], [], []
    for k in range(CAPTURES):
        x = rng.choice(INPUTS, 200)
        block = rng.normal(0.0, 0.05, (200, WIDTH))
        for neuron in range(NEURONS):
            block[:, LEAK[neuron]] += 0.02 * column_for(k, NEURONS)[neuron] * x
        if offsets is not None:
            block[:, 300] += offsets[k]
        traces.append(block.astype(np.float32))
        inputs.append(x.astype(np.float64))
        weights.append(column_for(k, NEURONS)[0])
    return traces, inputs, weights


def test_one_pass_tvla_matches_the_naive_form():
    rng = np.random.default_rng(11)
    traces, inputs, weights = _fake_campaign(rng)
    values = np.concatenate([(w * x).astype(np.int64) for x, w in zip(inputs, weights)])
    fast = tvla_profile(traces, inputs, weights, min_class=30)
    slow = _naive_tvla(np.concatenate(traces), values, 30)
    assert np.allclose(fast, slow, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("seed", [12, 13, 14])
def test_per_capture_offset_fools_the_pooled_contrast_only(seed):
    # Sample 300 says nothing about v, only which capture a trace came from -
    # the neighbouring weights' loads, which change at every flash. Pooled, that
    # is worth as much t as the genuine multiply (it decides the argmax on two of
    # these three seeds); inside a capture the other weights are constant, so the
    # same sample sits at the noise floor and the real point always wins.
    traces, inputs, weights = _fake_campaign(np.random.default_rng(seed),
                                             offsets=np.linspace(0.0, 2.0, CAPTURES))
    pooled = tvla_profile(traces, inputs, weights, min_class=30)
    within = tvla_profile(traces, inputs, weights, min_class=30, scope="capture")

    assert pooled[300] > 0.5 * pooled[LEAK[0]], "confound no longer rivals the leak"
    assert int(within.argmax()) == LEAK[0]
    quiet = np.delete(within, LEAK + [300])          # samples carrying nothing
    assert within[300] <= quiet.max(), "the offset still shows inside a capture"


def test_pooled_contrast_can_pick_the_confounded_sample():
    # The failure itself, on the seed where it bites: the pooled profile's peak is
    # the per-capture offset, not either neuron's multiply.
    traces, inputs, weights = _fake_campaign(np.random.default_rng(12),
                                             offsets=np.linspace(0.0, 2.0, CAPTURES))
    assert int(tvla_profile(traces, inputs, weights, min_class=30).argmax()) == 300


def test_truth_is_refused_while_a_capture_holds_the_header(tmp_path, monkeypatch):
    # Ground truth comes from the header a campaign overwrites, so an analysis run
    # that overlaps a capture scores against the campaign's own test weights.
    from siphon.traces import ground_truth
    header = _header(tmp_path / "network_config.h", 0, TRUTH)
    monkeypatch.setattr(firmware, "LOCK", tmp_path / ".header-patched")

    assert ground_truth(header, 0) == TRUTH          # nothing running
    with firmware.patched("a capture into camp_zero"):
        with pytest.raises(RuntimeError, match="camp_zero"):
            ground_truth(header, 0)
    assert ground_truth(header, 0) == TRUTH          # released again


def test_a_lock_from_a_dead_run_does_not_block(tmp_path, monkeypatch):
    # A killed campaign must not wedge every later analysis run.
    lock = tmp_path / ".header-patched"
    monkeypatch.setattr(firmware, "LOCK", lock)
    lock.write_text("999999\na capture that was killed\n")      # no such process

    assert firmware.patch_in_progress() is None
    assert not lock.exists()


@pytest.mark.parametrize("context", ["zero", "random"])
def test_context_patch_touches_only_l1_w(tmp_path, monkeypatch, context):
    # Dry run of the patcher on a copy: the attacked column must hold the flashed
    # values, the other nine must follow the context, and nothing else may move.
    from siphon.cli import _campaign_rows

    pristine = ROOT / "firmware" / "source" / "network_config.h"
    work = tmp_path / "network_config.h"
    shutil.copy(pristine, work)
    monkeypatch.setattr(firmware, "CONFIG_HEADER", work)

    config = Config(name="dry", column=3, neurons=6,
                    profiling=tmp_path, attack=tmp_path, firmware_config=pristine)
    values = column_for(0, config.neurons)
    rows = _campaign_rows(config, values, context, seed=1)
    original = pristine.read_text()
    try:
        for name, row, row_values in rows:
            firmware.set_row(name, list(row_values), row=row)
    finally:
        # if the redirect ever stops covering set_row's write path, the real
        # header must not be left patched - that is a burned campaign
        if pristine.read_text() != original:
            pristine.write_text(original)
            raise AssertionError("the patch reached the real network_config.h")

    import re
    block = re.search(r"L1_W\[NET_L1\]\[NET_INPUTS\]\s*=\s*\{(.*?)\};",
                      work.read_text(), re.S).group(1)
    written = [[int(v) for v in r.split(",")] for r in re.findall(r"\{([^}]*)\}", block)]
    assert [r[config.column] for r in written] == values
    others = [v for r in written for i, v in enumerate(r) if i != config.column]
    if context == "zero":
        assert set(others) == {0}
    else:
        assert all(-7 <= v <= 7 for v in others) and len(set(others)) > 1

    before = pristine.read_text().splitlines()
    after = work.read_text().splitlines()
    assert len(before) == len(after)
    start = next(i for i, l in enumerate(before) if "L1_W[NET_L1][NET_INPUTS]" in l)
    moved = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
    assert all(start < i <= start + config.neurons for i in moved), \
        f"patch changed lines outside L1_W: {moved}"
