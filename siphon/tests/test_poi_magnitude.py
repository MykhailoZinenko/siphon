"""Regression: magnitude-graded POI recovers small weights without breaking the reference net.

The SNR-over-v criterion rewards sign-only leak points, which carry no magnitude
information; on a net whose active weights are all +/-2 that drops layer-1 to
40/60 (sign right, magnitude arbitrary). The magnitude-graded criterion must
recover >=50/60 there while leaving seed-4242 at a full 60/60.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from siphon.progress import Reporter
from siphon.extract import layer1_weights

ROOT = Path(__file__).resolve().parents[1]          # .../siphon
RB = ROOT / "data" / "rebuilt_firmware"
HDR = ROOT / "firmware" / "source" / "network_config.h"


def _recover(campaigns, attack_dir, truth_path, prefix="attack_c"):
    truth = np.array(json.loads(Path(truth_path).read_text())["L1_W"])
    cols = [c for c in range(10)
            if (attack_dir / f"{prefix}{c}" / "_cache_300_0_10000.npy").exists()]
    if not cols:
        pytest.skip(f"no cached attack traces under {attack_dir}")
    attacks = {c: attack_dir / f"{prefix}{c}" for c in cols}
    W = np.array(layer1_weights(campaigns, attacks, HDR, 300, Reporter()))
    ok = sum(int(W[n][i] == truth[n][c]) for i, c in enumerate(cols) for n in range(6))
    return ok, 6 * len(cols)


@pytest.mark.skipif(not (RB / "c4242_1").exists(), reason="seed-4242 data absent")
def test_seed4242_stays_full():
    ok, tot = _recover([(RB / "c4242_1", 1)], RB / "target4242",
                       ROOT / "data/targets/seed4242.json")
    assert ok == tot, f"seed-4242 regressed: {ok}/{tot}"


@pytest.mark.skipif(not (ROOT / "data/phase2/campaign").exists(), reason="phase2 data absent")
def test_phase2_recovers_more():
    ok, tot = _recover([(ROOT / "data/phase2/campaign", 1)], ROOT / "data/phase2",
                       ROOT / "data/phase2/truth.json")
    assert ok >= 50, f"phase2 only {ok}/{tot}; magnitude POI should reach >=50"
