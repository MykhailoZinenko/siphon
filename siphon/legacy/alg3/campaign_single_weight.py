"""
Profiling campaign for the stochastic leakage model.

Reflashes L1_W[0][1] = w for w = 1..7, capturing traces at each weight, then a
final attack set at w = 4. Every capture uses the SAME build, so the POI is
stable across the whole campaign and all (v, leakage) pairs come from one
instruction — which is what makes the coefficients identifiable.

This is the book's P-DPA Step 2 taken literally: profile with random keys.
"""
import os
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import (  # noqa: E402
    DATA, FIRMWARE, STUDENT_ATTACK, STUDENT_PROFILING, REBUILT_ATTACK,
    CAMPAIGN_WEIGHT, CAMPAIGN_COLUMN,
)
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

def _find_root(start):
    """Walk up until we find the venv, instead of hardcoding directory depth."""
    p = os.path.abspath(start)
    while p != os.path.dirname(p):
        if os.path.isfile(os.path.join(p, "venv", "bin", "python")):
            return p
        p = os.path.dirname(p)
    raise SystemExit("could not locate venv above " + start)


ROOT = _find_root(HERE)
FW = os.path.join(HERE, "firmware_src")
CFG = os.path.join(FW, "network_config.h")
HEX = os.path.join(FW, "simpleserial-target-CWLITEARM.hex")
ARMBIN = "/Applications/ArmGNUToolchain/15.2.rel1/arm-none-eabi/bin"
PY = os.path.join(ROOT, "venv", "bin", "python")

PROFILING_WEIGHTS = [1, 2, 3, 4, 5, 6, 7]
PROFILING_COUNT = 1000
ATTACK_WEIGHT = 4
ATTACK_COUNT = 5000


def set_weight(w):
    """Rewrite L1_W[0][1]. Line-based: the row we want is the line right after
    the array declaration, which avoids matching braces elsewhere in the file."""
    lines = open(CFG).read().splitlines(keepends=True)
    row_idx = None
    for i, line in enumerate(lines):
        if "L1_W[NET_L1][NET_INPUTS]" in line and "{" in line:
            row_idx = i + 1
            break
    if row_idx is None:
        raise SystemExit("L1_W declaration not found in network_config.h")

    row = lines[row_idx]
    inner = re.search(r"\{([^}]*)\}", row)
    if inner is None:
        raise SystemExit(f"no braces on the row after the declaration: {row!r}")
    vals = [int(v.strip()) for v in inner.group(1).split(",")]
    vals[1] = w
    lines[row_idx] = (row[:inner.start()] + "{ " + ", ".join(str(v) for v in vals)
                      + " }" + row[inner.end():])
    open(CFG, "w").write("".join(lines))

    verify = re.search(r"\{([^}]*)\}", open(CFG).read().splitlines()[row_idx])
    got = [int(v.strip()) for v in verify.group(1).split(",")]
    assert got[1] == w, f"patch failed: row is {got}"
    return got


def build():
    env = dict(os.environ, PATH=f"{ARMBIN}:{os.environ['PATH']}")
    r = subprocess.run(["make", "PLATFORM=CWLITEARM", "CRYPTO_TARGET=NONE"],
                       cwd=FW, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(HEX):
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("build failed")


def capture(out, count):
    r = subprocess.run([PY, os.path.join(HERE, "capture_generic.py"),
                        "--hex", HEX, "--out", out, "--count", str(count), "--force"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("capture failed")
    return r.stdout.strip().splitlines()[-1]


def main():
    started = time.time()
    try:
        for w in PROFILING_WEIGHTS:
            print(f"[w={w:+d}] patching, building, capturing {PROFILING_COUNT} ...", flush=True)
            set_weight(w)
            build()
            out = os.path.join(CAMPAIGN_WEIGHT, f"w{w}")
            print("        " + capture(out, PROFILING_COUNT), flush=True)

        print(f"[attack] weight {ATTACK_WEIGHT}, {ATTACK_COUNT} traces ...", flush=True)
        set_weight(ATTACK_WEIGHT)
        build()
        out = REBUILT_ATTACK
        print("        " + capture(out, ATTACK_COUNT), flush=True)
    finally:
        # always leave the config as we found it
        shutil.copy(os.path.join(HERE, "network_config.h.orig"), CFG)
        build()
        print("[*] config restored to original and rebuilt", flush=True)

    print(f"[*] campaign done in {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
