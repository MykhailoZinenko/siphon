"""Patch constants in network_config.h and rebuild the target firmware."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from .config import PACKAGE_ROOT

FIRMWARE = PACKAGE_ROOT / "firmware"
SOURCE = FIRMWARE / "source"
CONFIG_HEADER = SOURCE / "network_config.h"   # the header the configs read too
PRISTINE = FIRMWARE / "network_config.h.orig"
HEX = SOURCE / "simpleserial-target-CWLITEARM.hex"
TOOLCHAIN = Path("/Applications/ArmGNUToolchain/15.2.rel1/arm-none-eabi/bin")


def set_column(values: list[int], column: int) -> list[int]:
    """Patch L1_W[o][column] = values[o] for every neuron; returns the values read back."""
    lines = CONFIG_HEADER.read_text().splitlines(keepends=True)
    start = next((i + 1 for i, l in enumerate(lines)
                  if "L1_W[NET_L1][NET_INPUTS]" in l and "{" in l), None)  # line after declaration
    if start is None:
        raise RuntimeError("L1_W declaration not found in network_config.h")

    for offset, value in enumerate(values):
        row = lines[start + offset]
        inner = re.search(r"\{([^}]*)\}", row)     # this row's braced numbers
        if inner is None:
            raise RuntimeError(f"no braces on weight row {offset}: {row!r}")
        nums = [int(t.strip()) for t in inner.group(1).split(",")]
        nums[column] = value                       # overwrite just this column
        lines[start + offset] = (row[:inner.start()] + "{ " + ", ".join(map(str, nums))
                                 + " }" + row[inner.end():])
    CONFIG_HEADER.write_text("".join(lines))

    fresh = CONFIG_HEADER.read_text().splitlines()
    written = [int(re.search(r"\{([^}]*)\}", fresh[start + o]).group(1).split(",")[column])
               for o in range(len(values))]         # read back what was actually written
    if written != list(values):
        raise RuntimeError(f"patch failed: wanted {values}, header now has {written}")
    return written


def set_array(name: str, values: list[int]) -> list[int]:
    """Overwrite a flat const array's braced values in network_config.h; returns what was read back."""
    lines = CONFIG_HEADER.read_text().splitlines(keepends=True)
    hit = next((i for i, l in enumerate(lines) if f"{name}[" in l and "=" in l), None)  # declaration line
    if hit is None:
        raise RuntimeError(f"{name} declaration not found in network_config.h")
    inner = re.search(r"\{([^}]*)\}", lines[hit])   # its braced values
    if inner is None:
        raise RuntimeError(f"no braces on {name} line: {lines[hit]!r}")
    if len(inner.group(1).split(",")) != len(values):
        raise RuntimeError(f"{name} has {len(inner.group(1).split(','))} entries, "
                           f"got {len(values)}")
    lines[hit] = (lines[hit][:inner.start()] + "{ " + ", ".join(map(str, values))
                  + " }" + lines[hit][inner.end():])
    CONFIG_HEADER.write_text("".join(lines))

    fresh = CONFIG_HEADER.read_text().splitlines()
    written = [int(t.strip()) for t in
               re.search(r"\{([^}]*)\}", fresh[hit]).group(1).split(",")]  # read back to verify
    if written != list(values):
        raise RuntimeError(f"patch failed: wanted {values}, header now has {written}")
    return written


def set_row(name: str, values: list[int], row: int = 0) -> list[int]:
    """Overwrite row `row` of a 2-D const array, e.g. L2_W[0]; returns what was read back."""
    lines = CONFIG_HEADER.read_text().splitlines(keepends=True)
    start = next((i + 1 for i, l in enumerate(lines) if f"{name}[" in l and "{" in l), None)  # line after declaration
    if start is None:
        raise RuntimeError(f"{name} declaration not found in network_config.h")
    target = lines[start + row]
    inner = re.search(r"\{([^}]*)\}", target)      # this row's braced values
    if inner is None:
        raise RuntimeError(f"no braces on {name} row {row}: {target!r}")
    if len(inner.group(1).split(",")) != len(values):
        raise RuntimeError(f"{name} row {row} has "
                           f"{len(inner.group(1).split(','))} entries, got {len(values)}")
    lines[start + row] = (target[:inner.start()] + "{ " + ", ".join(map(str, values))
                          + " }" + target[inner.end():])
    CONFIG_HEADER.write_text("".join(lines))

    fresh = CONFIG_HEADER.read_text().splitlines()
    written = [int(t.strip()) for t in
               re.search(r"\{([^}]*)\}", fresh[start + row]).group(1).split(",")]  # read back to verify
    if written != list(values):
        raise RuntimeError(f"patch failed: wanted {values}, header now has {written}")
    return written


def restore() -> None:
    """Copy the pristine header back over network_config.h."""
    shutil.copy(PRISTINE, CONFIG_HEADER)


def build() -> Path:
    """Cross-compile the firmware; returns the path to the built .hex."""
    env = dict(os.environ, PATH=f"{TOOLCHAIN}:{os.environ['PATH']}")   # prepend the ARM toolchain
    result = subprocess.run(["make", "PLATFORM=CWLITEARM", "CRYPTO_TARGET=NONE"],
                            cwd=SOURCE, env=env, capture_output=True, text=True)
    if result.returncode != 0 or not HEX.exists():
        raise RuntimeError(f"firmware build failed:\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
    return HEX
