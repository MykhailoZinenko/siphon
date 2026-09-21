"""Capture traces for the Binary_Codes quantized UNPROTECTED attack on our own CW-Lite + STM32F303.
Flashes their firmware (10->6->1, 4-bit) and captures with their protocol: input[1] = random nonzero
weight value in [-7,7], all other inputs 0 (hot-spot). Saves trace_i.txt + inputs.txt (their format).
Run: python capture.py
"""
import os
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import (  # noqa: E402
    DATA, FIRMWARE, STUDENT_ATTACK, STUDENT_PROFILING, REBUILT_ATTACK,
    CAMPAIGN_WEIGHT, CAMPAIGN_COLUMN,
)
import random
import time

import numpy as np
import chipwhisperer as cw
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
HEX = os.path.join(HERE, "firmware-CWLITEARM.hex")
OUT = os.path.join(HERE, "traces")
NUM_TRACES = 5000
SAMPLES = 24430
ALLOWED_INPUTS = [-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]   # nonzero 4-bit values

scope = cw.scope()
scope.default_setup()
target = cw.target(scope, cw.targets.SimpleSerial2)
scope.adc.samples = SAMPLES
scope.adc.decimate = 1

print("programming target with the reference unprotected firmware...")
cw.program_target(scope, cw.programmers.STM32FProgrammer, HEX)


def capture_trace(cmd_data):
    scope.arm()
    target.flush()
    target.send_cmd("p", 0, cmd_data)          # scmd=0 -> unprotected
    scope.capture()
    wave = scope.get_last_trace()
    target.read_cmd("r")
    target.read_cmd("e")
    return wave


def build_cmd(vals):
    return bytearray([x & 0xFF for x in vals])  # 10 signed int8 bytes


# warm-up
for _ in range(50):
    capture_trace(build_cmd([random.choice(ALLOWED_INPUTS)] * 10))
print("warm-up done")

os.makedirs(OUT, exist_ok=True)
inputs = []
start = time.time()
for i in tqdm(range(NUM_TRACES), desc="capturing", ncols=70):
    vec = [0] * 10
    vec[1] = random.choice(ALLOWED_INPUTS)       # hot-spot on input column 1
    inputs.append(vec)
    wave = capture_trace(build_cmd(vec))
    with open(os.path.join(OUT, f"trace_{i}.txt"), "w") as f:
        f.write("\n".join(str(s) for s in wave) + "\n")

with open(os.path.join(OUT, "inputs.txt"), "w") as f:
    for vec in inputs:
        f.write(" ".join(str(x) for x in vec) + "\n")

scope.dis(); target.dis()
print(f"saved {NUM_TRACES} traces ({SAMPLES} samples) + inputs to {OUT}  ({time.time()-start:.1f}s)")
