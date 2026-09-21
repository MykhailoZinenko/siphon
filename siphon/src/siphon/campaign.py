# The layer-1 profiling schedule: which weight to flash per capture per neuron.
from __future__ import annotations



CAPTURES = 8          # weights 0..7; negatives come from the sign of the input


def column_for(capture_index: int, neurons: int) -> list[int]:
    # Latin square w = (o + k) mod 8: each neuron sees all of 0..7 across
    # captures, while neurons hold different weights within one capture (so
    # their leak points stay distinguishable). Zero is included because a
    # zero weight needs a template for v=0, which profiling 1..7 never produces.
    return [(o + capture_index) % CAPTURES for o in range(neurons)]
