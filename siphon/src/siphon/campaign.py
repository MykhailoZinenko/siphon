# The layer-1 profiling schedule: which weight to flash per capture per neuron.
from __future__ import annotations



CAPTURES = 8          # weights 0..7; negatives come from the sign of the input


def captures_for(hypotheses) -> int:
    # One flash per non-negative weight value: the campaign never flashes a
    # negative weight, the sign of v comes from the sign of the input. So a
    # -7..7 alphabet needs 8 flashes and an unsigned 0..15 one needs 16.
    return max(hypotheses) + 1


JOINT_BIASES = list(range(-56, 57, 8))   # with the unit-weight capture below,
                                         # acc is covered contiguously over +-63


def joint_schedule(neurons: int) -> list[tuple[int, list[int]]]:
    # (bias, weights) per capture, two captures per flashed bias.
    #
    # The unit capture carries weight 1 on every neuron, so acc = x + b sweeps
    # b-7..b+7 with no holes; stepping the bias by 8 makes consecutive windows
    # touch, and the whole swept range is covered. That is what the hypothesis
    # search needs: a pair (w, b) is scored only where its accumulator values
    # were profiled, and a single missing class is enough to lose the true pair
    # to a wrong one (measured while building this).
    #
    # The rotating capture gives each neuron a different weight, which is what
    # separates the neurons' leak points, and builds each acc class out of
    # several (w, x) decompositions - the point is only nearly a function of
    # acc, the decomposition moves its mean by 11-40% of a class gap.
    out = []
    for level, bias in enumerate(JOINT_BIASES):
        out.append((bias, [1] * neurons))                            # coverage
        out.append((bias, [(o + level) % 7 + 1 for o in range(neurons)]))  # separation
    return out


def column_for(capture_index: int, neurons: int, captures: int = CAPTURES) -> list[int]:
    # Latin square w = (o + k) mod captures: each neuron sees every value across
    # captures, while neurons hold different weights within one capture (so
    # their leak points stay distinguishable). Zero is included because a
    # zero weight needs a template for v=0, which profiling 1..7 never produces.
    return [(o + capture_index) % captures for o in range(neurons)]
