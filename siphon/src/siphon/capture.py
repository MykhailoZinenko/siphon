"""Trace capture on the ChipWhisperer: flash the target, then collect traces."""
from __future__ import annotations

import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path


def _hot_spot_inputs() -> list[int]:
    # The hot-spot input values (layer-1 inputs are int8, so any -128..127 is
    # legal). SIPHON_INPUTS overrides the default with a comma-separated set;
    # positive-only magnitudes are mirrored to negative. Wider magnitudes make
    # v = w*x large enough that the accumulator leaks the weight MAGNITUDE, not
    # only its sign - needed for small weights (see notes/26-magnitude-poi.md).
    env = os.environ.get("SIPHON_INPUTS")
    if not env:
        return [-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]
    vals = sorted({int(v) for v in env.split(",") if v.strip()})
    if all(v > 0 for v in vals):
        vals = sorted(vals + [-v for v in vals])
    return vals


NONZERO_INPUTS = _hot_spot_inputs()
SAMPLES = 24430
WARMUP = 50                     # the first traces after a flash drift


def check_free(out: Path) -> None:
    """Refuse to write into a directory that already holds traces."""
    if out.exists() and any(out.glob("trace_*.txt")):
        raise RuntimeError(f"{out} already holds traces; move it aside first")


@contextmanager
def _quiet(label: str, progress):
    """Suppress the wrapped block's stdout, replaying it only on failure."""
    import io
    from contextlib import redirect_stdout
    buffer = io.StringIO()               # capture stdout instead of printing it
    try:
        with redirect_stdout(buffer):
            yield
    except Exception:
        for line in buffer.getvalue().splitlines():   # replay captured output on error
            progress(line)
        raise
    progress(label)                      # success: report just the short label


@contextmanager
def session(hex_path: Path, progress=lambda _m: None):
    """Flash the target once and yield a `collect(out, count, column)`."""
    import chipwhisperer as cw          # lazy import: keeps analysis working without hardware

    with _quiet("scope ready", progress):
        scope = cw.scope()
        scope.default_setup()
        target = cw.target(scope, cw.targets.SimpleSerial2)
        scope.adc.samples = SAMPLES     # samples per trace
        scope.adc.decimate = 1          # no decimation

    with _quiet("flashed and verified", progress):
        cw.program_target(scope, cw.programmers.STM32FProgrammer, str(hex_path))

    def one(values):
        """Send one input vector, capture the trace, and return it."""
        scope.arm()                     # arm the scope for the next capture
        target.flush()                  # drop any stale serial data
        target.send_cmd("p", 0, bytearray([v & 0xFF for v in values]))  # push the input vector
        scope.capture()                 # block until the triggered capture finishes
        wave = scope.get_last_trace()   # read back the waveform
        target.read_cmd("r")            # drain the firmware's reply
        target.read_cmd("e")
        return wave

    for _ in range(WARMUP):
        one([random.choice(NONZERO_INPUTS)] * 10)   # discard traces while behaviour settles

    def collect(out: Path, count: int, column: int | None = None,
                vectors=None, track=None) -> None:
        """Capture count traces, either hot-spotting on column or replaying vectors."""
        check_free(out)
        out.mkdir(parents=True, exist_ok=True)
        recorded = []
        steps = range(count) if track is None else track(range(count), count)  # optional progress wrap
        for i in steps:
            if vectors is None:
                vector = [0] * 10
                vector[column] = random.choice(NONZERO_INPUTS)   # hot-spot: only this input is nonzero
            else:
                vector = [int(v) for v in vectors[i % len(vectors)]]   # replay a crafted vector
            recorded.append(vector)
            (out / f"trace_{i}.txt").write_text("\n".join(map(str, one(vector))) + "\n")
        (out / "inputs.txt").write_text(
            "".join(" ".join(map(str, v)) + "\n" for v in recorded))   # log every input used

    try:
        yield collect
    finally:
        scope.dis()
        target.dis()



@dataclass
class Step:
    """One capture, plus whatever the firmware must say for it."""
    label: str
    out: Path
    count: int
    column: int | None = None            # hot spot on this input
    vectors: object | None = None        # or replay these input vectors
    arrays: tuple[tuple[str, tuple[int, ...]], ...] = ()   # flat consts, e.g. L1_B
    rows: tuple[tuple[str, int, tuple[int, ...]], ...] = ()  # rows of 2-D consts
    note: str = ""                       # how to say this firmware state in one line

    @property
    def firmware(self):
        """Identity used to group consecutive steps that need the same flash."""
        return (self.arrays, self.rows)

    def describe(self) -> str:
        """One line. A whole patched weight matrix does not belong on a terminal."""
        if self.note:
            return self.note                     # explicit note wins
        if not self.arrays and not self.rows:
            return "real parameters in place"     # nothing patched
        names = [n for n, _ in self.arrays] + sorted({n for n, _, _ in self.rows})
        if len(self.arrays) == 1 and not self.rows:
            name, values = self.arrays[0]
            shown = list(values) if len(values) <= 6 else f"{len(values)} values"
            return f"{name} = {shown}"            # short single array: show its values
        return "patched " + ", ".join(dict.fromkeys(names))   # otherwise just list what changed


def execute(plan: "list[Step]", reporter) -> None:
    """Run a capture plan, reflashing only when consecutive steps' firmware differs."""
    from .firmware import patched

    for step in plan:
        check_free(step.out)            # fail fast before touching the device

    groups: list[tuple[object, list[Step]]] = []
    for step in plan:
        if groups and groups[-1][0] == step.firmware:
            groups[-1][1].append(step)  # same firmware as previous step: same flash
        else:
            groups.append((step.firmware, [step]))

    total = sum(s.count for s in plan)
    reporter.detail(f"{len(plan)} captures, {len(groups)} flash(es), {total} traces")
    patches = any(step.arrays or step.rows for step in plan)
    with patched(f"a capture into {plan[0].out.parent.name}") if patches else nullcontext():
        _run_groups(groups, reporter)


def _run_groups(groups, reporter) -> None:
    """Flash each group once and collect its steps; the header is restored whatever happens."""
    from .firmware import build, restore, set_array, set_row

    try:
        for i, (_, steps) in enumerate(groups, 1):
            restore()                   # start each group from the pristine header
            for name, values in steps[0].arrays:
                set_array(name, list(values))
            for name, row, values in steps[0].rows:
                set_row(name, list(values), row=row)
            reporter.stage(steps[0].describe()
                           if len(groups) == 1 else
                           f"{steps[0].describe()}   [dim]flash {i}/{len(groups)}[/dim]")
            with session(build(), reporter.detail) as collect:
                for step in steps:      # one flash serves every step in the group
                    collect(step.out, step.count, step.column, step.vectors,
                            track=lambda it, n, lbl=step.label:
                                reporter.track(it, lbl, total=n))
    finally:
        restore()                       # always leave the pristine firmware in place
        build()
        reporter.detail("firmware restored")
