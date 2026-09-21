"""Shared roots for the legacy scripts.

These scripts produced every result recorded in ../../notes/. They are kept
runnable rather than merely archived, so that anything the new implementation
reports can be checked against the code that originally produced it.
"""
import os
import sys

LEGACY = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(LEGACY)

DATA = os.path.join(ROOT, "data")
FIRMWARE = os.path.join(ROOT, "firmware")

STUDENT_ATTACK = os.path.join(DATA, "student_firmware", "attack")
STUDENT_PROFILING = os.path.join(DATA, "student_firmware", "profiling")
REBUILT_ATTACK = os.path.join(DATA, "rebuilt_firmware", "attack")
CAMPAIGN_WEIGHT = os.path.join(DATA, "rebuilt_firmware", "campaign_single_weight")
CAMPAIGN_COLUMN = os.path.join(DATA, "rebuilt_firmware", "campaign_column")

# the algorithm folders import from each other
for _d in ("alg1", "alg2", "alg3", "alg3_modified"):
    _p = os.path.join(LEGACY, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
