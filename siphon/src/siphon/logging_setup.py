"""File logging alongside the console output."""
from __future__ import annotations

import logging
from pathlib import Path


def setup(log_file: Path) -> logging.Logger:
    """(Re)configure the "siphon" logger to write timestamped lines to log_file only."""
    logger = logging.getLogger("siphon")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # drop handlers from a previous run in this process
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                                           datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False  # don't also spam the root logger/console
    return logger
