"""Synthetic lane-count generator so the whole pipeline can be run WITHOUT a traffic video.

Creates a realistic daily pattern (morning rush on N/S, evening rush on E/W, noise) and stores it
with source='synthetic' so it can never be confused with real YOLO output.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from . import db
from .simulator import LANES


def _profile(hour: float, lane: str) -> float:
    base = {"N": 6.0, "E": 3.0, "S": 5.0, "W": 2.0}[lane]
    morning = np.exp(-((hour - 9) ** 2) / 4.0)
    evening = np.exp(-((hour - 18) ** 2) / 5.0)
    if lane in ("N", "S"):
        return base * (1 + 1.8 * morning + 0.5 * evening)
    return base * (1 + 0.4 * morning + 2.2 * evening)


def generate(db_path, days: int = 3, every_min: int = 5, seed: int = 7, start: datetime | None = None) -> int:
    rng = np.random.default_rng(seed)
    start = start or datetime(2026, 1, 5, 0, 0, 0)
    rows = []
    t = start
    end = start + timedelta(days=days)
    while t < end:
        h = t.hour + t.minute / 60
        for lane in LANES:
            lam = _profile(h, lane)
            rows.append((t.strftime("%Y-%m-%d %H:%M:%S"), lane, int(rng.poisson(lam)), "synthetic"))
        t += timedelta(minutes=every_min)
    return db.insert_counts(db_path, rows)
