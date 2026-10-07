"""Discrete-time (1 s) simulator of a 4-approach, 4-phase signalised intersection.

Phases run in order N -> E -> S -> W. Each phase i gets ``greens[i]`` seconds of green followed by
``clearance`` seconds of yellow/all-red. Vehicles arrive as a Poisson process; during green the queue
discharges at the saturation flow rate after a start-up lost time.

The key output is *average delay per vehicle* (seconds) = total queue-seconds / vehicles arrived,
which is Little's law applied to the queues.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LANES = ("N", "E", "S", "W")


@dataclass(frozen=True)
class SimConfig:
    horizon_s: int = 1800          # simulated seconds per run
    sat_flow: float = 0.5          # veh/s discharged while green (=1800 veh/h/lane)
    startup_lost_s: int = 2        # seconds at the start of green with no discharge
    clearance_s: int = 4           # yellow + all-red after every phase


def rates_from_density(density: dict[str, float], total_veh_per_hour: float = 1200.0) -> dict[str, float]:
    """Turn measured lane densities (e.g. mean vehicles in frame) into arrival rates in veh/s.

    Camera counts give *relative* demand; ``total_veh_per_hour`` sets the absolute scale.
    """
    vals = np.array([max(float(density.get(l, 0.0)), 0.0) for l in LANES])
    if vals.sum() <= 0:
        vals = np.ones(len(LANES))
    share = vals / vals.sum()
    return {l: float(s * total_veh_per_hour / 3600.0) for l, s in zip(LANES, share)}


def _schedule(greens, cfg: SimConfig):
    """Per-second arrays: which phase is green, and whether discharge is active (past start-up loss)."""
    phase, active = [], []
    while len(phase) < cfg.horizon_s:
        for i, g in enumerate(greens):
            g = int(round(g))
            for s in range(g):
                phase.append(i)
                active.append(s >= cfg.startup_lost_s)
            for _ in range(cfg.clearance_s):
                phase.append(-1)
                active.append(False)
    return phase[: cfg.horizon_s], active[: cfg.horizon_s]


def simulate(greens, rates: dict[str, float], cfg: SimConfig = SimConfig(), seed: int = 0) -> dict[str, float]:
    """Simulate one run. ``greens`` = green seconds for (N, E, S, W)."""
    rng = np.random.default_rng(seed)
    lam = np.array([rates[l] for l in LANES])
    arrivals = rng.poisson(lam[:, None], size=(len(LANES), cfg.horizon_s)).T.tolist()  # [t][lane]
    phase, active = _schedule(greens, cfg)

    q = [0, 0, 0, 0]
    credit, last_phase = 0.0, -2
    queue_seconds = served = arrived = max_q = 0
    for t in range(cfg.horizon_s):
        a = arrivals[t]
        for i in range(4):
            q[i] += a[i]
        arrived += sum(a)
        ph = phase[t]
        if ph != last_phase:
            credit, last_phase = 0.0, ph
        if active[t]:
            credit += cfg.sat_flow
            k = min(int(credit), q[ph])
            q[ph] -= k
            credit -= k
            served += k
            if q[ph] == 0:
                credit = 0.0
        tq = q[0] + q[1] + q[2] + q[3]
        queue_seconds += tq
        if max(q) > max_q:
            max_q = max(q)
    return {
        "avg_delay_s": queue_seconds / max(arrived, 1),
        "throughput": served,
        "arrived": arrived,
        "left_in_queue": sum(q),
        "max_queue": max_q,
        "cycle_s": sum(int(round(g)) for g in greens) + 4 * cfg.clearance_s,
    }


def evaluate(greens, rates, cfg: SimConfig = SimConfig(), seeds=(0, 1, 2)) -> dict[str, float]:
    """Average ``simulate`` over several random seeds."""
    runs = [simulate(greens, rates, cfg, s) for s in seeds]
    return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
