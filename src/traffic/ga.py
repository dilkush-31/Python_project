"""Genetic Algorithm that searches for the best green time of each phase.

Chromosome = [g_N, g_E, g_S, g_W] (seconds, real-valued, clipped to [g_min, g_max]).
Fitness    = mean vehicle delay from the simulator (lower is better), averaged over several random
             traffic seeds so the GA does not over-fit to one lucky arrival pattern.
Operators  = tournament selection, blend (BLX-alpha) crossover, Gaussian mutation, elitism.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .simulator import LANES, SimConfig, evaluate


@dataclass
class GAConfig:
    pop_size: int = 40
    generations: int = 40
    elite: int = 2
    tournament_k: int = 3
    crossover_alpha: float = 0.3
    mutation_rate: float = 0.3
    mutation_sigma: float = 4.0
    g_min: float = 8.0
    g_max: float = 60.0
    train_seeds: tuple = (100, 101, 102)
    seed: int = 42


@dataclass
class GAResult:
    best_greens: list
    best_fitness: float
    history_best: list = field(default_factory=list)
    history_mean: list = field(default_factory=list)


def _fitness(ind, rates, sim_cfg, seeds):
    m = evaluate(ind, rates, sim_cfg, seeds)
    # tiny penalty on vehicles still stuck at the end so the GA can't "win" by ignoring a lane
    return m["avg_delay_s"] + 0.05 * m["left_in_queue"]


def run_ga(rates, sim_cfg: SimConfig = SimConfig(), cfg: GAConfig = GAConfig(), callback=None) -> GAResult:
    rng = np.random.default_rng(cfg.seed)
    n = len(LANES)
    pop = rng.uniform(cfg.g_min, cfg.g_max, size=(cfg.pop_size, n))
    best_hist, mean_hist = [], []
    best_ind, best_fit = None, np.inf

    for gen in range(cfg.generations):
        fit = np.array([_fitness(ind, rates, sim_cfg, cfg.train_seeds) for ind in pop])
        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]
        if fit[0] < best_fit:
            best_fit, best_ind = float(fit[0]), pop[0].copy()
        best_hist.append(float(fit[0]))
        mean_hist.append(float(fit.mean()))
        if callback:
            callback(gen, float(fit[0]), float(fit.mean()), pop[0])

        nxt = [pop[i].copy() for i in range(cfg.elite)]                      # elitism
        while len(nxt) < cfg.pop_size:
            p1 = _tournament(pop, fit, cfg.tournament_k, rng)
            p2 = _tournament(pop, fit, cfg.tournament_k, rng)
            lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
            span = hi - lo
            child = rng.uniform(lo - cfg.crossover_alpha * span, hi + cfg.crossover_alpha * span)  # BLX-alpha
            mask = rng.random(n) < cfg.mutation_rate
            child = child + mask * rng.normal(0, cfg.mutation_sigma, n)                              # mutation
            nxt.append(np.clip(child, cfg.g_min, cfg.g_max))
        pop = np.array(nxt)

    return GAResult([round(float(g), 1) for g in best_ind], best_fit, best_hist, mean_hist)


def _tournament(pop, fit, k, rng):
    idx = rng.integers(0, len(pop), size=k)
    return pop[idx[np.argmin(fit[idx])]]


# --------------------------------------------------------------------------- baselines
def fixed_timer(green: float = 30.0):
    """What most real junctions do without sensing: the same green for every approach."""
    return [green] * len(LANES)


def proportional_timer(rates, cycle_green_total: float = 120.0, g_min: float = 8.0, g_max: float = 60.0):
    """Stronger heuristic baseline: split a fixed green budget in proportion to demand."""
    r = np.array([rates[l] for l in LANES])
    g = np.clip(cycle_green_total * r / r.sum(), g_min, g_max)
    return [round(float(x), 1) for x in g]
