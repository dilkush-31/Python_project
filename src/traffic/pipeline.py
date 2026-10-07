"""Glue: SQL densities -> arrival rates -> GA -> held-out comparison against baselines."""
from __future__ import annotations

import json
from pathlib import Path

from . import db, report
from .ga import GAConfig, fixed_timer, proportional_timer, run_ga
from .simulator import LANES, SimConfig, evaluate, rates_from_density

HELD_OUT_SEEDS = tuple(range(200, 210))   # never seen by the GA during training


def optimise(db_path, out_dir="outputs", hour_from=None, hour_to=None, demand_vph=1200.0,
             ga_cfg: GAConfig = GAConfig(), sim_cfg: SimConfig = SimConfig(), use_mlflow=False, verbose=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    density = db.mean_density(db_path, hour_from, hour_to)
    if not density:
        raise RuntimeError("No density data in the database. Run `detect` on a video or `synth` first.")
    rates = rates_from_density(density, demand_vph)
    if verbose:
        win = f"{hour_from:02d}:00-{hour_to:02d}:59" if hour_from is not None else "all day"
        print(f"\nMean density per lane ({win}): " + ", ".join(f"{k}={v:.2f}" for k, v in density.items()))
        print("Arrival rates (veh/h):   " + ", ".join(f"{l}={rates[l] * 3600:.0f}" for l in LANES))

    def cb(gen, best, mean, _ind):
        if verbose and (gen % 5 == 0 or gen == ga_cfg.generations - 1):
            print(f"  gen {gen:3d}  best={best:7.2f}  mean={mean:7.2f}")

    if verbose:
        print("\nRunning Genetic Algorithm ...")
    result = run_ga(rates, sim_cfg, ga_cfg, cb)

    strategies = {
        "Fixed timer": fixed_timer(30.0),
        "Proportional": proportional_timer(rates),
        "GA optimised": result.best_greens,
    }
    table = {name: {**evaluate(g, rates, sim_cfg, HELD_OUT_SEEDS), "greens": [round(float(x), 1) for x in g]}
             for name, g in strategies.items()}

    base = table["Fixed timer"]["avg_delay_s"]
    for name, m in table.items():
        m["improvement_vs_fixed_pct"] = round(100 * (base - m["avg_delay_s"]) / base, 1) if base else 0.0

    if verbose:
        print("\nResults on 10 held-out traffic seeds")
        print(f"{'Strategy':<14}{'Greens N/E/S/W (s)':<26}{'Delay (s)':>10}{'Left in queue':>15}{'vs Fixed':>10}")
        for name, m in table.items():
            g = "/".join(f"{x:.0f}" for x in m["greens"])
            print(f"{name:<14}{g:<26}{m['avg_delay_s']:>10.1f}{m['left_in_queue']:>15.1f}{m['improvement_vs_fixed_pct']:>9.1f}%")

    (out / "results.json").write_text(json.dumps(
        {"density": density, "rates_veh_per_hour": {l: rates[l] * 3600 for l in LANES},
         "ga_config": ga_cfg.__dict__, "results": table}, indent=2, default=list))
    report.plot_convergence(result, out / "ga_convergence.png")
    report.plot_comparison(table, out / "strategy_comparison.png")
    report.plot_hourly(db.hourly_density(db_path), out / "hourly_density.png")

    if use_mlflow:
        try:
            import mlflow

            mlflow.set_experiment("traffic-signal-ga")
            with mlflow.start_run():
                mlflow.log_params({k: v for k, v in ga_cfg.__dict__.items() if k != "train_seeds"})
                mlflow.log_param("demand_vph", demand_vph)
                for i, (b, m) in enumerate(zip(result.history_best, result.history_mean)):
                    mlflow.log_metrics({"best_fitness": b, "mean_fitness": m}, step=i)
                for name, m in table.items():
                    key = name.lower().replace(" ", "_")
                    mlflow.log_metric(f"{key}_avg_delay_s", m["avg_delay_s"])
                mlflow.log_metric("improvement_vs_fixed_pct", table["GA optimised"]["improvement_vs_fixed_pct"])
                mlflow.log_artifacts(str(out))
        except ImportError:
            print("mlflow not installed; skipping tracking (pip install mlflow).")
    return result, table
