import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic import db, synthetic
from traffic.detector import Detection, count_per_lane, default_lanes
from traffic.ga import GAConfig, fixed_timer, proportional_timer, run_ga
from traffic.simulator import LANES, SimConfig, evaluate, rates_from_density, simulate


# ----------------------------------------------------------------- detector / lanes
def _lanes():
    return {k: np.array(v, dtype=np.int32) for k, v in default_lanes(100, 100).items()}


def test_vehicle_assigned_to_lane_by_bottom_centre():
    # box whose bottom-centre (25, 40) lies in the top-left quadrant = lane N
    dets = [Detection((10, 10, 40, 40), "car", 0.9)]
    c = count_per_lane(dets, _lanes())
    assert c["N"] == 1 and sum(c.values()) == 1


def test_pcu_weighting():
    dets = [Detection((60, 60, 90, 90), "bus", 0.9), Detection((60, 60, 90, 90), "motorcycle", 0.9)]
    c = count_per_lane(dets, _lanes(), weighted=True)
    assert c["S"] == pytest.approx(2.9)


# ------------------------------------------------------------------------------ SQL
def test_sql_peak_hour_and_share(tmp_path):
    path = tmp_path / "t.db"
    rows = [("2026-01-01 08:00:00", "N", 10, "t"), ("2026-01-01 08:30:00", "N", 14, "t"),
            ("2026-01-01 12:00:00", "N", 2, "t"), ("2026-01-01 08:00:00", "E", 4, "t")]
    db.insert_counts(path, rows)
    peaks = db.peak_hours(path, 1)
    assert peaks[peaks.lane == "N"].iloc[0].hour == 8
    assert db.mean_density(path, 8, 8)["N"] == 12
    share = db.lane_share(path).set_index("lane")["pct_of_traffic"]
    assert share.sum() == pytest.approx(100, abs=0.2)


def test_synthetic_has_evening_peak_on_east(tmp_path):
    path = tmp_path / "s.db"
    synthetic.generate(path, days=2)
    pk = db.peak_hours(path, 1)
    assert 16 <= pk[pk.lane == "E"].iloc[0].hour <= 19
    assert 7 <= pk[pk.lane == "N"].iloc[0].hour <= 10


# ------------------------------------------------------------------------ simulator
def test_rates_scale_to_demand():
    r = rates_from_density({"N": 6, "E": 2, "S": 2, "W": 0}, 1000)
    assert sum(r.values()) * 3600 == pytest.approx(1000)
    assert r["N"] > r["E"] > r["W"]


def test_simulation_is_deterministic_per_seed():
    r = rates_from_density({"N": 5, "E": 3, "S": 4, "W": 2})
    assert simulate([30] * 4, r, seed=3) == simulate([30] * 4, r, seed=3)


def test_more_green_for_heavy_lane_reduces_delay():
    r = rates_from_density({"N": 10, "E": 1, "S": 1, "W": 1}, 900)
    cfg = SimConfig()
    equal = evaluate([30, 30, 30, 30], r, cfg)["avg_delay_s"]
    skewed = evaluate([55, 10, 10, 10], r, cfg)["avg_delay_s"]
    assert skewed < equal


# ------------------------------------------------------------------------------- GA
def test_ga_beats_fixed_timer_and_respects_bounds():
    r = rates_from_density({"N": 10, "E": 2, "S": 6, "W": 1}, 1200)
    cfg = GAConfig(pop_size=20, generations=12)
    res = run_ga(r, SimConfig(), cfg)
    assert all(cfg.g_min <= g <= cfg.g_max for g in res.best_greens)
    assert res.history_best[-1] <= res.history_best[0]          # elitism => monotone
    held = (500, 501, 502)
    ga = evaluate(res.best_greens, r, SimConfig(), held)["avg_delay_s"]
    fixed = evaluate(fixed_timer(), r, SimConfig(), held)["avg_delay_s"]
    assert ga < fixed


def test_proportional_baseline_respects_bounds():
    g = proportional_timer({"N": 0.5, "E": 0.01, "S": 0.2, "W": 0.01})
    assert min(g) >= 8 and max(g) <= 60
