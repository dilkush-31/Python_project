"""SQLite storage for lane-wise vehicle counts + the SQL analytics on top of it."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS density (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     TEXT    NOT NULL,          -- ISO timestamp 'YYYY-MM-DD HH:MM:SS'
    lane   TEXT    NOT NULL,          -- approach name, e.g. N / E / S / W
    count  INTEGER NOT NULL,          -- vehicles detected in that lane region
    source TEXT    NOT NULL           -- 'video:<file>' or 'synthetic'
);
CREATE INDEX IF NOT EXISTS idx_density_ts   ON density(ts);
CREATE INDEX IF NOT EXISTS idx_density_lane ON density(lane);
"""


@contextmanager
def connect(db_path: str | Path):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def insert_counts(db_path, rows: Iterable[tuple[str, str, int, str]]) -> int:
    rows = list(rows)
    with connect(db_path) as con:
        con.executemany("INSERT INTO density(ts, lane, count, source) VALUES (?,?,?,?)", rows)
    return len(rows)


def clear(db_path, source_prefix: str | None = None) -> None:
    with connect(db_path) as con:
        if source_prefix:
            con.execute("DELETE FROM density WHERE source LIKE ?", (source_prefix + "%",))
        else:
            con.execute("DELETE FROM density")


def _df(db_path, sql: str, params: tuple = ()) -> pd.DataFrame:
    with connect(db_path) as con:
        return pd.read_sql_query(sql, con, params=params)


# ----------------------------------------------------------------------------
# Analytics queries (CTEs, GROUP BY, window functions)
# ----------------------------------------------------------------------------
def mean_density(db_path, hour_from: int | None = None, hour_to: int | None = None) -> dict[str, float]:
    """Average vehicles-in-frame per lane, optionally restricted to an hour window [from, to]."""
    sql = "SELECT lane, AVG(count) AS avg_count FROM density"
    params: tuple = ()
    if hour_from is not None and hour_to is not None:
        sql += " WHERE CAST(strftime('%H', ts) AS INTEGER) BETWEEN ? AND ?"
        params = (hour_from, hour_to)
    sql += " GROUP BY lane ORDER BY lane"
    df = _df(db_path, sql, params)
    return dict(zip(df["lane"], df["avg_count"]))


def hourly_density(db_path) -> pd.DataFrame:
    """Average count per lane per hour of day."""
    sql = """
    SELECT CAST(strftime('%H', ts) AS INTEGER) AS hour, lane, ROUND(AVG(count), 2) AS avg_count
    FROM density
    GROUP BY hour, lane
    ORDER BY hour, lane
    """
    return _df(db_path, sql)


def peak_hours(db_path, top_n: int = 3) -> pd.DataFrame:
    """Top-N busiest hours per lane, using a CTE + RANK() window function."""
    sql = """
    WITH hourly AS (
        SELECT lane,
               CAST(strftime('%H', ts) AS INTEGER) AS hour,
               AVG(count) AS avg_count
        FROM density
        GROUP BY lane, hour
    ),
    ranked AS (
        SELECT lane, hour, ROUND(avg_count, 2) AS avg_count,
               RANK() OVER (PARTITION BY lane ORDER BY avg_count DESC) AS rnk
        FROM hourly
    )
    SELECT lane, hour, avg_count, rnk FROM ranked WHERE rnk <= ? ORDER BY lane, rnk
    """
    return _df(db_path, sql, (top_n,))


def lane_share(db_path) -> pd.DataFrame:
    """Each lane's share of total traffic (window SUM over all lanes)."""
    sql = """
    WITH per_lane AS (SELECT lane, SUM(count) AS total FROM density GROUP BY lane)
    SELECT lane, total,
           ROUND(100.0 * total / SUM(total) OVER (), 1) AS pct_of_traffic
    FROM per_lane ORDER BY total DESC
    """
    return _df(db_path, sql)


def rolling_density(db_path, lane: str, window: int = 6) -> pd.DataFrame:
    """Moving average of count for one lane (window function with ROWS frame)."""
    sql = f"""
    SELECT ts, count,
           ROUND(AVG(count) OVER (ORDER BY ts ROWS BETWEEN {int(window) - 1} PRECEDING AND CURRENT ROW), 2) AS moving_avg
    FROM density WHERE lane = ? ORDER BY ts
    """
    return _df(db_path, sql, (lane,))
