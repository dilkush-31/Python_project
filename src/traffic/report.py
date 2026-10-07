"""Plots for the README / resume screenshots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .simulator import LANES


def plot_convergence(result, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result.history_best, label="best of generation", lw=2)
    ax.plot(result.history_mean, label="population mean", ls="--")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (avg delay s + stuck penalty)")
    ax.set_title("Genetic Algorithm convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(table: dict[str, dict], path):
    """table = {'Fixed timer': metrics, ...}; bar chart of average delay."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    names = list(table)
    delays = [table[n]["avg_delay_s"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, delays, color=["#c0392b", "#e67e22", "#27ae60"][: len(names)])
    for b, d in zip(bars, delays):
        ax.text(b.get_x() + b.get_width() / 2, d, f"{d:.1f}s", ha="center", va="bottom")
    ax.set_ylabel("Average delay per vehicle (s)")
    ax.set_title("Signal strategies on held-out traffic")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hourly(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for lane in LANES:
        d = df[df["lane"] == lane]
        ax.plot(d["hour"], d["avg_count"], marker="o", ms=3, label=lane)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Avg vehicles in lane region")
    ax.set_title("Lane density by hour (from SQL)")
    ax.legend(title="Lane")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
