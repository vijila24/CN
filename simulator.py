from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from allocators import dispatch_allocate


def default_ladder() -> np.ndarray:
    # Mbps ladder typical for e-learning video
    return np.array([0.4, 0.8, 1.5, 3.0, 5.0], dtype=float)


@dataclass
class SimConfig:
    users: int = 20
    seconds: int = 600
    capacity_mean: float = 20.0  # Mbps
    capacity_var: float = 0.3    # fraction of mean for sinusoidal amplitude
    noise_std: float = 1.0       # Mbps std
    alpha_throughput: float = 0.85  # ABR safety factor
    ema: float = 0.7             # throughput estimator EMA
    init_buffer: float = 10.0    # seconds
    max_buffer: float = 30.0     # seconds
    stall_penalty: float = 4.0   # QoE penalty per stall second
    switch_penalty: float = 0.5  # per switch per minute scale
    seed: int = 42


def capacity_trace(cfg: SimConfig) -> np.ndarray:
    t = np.arange(cfg.seconds, dtype=float)
    base = cfg.capacity_mean * (1.0 + cfg.capacity_var * np.sin(2 * np.pi * t / max(cfg.seconds, 1)))
    rng = np.random.default_rng(cfg.seed)
    noise = rng.normal(0.0, cfg.noise_std, size=cfg.seconds)
    cap = np.maximum(base + noise, 0.0)
    return cap


def run_simulation(strategy: str,
                   cfg: SimConfig,
                   weights: Optional[np.ndarray] = None,
                   min_rates: Optional[np.ndarray] = None,
                   bitrate_ladder: Optional[np.ndarray] = None) -> Dict[str, object]:
    n = cfg.users
    ladder = bitrate_ladder if bitrate_ladder is not None else default_ladder()
    cap = capacity_trace(cfg)
    rng = np.random.default_rng(cfg.seed)

    # Initialize per-user state
    ema_tp = np.full(n, ladder[1])  # initial estimated throughput
    bitrate_idx = np.full(n, 1, dtype=int)
    bitrate = ladder[bitrate_idx]
    buffer = np.full(n, cfg.init_buffer)
    stalls = np.zeros(n)
    switches = np.zeros(n)

    # If weights/mins not provided, set defaults
    if weights is None:
        weights = np.ones(n)
    if min_rates is None:
        min_rates = np.zeros(n)

    # Collections
    alloc_hist = np.zeros((cfg.seconds, n))
    bitrate_hist = np.zeros((cfg.seconds, n))
    buffer_hist = np.zeros((cfg.seconds, n))
    cap_hist = cap.copy()

    for t in range(cfg.seconds):
        demand = bitrate.copy()
        alloc = dispatch_allocate(strategy, cap[t], demand, weights=weights, min_rates=min_rates)

        # Update EMA throughput estimate
        ema_tp = cfg.ema * ema_tp + (1 - cfg.ema) * alloc

        # ABR selection: choose highest rung <= alpha * ema_tp
        target_tp = cfg.alpha_throughput * ema_tp
        new_idx = np.minimum.reduce([
            np.searchsorted(ladder, target_tp, side='right') - 1,
            np.full(n, len(ladder) - 1)
        ])
        new_idx = np.maximum(new_idx, 0)
        switches += (new_idx != bitrate_idx).astype(float)
        bitrate_idx = new_idx
        bitrate = ladder[bitrate_idx]

        # Buffer dynamics (1s timestep)
        # If alloc >= bitrate: buffer increases by (alloc/bitrate - 1)
        # If alloc < bitrate: buffer decreases by (1 - alloc/bitrate)
        ratio = np.divide(alloc, np.maximum(bitrate, 1e-9))
        buffer += (ratio - 1.0)
        stalled = buffer < 1e-9
        stalls += stalled.astype(float)
        buffer = np.clip(buffer, 0.0, cfg.max_buffer)

        alloc_hist[t, :] = alloc
        bitrate_hist[t, :] = bitrate
        buffer_hist[t, :] = buffer

    # Metrics
    avg_alloc = alloc_hist.mean(axis=0)
    avg_bitrate = bitrate_hist.mean(axis=0)
    total_stall = stalls  # seconds

    # Jain's fairness on average allocation
    denom = (n * np.square(avg_alloc).sum())
    jain = (np.sum(avg_alloc) ** 2) / denom if denom > 0 else 0.0

    # QoE: bitrate - stall_penalty * stall_rate - switch_penalty * switches_per_min
    stall_rate = total_stall / max(cfg.seconds, 1)
    switches_per_min = switches / max(cfg.seconds / 60.0, 1e-9)
    qoe = avg_bitrate - cfg.stall_penalty * stall_rate - cfg.switch_penalty * switches_per_min

    # DataFrames for plotting
    time = np.arange(cfg.seconds)
    df_alloc = pd.DataFrame(alloc_hist, columns=[f"u{i+1}" for i in range(n)])
    df_alloc["time"] = time
    df_bitrate = pd.DataFrame(bitrate_hist, columns=[f"u{i+1}" for i in range(n)])
    df_bitrate["time"] = time
    df_buffer = pd.DataFrame(buffer_hist, columns=[f"u{i+1}" for i in range(n)])
    df_buffer["time"] = time

    summary = pd.DataFrame({
        "user": [f"u{i+1}" for i in range(n)],
        "avg_alloc_mbps": avg_alloc,
        "avg_bitrate_mbps": avg_bitrate,
        "stall_seconds": total_stall,
        "switches": switches,
        "qoe": qoe,
        "weight": weights,
        "min_rate": min_rates,
    })

    overall = {
        "mean_capacity_mbps": cap.mean(),
        "mean_alloc_mbps": avg_alloc.mean(),
        "fairness_jain": jain,
        "avg_stall_seconds": total_stall.mean(),
        "avg_qoe": qoe.mean(),
    }

    return {
        "capacity": cap_hist,
        "df_alloc": df_alloc,
        "df_bitrate": df_bitrate,
        "df_buffer": df_buffer,
        "summary": summary,
        "overall": overall,
    }
