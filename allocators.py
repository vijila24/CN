from typing import Optional
import numpy as np


def _waterfill(capacity: float, caps: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted water-filling up to caps. If weights is None, use equal weights."""
    n = len(caps)
    if n == 0 or capacity <= 0:
        return np.zeros_like(caps)
    caps = np.maximum(caps.astype(float), 0.0)
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.maximum(weights.astype(float), 1e-12)

    # Active set method
    active = np.ones(n, dtype=bool)
    alloc = np.zeros(n, dtype=float)
    remaining = capacity
    while remaining > 1e-9 and np.any(active):
        w_active = w[active]
        caps_active = caps[active] - alloc[active]
        if caps_active.sum() <= 1e-12:
            break
        share = remaining * (w_active / w_active.sum())
        give = np.minimum(share, caps_active)
        alloc[active] += give
        remaining = capacity - alloc.sum()
        # Remove those that hit cap
        new_active = np.zeros_like(active)
        new_active[active] = (alloc[active] < caps[active] - 1e-9)
        active = new_active
    return alloc


def allocate_equal(capacity: float, demand: np.ndarray) -> np.ndarray:
    return _waterfill(capacity, demand, None)


def allocate_weighted(capacity: float, demand: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    return _waterfill(capacity, demand, weights)


def allocate_proportional_fair(capacity: float, demand: np.ndarray) -> np.ndarray:
    """Approximate PF via water-filling on caps=demand with equal weights.
    In continuous form, PF is water-filling in log domain; with caps it reduces well to equal weight fill.
    """
    return _waterfill(capacity, demand, None)


def allocate_min_guarantee(capacity: float, demand: np.ndarray, min_rates: Optional[np.ndarray], weights: Optional[np.ndarray]) -> np.ndarray:
    n = len(demand)
    if n == 0:
        return np.zeros_like(demand)
    d = np.maximum(demand.astype(float), 0.0)
    mins = np.maximum(min_rates.astype(float), 0.0) if min_rates is not None else np.zeros(n)
    mins = np.minimum(mins, d)

    cap = float(capacity)
    if cap <= 0:
        return np.zeros_like(d)

    sum_mins = mins.sum()
    if sum_mins >= cap:
        # Not enough to satisfy all mins: allocate proportional to mins
        w = mins / (sum_mins + 1e-12)
        return cap * w

    # First give everyone their minimum
    alloc = mins.copy()
    remaining = cap - sum_mins
    # Now distribute surplus respecting remaining caps and optional weights
    remaining_caps = np.maximum(d - alloc, 0.0)
    extra = _waterfill(remaining, remaining_caps, weights)
    alloc += extra
    return alloc


STRATEGY_NAMES = [
    "Equal",
    "Weighted",
    "Proportional Fair",
    "Min-Guarantee",
]


def dispatch_allocate(name: str, capacity: float, demand: np.ndarray, weights: Optional[np.ndarray] = None, min_rates: Optional[np.ndarray] = None) -> np.ndarray:
    n = len(demand)
    if n == 0:
        return np.zeros_like(demand)
    if name == "Equal":
        return allocate_equal(capacity, demand)
    if name == "Weighted":
        return allocate_weighted(capacity, demand, weights)
    if name == "Proportional Fair":
        return allocate_proportional_fair(capacity, demand)
    if name == "Min-Guarantee":
        return allocate_min_guarantee(capacity, demand, min_rates, weights)
    raise ValueError(f"Unknown strategy: {name}")
