from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np


@dataclass
class RunResult:
    algo_id: str
    name: str
    mode: str
    best_position: np.ndarray
    best_fitness: float
    convergence_curve: List[float]
    a_curve: List[float]
    diversity_curve: List[float]
    trajectory_history: List[List[List[float]]] = field(default_factory=list)
    leaders_history: List[List[List[float]]] = field(default_factory=list)
    initial_pop_audit: List[Dict] = field(default_factory=list)
    init_best_pop_value: Optional[float] = None
    runtime_ms: float = 0.0
    convergence_iter: int = 0


def is_better(candidate: float, reference: float, mode: str) -> bool:
    return candidate < reference if mode == "min" else candidate > reference


def pack_diversity(positions: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> float:
    span = np.maximum(ub - lb, 1e-9)
    norm_pos = (positions - lb) / span
    centroid = np.mean(norm_pos, axis=0)
    return float(np.mean(np.linalg.norm(norm_pos - centroid, axis=1)))


def convergence_iteration(curve: List[float], mode: str) -> int:
    init_f, final_f = curve[0], curve[-1]
    if mode == "min":
        target = final_f + 0.05 * max(0.0, init_f - final_f)
        return next((i for i, v in enumerate(curve) if v <= target), len(curve) - 1)
    target = final_f - 0.05 * max(0.0, final_f - init_f)
    return next((i for i, v in enumerate(curve) if v >= target), len(curve) - 1)


def rank_normalize(values: np.ndarray, mode: str) -> np.ndarray:
    n = len(values)
    if n <= 1:
        return np.ones(n, dtype=float)
    order = np.argsort(values) if mode == "min" else np.argsort(values)[::-1]
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.linspace(1.0, 0.0, n)
    return ranks


def default_pop_value(x: np.ndarray) -> float:
    return float(np.mean(x))
