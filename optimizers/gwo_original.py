import time
import numpy as np
from optimizers.common import RunResult, is_better, pack_diversity, convergence_iteration


def run_original_gwo(
    obj_func,
    lb,
    ub,
    mode="min",
    n_agents=20,
    max_iter=40,
    seed=42,
    record_2d=False,
    **kwargs,
):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    dim = len(lb)

    positions = lb + rng.random((n_agents, dim)) * (ub - lb)

    worst = np.inf if mode == "min" else -np.inf
    alpha_pos, alpha_score = np.zeros(dim), worst
    beta_pos, beta_score = np.zeros(dim), worst
    delta_pos, delta_score = np.zeros(dim), worst

    convergence_curve = []
    a_curve = []
    diversity_curve = []
    trajectories = []
    leaders = []

    for t in range(max_iter):
        for i in range(n_agents):
            positions[i] = np.clip(positions[i], lb, ub)
            fitness = float(obj_func(positions[i]))

            if is_better(fitness, alpha_score, mode):
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, positions[i].copy()
            elif is_better(fitness, beta_score, mode):
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, positions[i].copy()
            elif is_better(fitness, delta_score, mode):
                delta_score, delta_pos = fitness, positions[i].copy()

        progress = t / max(1, max_iter - 1)
        a = 2.0 * (1.0 - progress)

        convergence_curve.append(float(alpha_score))
        a_curve.append(float(a))
        diversity_curve.append(pack_diversity(positions, lb, ub))
        if record_2d:
            trajectories.append(positions[:, :2].round(4).tolist())
            leaders.append([alpha_pos[:2].round(4).tolist(), beta_pos[:2].round(4).tolist(), delta_pos[:2].round(4).tolist()])

        for i in range(n_agents):
            r1, r2 = rng.random(dim), rng.random(dim)
            A1, C1 = 2.0 * a * r1 - a, 2.0 * r2
            X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - positions[i])

            r1, r2 = rng.random(dim), rng.random(dim)
            A2, C2 = 2.0 * a * r1 - a, 2.0 * r2
            X2 = beta_pos - A2 * np.abs(C2 * beta_pos - positions[i])

            r1, r2 = rng.random(dim), rng.random(dim)
            A3, C3 = 2.0 * a * r1 - a, 2.0 * r2
            X3 = delta_pos - A3 * np.abs(C3 * delta_pos - positions[i])

            positions[i] = np.clip((X1 + X2 + X3) / 3.0, lb, ub)

    return RunResult(
        algo_id="original_gwo",
        name="Original GWO",
        mode=mode,
        best_position=alpha_pos,
        best_fitness=float(alpha_score),
        convergence_curve=convergence_curve,
        a_curve=a_curve,
        diversity_curve=diversity_curve,
        trajectory_history=trajectories,
        leaders_history=leaders,
        runtime_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        convergence_iter=convergence_iteration(convergence_curve, mode),
    )
