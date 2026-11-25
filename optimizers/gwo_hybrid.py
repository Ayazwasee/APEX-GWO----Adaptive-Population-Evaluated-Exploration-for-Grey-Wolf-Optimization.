import time
import numpy as np
from optimizers.common import (
    RunResult,
    is_better,
    pack_diversity,
    convergence_iteration,
    rank_normalize,
    default_pop_value,
)


def select_dual_leaders(fitness_vals, pop_vals, init_best_pop, mode, pop_weight):
    f_score = rank_normalize(fitness_vals, mode)
    p_score = rank_normalize(pop_vals, mode)
    if mode == "min":
        anchor_bonus = np.where(pop_vals <= init_best_pop + 1e-9, 0.08, 0.0)
    else:
        anchor_bonus = np.where(pop_vals >= init_best_pop - 1e-9, 0.08, 0.0)
    p_score = np.clip(p_score + anchor_bonus, 0.0, 1.0)
    composite = (1.0 - pop_weight) * f_score + pop_weight * p_score
    ranked = np.argsort(composite)[::-1]
    return ranked, f_score, p_score, composite


def run_hybrid_gwo(
    obj_func,
    lb,
    ub,
    mode="min",
    pop_value_func=None,
    pop_weight=0.25,
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
    pop_fn = pop_value_func or default_pop_value

    positions = lb + rng.random((n_agents, dim)) * (ub - lb)
    init_pop_vals = np.array([float(pop_fn(p)) for p in positions], dtype=float)
    init_best_pop = float(np.min(init_pop_vals) if mode == "min" else np.max(init_pop_vals))

    archive_pos = None
    archive_fit = None
    archive_pop = None
    elite_pos = positions[0].copy()
    elite_fit = np.inf if mode == "min" else -np.inf
    stagnation = 0
    prev_elite = elite_fit

    convergence_curve = []
    a_curve = []
    diversity_curve = []
    trajectories = []
    leaders = []
    initial_audit = []

    for t in range(max_iter):
        positions = np.clip(positions, lb, ub)
        fitness_vals = np.array([float(obj_func(p)) for p in positions], dtype=float)
        pop_vals = np.array([float(pop_fn(p)) for p in positions], dtype=float)

        for i in range(n_agents):
            if is_better(fitness_vals[i], elite_fit, mode):
                elite_fit = float(fitness_vals[i])
                elite_pos = positions[i].copy()

        progress = t / max(1, max_iter - 1)
        lambda_p = pop_weight * ((1.0 - progress) ** 1.4) + 0.04

        if t == 0:
            ranked0, f0, p0, c0 = select_dual_leaders(fitness_vals, pop_vals, init_best_pop, mode, lambda_p)
            roles = {int(ranked0[0]): "Alpha", int(ranked0[1]): "Beta", int(ranked0[2]): "Delta"}
            for rank_i, wi in enumerate(ranked0, start=1):
                wi = int(wi)
                initial_audit.append({
                    "rank": rank_i,
                    "wolf": f"W-{wi + 1:02d}",
                    "role": roles.get(wi, "Omega"),
                    "pop_value": round(float(pop_vals[wi]), 4),
                    "fitness": round(float(fitness_vals[wi]), 5),
                    "pop_score": round(float(p0[wi]), 4),
                    "fit_score": round(float(f0[wi]), 4),
                    "combined": round(float(c0[wi]), 4),
                })

        if archive_pos is not None:
            pool_pos = np.vstack([positions, archive_pos])
            pool_fit = np.concatenate([fitness_vals, archive_fit])
            pool_pop = np.concatenate([pop_vals, archive_pop])
        else:
            pool_pos, pool_fit, pool_pop = positions, fitness_vals, pop_vals

        ranked, _, _, composite = select_dual_leaders(pool_fit, pool_pop, init_best_pop, mode, lambda_p)
        chosen = []
        for idx in ranked:
            ci = int(idx)
            if all(np.linalg.norm(pool_pos[ci] - pool_pos[prev]) > 1e-7 for prev in chosen):
                chosen.append(ci)
            if len(chosen) == 3:
                break
        while len(chosen) < 3:
            chosen.append(int(ranked[len(chosen)]))

        alpha_pos = pool_pos[chosen[0]].copy()
        beta_pos = pool_pos[chosen[1]].copy()
        delta_pos = pool_pos[chosen[2]].copy()

        archive_pos = np.vstack([alpha_pos, beta_pos, delta_pos, elite_pos])
        archive_fit = np.array([pool_fit[chosen[0]], pool_fit[chosen[1]], pool_fit[chosen[2]], elite_fit], dtype=float)
        archive_pop = np.array([pool_pop[chosen[0]], pool_pop[chosen[1]], pool_pop[chosen[2]], float(pop_fn(elite_pos))], dtype=float)

        if abs(prev_elite - elite_fit) < 1e-7:
            stagnation += 1
        else:
            stagnation = 0
            prev_elite = elite_fit

        if t >= max_iter // 4:
            step = 0.045 * (1.0 - 0.80 * progress) * (ub - lb)
            candidate = np.clip(elite_pos + rng.normal(0.0, step, size=dim), lb, ub)
            cand_fit = float(obj_func(candidate))
            if is_better(cand_fit, elite_fit, mode):
                elite_fit = cand_fit
                elite_pos = candidate.copy()
                alpha_pos = elite_pos.copy()
                stagnation = 0

        a = float(2.0 * np.cos((np.pi / 2.0) * (progress ** 1.25)))
        if stagnation >= 4 and progress < 0.85:
            a = float(np.clip(a + 0.35 * (1.0 - progress), 0.0, 2.0))

        convergence_curve.append(float(elite_fit))
        a_curve.append(a)
        diversity_curve.append(pack_diversity(positions, lb, ub))
        if record_2d:
            trajectories.append(positions[:, :2].round(4).tolist())
            leaders.append([alpha_pos[:2].round(4).tolist(), beta_pos[:2].round(4).tolist(), delta_pos[:2].round(4).tolist()])

        top3 = np.array([composite[c] for c in chosen], dtype=float)
        weights = (top3 + 0.15) * np.array([0.52, 0.30, 0.18])
        weights = weights / np.sum(weights)

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

            new_pos = weights[0] * X1 + weights[1] * X2 + weights[2] * X3
            if stagnation >= 4 and i >= n_agents - 2:
                opp = lb + ub - positions[i]
                new_pos = 0.5 * opp + 0.5 * (lb + rng.random(dim) * (ub - lb))
            positions[i] = np.clip(new_pos, lb, ub)

    return RunResult(
        algo_id="hybrid_gwo",
        name="Combined GWO (Non-Linear a + Dual-Criteria)",
        mode=mode,
        best_position=elite_pos,
        best_fitness=float(elite_fit),
        convergence_curve=convergence_curve,
        a_curve=a_curve,
        diversity_curve=diversity_curve,
        trajectory_history=trajectories,
        leaders_history=leaders,
        initial_pop_audit=initial_audit,
        init_best_pop_value=init_best_pop,
        runtime_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        convergence_iter=convergence_iteration(convergence_curve, mode),
    )
