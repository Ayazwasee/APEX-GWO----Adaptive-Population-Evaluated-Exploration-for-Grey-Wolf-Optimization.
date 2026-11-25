import time
import numpy as np
from optimizers.common import RunResult, is_better, pack_diversity, convergence_iteration


def select_column_best_leaders(positions, fitness_vals, mode="min"):
    table = np.column_stack([positions, fitness_vals])
    n_wolves, n_cols = table.shape
    remaining = list(range(n_wolves))
    selected = []
    pass_records = []

    for _ in range(min(3, n_wolves)):
        sub = table[remaining]
        col_best = np.min(sub, axis=0) if mode == "min" else np.max(sub, axis=0)
        hits = np.isclose(sub, col_best[None, :], rtol=0.0, atol=1e-12)
        counts = np.sum(hits, axis=1)

        if len(pass_records) == 0:
            pass_records = [
                {
                    "index": remaining[k],
                    "boxes": int(counts[k]),
                    "total_cols": n_cols,
                    "won_cols": [
                        f"x{c + 1}" if c < n_cols - 1 else "f(x)"
                        for c in range(n_cols)
                        if hits[k, c]
                    ],
                }
                for k in range(len(remaining))
            ]

        best_local = max(
            range(len(remaining)),
            key=lambda k: (
                int(counts[k]),
                -fitness_vals[remaining[k]] if mode == "min" else fitness_vals[remaining[k]],
            ),
        )
        chosen_idx = remaining.pop(best_local)
        selected.append(chosen_idx)

    while len(selected) < 3:
        selected.append(selected[-1])

    return selected[0], selected[1], selected[2], pass_records


def run_dual_criteria_gwo(
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

    best_pos = positions[0].copy()
    best_fit = np.inf if mode == "min" else -np.inf

    convergence_curve = []
    a_curve = []
    diversity_curve = []
    trajectories = []
    leaders = []
    initial_audit = []

    for t in range(max_iter):
        positions = np.clip(positions, lb, ub)
        fitness_vals = np.array([float(obj_func(p)) for p in positions], dtype=float)

        for i in range(n_agents):
            if is_better(fitness_vals[i], best_fit, mode):
                best_fit = float(fitness_vals[i])
                best_pos = positions[i].copy()

        a_idx, b_idx, d_idx, pass0 = select_column_best_leaders(positions, fitness_vals, mode)
        alpha_pos = positions[a_idx].copy()
        beta_pos = positions[b_idx].copy()
        delta_pos = positions[d_idx].copy()

        if t == 0:
            roles = {a_idx: "Alpha", b_idx: "Beta", d_idx: "Delta"}
            ordered = sorted(
                pass0,
                key=lambda r: (
                    0 if r["index"] == a_idx else (1 if r["index"] == b_idx else (2 if r["index"] == d_idx else 3)),
                    -r["boxes"],
                    fitness_vals[r["index"]] if mode == "min" else -fitness_vals[r["index"]],
                ),
            )
            for rank_i, rec in enumerate(ordered, start=1):
                wi = rec["index"]
                initial_audit.append({
                    "rank": rank_i,
                    "wolf": f"W-{wi + 1:02d}",
                    "role": roles.get(wi, "Omega"),
                    "boxes": f"{rec['boxes']} / {rec['total_cols']}",
                    "won_cols": ", ".join(rec["won_cols"][:5]) + ("..." if len(rec["won_cols"]) > 5 else "") if rec["won_cols"] else "—",
                    "fitness": round(float(fitness_vals[wi]), 5),
                })

        progress = t / max(1, max_iter - 1)
        a = 2.0 * (1.0 - progress)

        convergence_curve.append(float(best_fit))
        a_curve.append(float(a))
        diversity_curve.append(pack_diversity(positions, lb, ub))
        if record_2d:
            trajectories.append(positions[:, :2].round(4).tolist())
            leaders.append([
                alpha_pos[:2].round(4).tolist(),
                beta_pos[:2].round(4).tolist(),
                delta_pos[:2].round(4).tolist(),
            ])

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
        algo_id="dual_criteria_gwo",
        name="Dual-Criteria GWO",
        mode=mode,
        best_position=best_pos,
        best_fitness=float(best_fit),
        convergence_curve=convergence_curve,
        a_curve=a_curve,
        diversity_curve=diversity_curve,
        trajectory_history=trajectories,
        leaders_history=leaders,
        initial_pop_audit=initial_audit,
        runtime_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        convergence_iter=convergence_iteration(convergence_curve, mode),
    )
