import inspect
import os
import sys
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from optimizers import ALGORITHMS, ALGO_ORDER
from environments.problem_suite import build_problem_suite, build_2d_benchmarks

PUBLIC_DIR = os.path.join(ROOT_DIR, "public")

app = FastAPI(title="GWO Metaheuristics Lab", version="2.1.0")


class RunRequest(BaseModel):
    problem_id: str = "knapsack"
    n_agents: int = Field(default=20, ge=6, le=50)
    max_iter: int = Field(default=40, ge=10, le=120)
    pop_weight: float = Field(default=0.25, ge=0.05, le=0.60)
    seed: int = Field(default=42, ge=0, le=99999)


class SuiteRequest(BaseModel):
    n_agents: int = Field(default=18, ge=6, le=40)
    max_iter: int = Field(default=35, ge=10, le=80)
    pop_weight: float = Field(default=0.25, ge=0.05, le=0.60)
    seed: int = Field(default=42, ge=0, le=99999)


class Arena2DRequest(BaseModel):
    benchmark_id: str = "rastrigin_2d"
    n_agents: int = Field(default=18, ge=6, le=40)
    max_iter: int = Field(default=35, ge=10, le=70)
    pop_weight: float = Field(default=0.25, ge=0.05, le=0.60)
    seed: int = Field(default=42, ge=0, le=99999)


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(PUBLIC_DIR, "index.html"))


@app.get("/styles.css")
def serve_css():
    return FileResponse(os.path.join(PUBLIC_DIR, "styles.css"), media_type="text/css")


@app.get("/app.js")
def serve_js():
    return FileResponse(os.path.join(PUBLIC_DIR, "app.js"), media_type="application/javascript")


@app.get("/api/health")
def health():
    return {"status": "ok", "algorithms": len(ALGO_ORDER)}


@app.get("/api/meta")
def get_meta():
    suite = build_problem_suite(seed=2026)
    b2d = build_2d_benchmarks()
    return {
        "algorithms": [
            {
                "id": ALGORITHMS[k]["id"],
                "code": ALGORITHMS[k]["code"],
                "name": ALGORITHMS[k]["name"],
                "short": ALGORITHMS[k]["short"],
                "summary": ALGORITHMS[k]["summary"],
                "a_type": ALGORITHMS[k]["a_type"],
                "leader_type": ALGORITHMS[k]["leader_type"],
            }
            for k in ALGO_ORDER
        ],
        "problems": [
            {
                "id": p.id,
                "name": p.name,
                "category": p.category,
                "mode": p.mode,
                "dim": p.dim,
                "unit": p.unit,
                "description": p.description,
            }
            for p in suite.values()
        ],
        "benchmarks_2d": [
            {"id": p.id, "name": p.name, "mode": p.mode, "description": p.description}
            for p in b2d.values()
        ],
    }


@app.get("/api/code/{algo_id}")
def get_algorithm_code(algo_id: str):
    if algo_id not in ALGORITHMS:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    meta = ALGORITHMS[algo_id]
    path = os.path.join(ROOT_DIR, meta["file"])
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
    elif "module" in meta:
        source = inspect.getsource(meta["module"])
    else:
        source = inspect.getsource(meta["runner"])
    return {
        "id": meta["id"],
        "code": meta["code"],
        "name": meta["name"],
        "filename": meta["file"],
        "source": source,
    }


@app.post("/api/run")
def run_single_problem(req: RunRequest):
    suite = build_problem_suite(seed=req.seed)
    if req.problem_id not in suite:
        raise HTTPException(status_code=404, detail="Problem not found")
    ps = suite[req.problem_id]

    results = {}
    for aid in ALGO_ORDER:
        results[aid] = ALGORITHMS[aid]["runner"](
            obj_func=ps.obj_func,
            lb=ps.lb,
            ub=ps.ub,
            mode=ps.mode,
            pop_value_func=getattr(ps, "pop_value_func", None),
            pop_weight=req.pop_weight,
            n_agents=req.n_agents,
            max_iter=req.max_iter,
            seed=req.seed,
            record_2d=False,
        )

    orig_fit = results["original_gwo"].best_fitness
    winner_id = (
        min(ALGO_ORDER, key=lambda k: results[k].best_fitness)
        if ps.mode == "min"
        else max(ALGO_ORDER, key=lambda k: results[k].best_fitness)
    )

    payload_algos = []
    for aid in ALGO_ORDER:
        r = results[aid]
        denom = abs(orig_fit) + 1e-9
        delta_pct = (
            ((orig_fit - r.best_fitness) / denom) * 100.0
            if ps.mode == "min"
            else ((r.best_fitness - orig_fit) / denom) * 100.0
        )
        init_anchor = getattr(r, "init_best_pop_value", None)
        payload_algos.append({
            "id": aid,
            "code": ALGORITHMS[aid]["code"],
            "name": ALGORITHMS[aid]["name"],
            "short": ALGORITHMS[aid]["short"],
            "a_type": ALGORITHMS[aid]["a_type"],
            "leader_type": ALGORITHMS[aid]["leader_type"],
            "best_fitness": round(r.best_fitness, 5),
            "delta_vs_original_pct": round(delta_pct, 2),
            "convergence_iter": r.convergence_iter + 1,
            "runtime_ms": r.runtime_ms,
            "convergence_curve": [round(v, 5) for v in r.convergence_curve],
            "a_curve": [round(v, 4) for v in r.a_curve],
            "diversity_curve": [round(v, 4) for v in r.diversity_curve],
            "init_best_pop_value": round(init_anchor, 4) if init_anchor is not None else None,
            "initial_pop_audit": r.initial_pop_audit[:10],
            "decoded": ps.decode_func(r.best_position),
            "is_winner": aid == winner_id,
        })

    return {
        "problem": {
            "id": ps.id,
            "name": ps.name,
            "category": ps.category,
            "mode": ps.mode,
            "dim": ps.dim,
            "unit": ps.unit,
            "description": ps.description,
        },
        "winner_id": winner_id,
        "results": payload_algos,
    }


@app.post("/api/suite")
def run_all_twelve(req: SuiteRequest):
    suite = build_problem_suite(seed=req.seed)
    rows = []
    wins = {aid: 0 for aid in ALGO_ORDER}
    deltas = {aid: [] for aid in ALGO_ORDER}

    for pid, ps in suite.items():
        scores = {}
        for aid in ALGO_ORDER:
            res = ALGORITHMS[aid]["runner"](
                obj_func=ps.obj_func,
                lb=ps.lb,
                ub=ps.ub,
                mode=ps.mode,
                pop_value_func=getattr(ps, "pop_value_func", None),
                pop_weight=req.pop_weight,
                n_agents=req.n_agents,
                max_iter=req.max_iter,
                seed=req.seed,
                record_2d=False,
            )
            scores[aid] = res.best_fitness

        winner_id = (
            min(ALGO_ORDER, key=lambda k: scores[k])
            if ps.mode == "min"
            else max(ALGO_ORDER, key=lambda k: scores[k])
        )
        wins[winner_id] += 1

        orig = scores["original_gwo"]
        denom = abs(orig) + 1e-9
        row_deltas = {}
        for aid in ALGO_ORDER:
            d = ((orig - scores[aid]) / denom * 100.0) if ps.mode == "min" else ((scores[aid] - orig) / denom * 100.0)
            row_deltas[aid] = round(d, 2)
            deltas[aid].append(d)

        rows.append({
            "problem_id": pid,
            "problem": ps.name,
            "category": ps.category,
            "mode": ps.mode,
            "dim": ps.dim,
            "scores": {aid: round(scores[aid], 4) for aid in ALGO_ORDER},
            "deltas": row_deltas,
            "winner_id": winner_id,
            "winner_name": ALGORITHMS[winner_id]["name"],
        })

    avg_deltas = {aid: round(float(np.mean(deltas[aid])), 2) for aid in ALGO_ORDER}
    return {
        "total_problems": len(rows),
        "wins": wins,
        "avg_delta_vs_original": avg_deltas,
        "rows": rows,
    }


@app.post("/api/arena2d")
def run_arena_2d(req: Arena2DRequest):
    b2d = build_2d_benchmarks()
    if req.benchmark_id not in b2d:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    ps = b2d[req.benchmark_id]

    grid_n = 40
    gx = np.linspace(ps.lb[0], ps.ub[0], grid_n)
    gy = np.linspace(ps.lb[1], ps.ub[1], grid_n)
    zz = np.zeros((grid_n, grid_n), dtype=float)
    for r in range(grid_n):
        for c in range(grid_n):
            zz[r, c] = ps.obj_func(np.array([gx[c], gy[r]]))

    runs = []
    for aid in ALGO_ORDER:
        res = ALGORITHMS[aid]["runner"](
            obj_func=ps.obj_func,
            lb=ps.lb,
            ub=ps.ub,
            mode=ps.mode,
            pop_value_func=getattr(ps, "pop_value_func", None),
            pop_weight=req.pop_weight,
            n_agents=req.n_agents,
            max_iter=req.max_iter,
            seed=req.seed,
            record_2d=True,
        )
        runs.append({
            "id": aid,
            "code": ALGORITHMS[aid]["code"],
            "name": ALGORITHMS[aid]["name"],
            "best_fitness": round(res.best_fitness, 6),
            "convergence_curve": [round(v, 5) for v in res.convergence_curve],
            "a_curve": [round(v, 4) for v in res.a_curve],
            "trajectories": res.trajectory_history,
            "leaders": res.leaders_history,
        })

    return {
        "benchmark": {
            "id": ps.id,
            "name": ps.name,
            "lb": float(ps.lb[0]),
            "ub": float(ps.ub[0]),
            "optimum": [ps.optimum_2d, ps.optimum_2d],
            "z_min": float(np.min(zz)),
            "z_max": float(np.max(zz)),
            "grid": np.round(zz, 3).tolist(),
        },
        "max_iter": req.max_iter,
        "runs": runs,
    }
