import inspect
import os
import sys
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="APEX-GWO Lab", version="2.4.0")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from optimizers import ALGORITHMS, ALGO_ORDER
from environments.problem_suite import build_problem_suite, build_2d_benchmarks


class RunRequest(BaseModel):
    problem_id: str = "knapsack"
    n_agents: int = Field(default=20, ge=6, le=50)
    max_iter: int = Field(default=40, ge=10, le=120)
    seed: int = Field(default=42, ge=0, le=99999)


class SuiteRequest(BaseModel):
    n_agents: int = Field(default=20, ge=6, le=40)
    max_iter: int = Field(default=40, ge=10, le=80)
    seed: int = Field(default=42, ge=0, le=99999)


class Arena2DRequest(BaseModel):
    benchmark_id: str = "rastrigin_2d"
    n_agents: int = Field(default=18, ge=6, le=40)
    max_iter: int = Field(default=35, ge=10, le=70)
    seed: int = Field(default=42, ge=0, le=99999)


def evaluate_outcome(scores_by_algo, mode, tol=1e-4):
    rounded = {aid: round(float(val), 4) for aid, val in scores_by_algo.items()}
    best_val = min(rounded.values()) if mode == "min" else max(rounded.values())
    top_ids = [aid for aid in ALGO_ORDER if abs(rounded[aid] - best_val) <= tol]

    if len(top_ids) > 1:
        return {
            "is_tie": True,
            "winner_id": None,
            "winner_name": "Tie",
            "tied_ids": top_ids,
            "tied_names": [ALGORITHMS[aid]["name"] for aid in top_ids],
        }
    wid = top_ids[0]
    return {
        "is_tie": False,
        "winner_id": wid,
        "winner_name": ALGORITHMS[wid]["name"],
        "tied_ids": [],
        "tied_names": [],
    }


@app.get("/", response_class=HTMLResponse)
def serve_index():
    for candidate in (
        os.path.join(ROOT_DIR, "index.html"),
        os.path.join(ROOT_DIR, "public", "index.html"),
    ):
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), status_code=200)
    raise HTTPException(status_code=404, detail="index.html not found at repository root or public/")


@app.get("/styles.css")
def serve_css():
    for candidate in (
        os.path.join(ROOT_DIR, "styles.css"),
        os.path.join(ROOT_DIR, "public", "styles.css"),
    ):
        if os.path.isfile(candidate):
            return FileResponse(candidate, media_type="text/css")
    raise HTTPException(status_code=404, detail="styles.css not found")


@app.get("/app.js")
def serve_js():
    for candidate in (
        os.path.join(ROOT_DIR, "app.js"),
        os.path.join(ROOT_DIR, "public", "app.js"),
    ):
        if os.path.isfile(candidate):
            return FileResponse(candidate, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "algorithms": len(ALGO_ORDER),
        "index_found": any(
            os.path.isfile(p)
            for p in (
                os.path.join(ROOT_DIR, "index.html"),
                os.path.join(ROOT_DIR, "public", "index.html"),
            )
        ),
    }


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
    if os.path.isfile(path):
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
            n_agents=req.n_agents,
            max_iter=req.max_iter,
            seed=req.seed,
            record_2d=False,
        )

    scores = {aid: results[aid].best_fitness for aid in ALGO_ORDER}
    outcome = evaluate_outcome(scores, ps.mode)
    orig_fit = results["original_gwo"].best_fitness

    payload_algos = []
    for aid in ALGO_ORDER:
        r = results[aid]
        denom = abs(orig_fit) + 1e-9
        delta_pct = (
            ((orig_fit - r.best_fitness) / denom) * 100.0
            if ps.mode == "min"
            else ((r.best_fitness - orig_fit) / denom) * 100.0
        )
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
            "initial_pop_audit": r.initial_pop_audit[:10],
            "decoded": ps.decode_func(r.best_position),
            "is_winner": (not outcome["is_tie"]) and (aid == outcome["winner_id"]),
            "is_tied": outcome["is_tie"] and (aid in outcome["tied_ids"]),
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
        "is_tie": outcome["is_tie"],
        "winner_id": outcome["winner_id"],
        "winner_name": outcome["winner_name"],
        "tied_ids": outcome["tied_ids"],
        "tied_names": outcome["tied_names"],
        "results": payload_algos,
    }


@app.post("/api/suite")
def run_all_twelve(req: SuiteRequest):
    suite = build_problem_suite(seed=req.seed)
    rows = []
    wins = {aid: 0 for aid in ALGO_ORDER}
    ties_count = 0
    deltas = {aid: [] for aid in ALGO_ORDER}

    for pid, ps in suite.items():
        scores = {}
        for aid in ALGO_ORDER:
            res = ALGORITHMS[aid]["runner"](
                obj_func=ps.obj_func,
                lb=ps.lb,
                ub=ps.ub,
                mode=ps.mode,
                n_agents=req.n_agents,
                max_iter=req.max_iter,
                seed=req.seed,
                record_2d=False,
            )
            scores[aid] = res.best_fitness

        outcome = evaluate_outcome(scores, ps.mode)
        if outcome["is_tie"]:
            ties_count += 1
        else:
            wins[outcome["winner_id"]] += 1

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
            "is_tie": outcome["is_tie"],
            "winner_id": outcome["winner_id"],
            "winner_name": outcome["winner_name"],
            "tied_ids": outcome["tied_ids"],
            "tied_names": outcome["tied_names"],
        })

    avg_deltas = {aid: round(float(np.mean(deltas[aid])), 2) for aid in ALGO_ORDER}
    return {
        "total_problems": len(rows),
        "wins": wins,
        "ties_count": ties_count,
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
