from dataclasses import dataclass
from typing import Any, Callable, Dict
import numpy as np


@dataclass
class ProblemSpec:
    id: str
    name: str
    category: str
    mode: str
    dim: int
    lb: np.ndarray
    ub: np.ndarray
    obj_func: Callable[[np.ndarray], float]
    pop_value_func: Callable[[np.ndarray], float]
    description: str
    unit: str
    decode_func: Callable[[np.ndarray], Dict[str, Any]]
    optimum_2d: float = 0.0


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    return float(10.0 * len(x) + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    n = len(x)
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
        - np.exp(np.sum(np.cos(2.0 * np.pi * x)) / n)
        + 20.0
        + np.e
    )


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def _make_knapsack(seed=2026) -> ProblemSpec:
    rng = np.random.default_rng(seed)
    n_items = 20
    weights = rng.integers(4, 28, size=n_items).astype(float)
    values = (weights * rng.uniform(1.8, 5.5, size=n_items) + rng.integers(5, 35, size=n_items)).round(1)
    capacity = float(np.round(0.45 * np.sum(weights), 1))
    densities = values / weights

    def decode_bits(x):
        return (1.0 / (1.0 + np.exp(-x)) >= 0.5).astype(int)

    def obj(x):
        bits = decode_bits(x)
        tw = float(np.sum(bits * weights))
        tv = float(np.sum(bits * values))
        if tw > capacity:
            over = tw - capacity
            return tv - 18.0 * over - 0.8 * (over ** 2)
        return tv

    def pop_val(x):
        probs = 1.0 / (1.0 + np.exp(-x))
        return float(np.sum(probs * densities) / np.sum(densities))

    def decode(x):
        bits = decode_bits(x)
        return {
            "type": "knapsack",
            "selected_count": int(np.sum(bits)),
            "total_weight": round(float(np.sum(bits * weights)), 1),
            "capacity": capacity,
            "total_value": round(float(np.sum(bits * values)), 1),
            "feasible": bool(np.sum(bits * weights) <= capacity),
            "items": [
                {"id": i + 1, "weight": float(weights[i]), "value": float(values[i]), "picked": bool(bits[i])}
                for i in range(n_items)
            ],
        }

    return ProblemSpec(
        id="knapsack",
        name="0/1 Knapsack (20 Items)",
        category="Combinatorial",
        mode="max",
        dim=n_items,
        lb=np.full(n_items, -4.0),
        ub=np.full(n_items, 4.0),
        obj_func=obj,
        pop_value_func=pop_val,
        description=f"Maximize total packed value across {n_items} items without exceeding {capacity} kg capacity.",
        unit="Value",
        decode_func=decode,
    )


def _make_tsp(seed=2026) -> ProblemSpec:
    rng = np.random.default_rng(seed + 1)
    n_cities = 14
    angles = np.linspace(0, 2 * np.pi, n_cities, endpoint=False)
    radii = rng.uniform(28.0, 46.0, size=n_cities)
    coords = np.column_stack([
        50.0 + radii * np.cos(angles) + rng.normal(0, 5.2, size=n_cities),
        50.0 + radii * np.sin(angles) + rng.normal(0, 5.2, size=n_cities),
    ])
    dist_mat = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))

    def obj(x):
        tour = np.argsort(x)
        return float(np.sum(dist_mat[tour[:-1], tour[1:]])) + float(dist_mat[tour[-1], tour[0]])

    def pop_val(x):
        tour = np.argsort(x)
        jumps = np.abs(np.diff(np.append(tour, tour[0])))
        return float(np.mean(np.minimum(jumps, n_cities - jumps)))

    def decode(x):
        tour = [int(c) for c in np.argsort(x)]
        return {
            "type": "tsp",
            "tour": tour,
            "tour_length": round(obj(x), 2),
            "coords": coords.round(2).tolist(),
        }

    return ProblemSpec(
        id="tsp",
        name="Traveling Salesman — TSP (14 Cities)",
        category="Routing",
        mode="min",
        dim=n_cities,
        lb=np.zeros(n_cities),
        ub=np.ones(n_cities),
        obj_func=obj,
        pop_value_func=pop_val,
        description=f"Minimize closed delivery tour distance across {n_cities} cities using Random-Key permutation encoding.",
        unit="km",
        decode_func=decode,
    )


def _make_network_flow() -> ProblemSpec:
    edges = [
        (0, 1, 22.0), (0, 2, 18.0), (0, 3, 14.0),
        (1, 2, 8.0), (1, 4, 16.0),
        (2, 4, 12.0), (2, 5, 15.0),
        (3, 2, 6.0), (3, 5, 13.0),
        (4, 6, 24.0), (5, 6, 22.0),
    ]
    caps = np.array([e[2] for e in edges], dtype=float)

    def obj(x):
        flows = np.clip(x, 0.0, caps)
        inflow, outflow = np.zeros(7), np.zeros(7)
        for i, (u, v, _) in enumerate(edges):
            outflow[u] += flows[i]
            inflow[v] += flows[i]
        imbalance = float(np.sum(np.abs(inflow[1:6] - outflow[1:6])))
        return float(inflow[6] - 3.5 * imbalance)

    def pop_val(x):
        return float(np.mean(x / caps))

    def decode(x):
        flows = np.clip(x, 0.0, caps)
        inflow, outflow = np.zeros(7), np.zeros(7)
        for i, (u, v, _) in enumerate(edges):
            outflow[u] += flows[i]
            inflow[v] += flows[i]
        return {
            "type": "kv",
            "Delivered Flow (Gbps)": round(float(inflow[6]), 2),
            "Conservation Error": round(float(np.sum(np.abs(inflow[1:6] - outflow[1:6]))), 2),
            "Active Links": int(np.sum(flows > 0.5)),
        }

    return ProblemSpec(
        id="network_flow",
        name="Max Network Flow (7 Nodes, 11 Links)",
        category="Graph Flow",
        mode="max",
        dim=len(edges),
        lb=np.zeros(len(edges)),
        ub=caps.copy(),
        obj_func=obj,
        pop_value_func=pop_val,
        description="Maximize end-to-end network throughput subject to link capacity and flow conservation constraints.",
        unit="Gbps",
        decode_func=decode,
    )


def _make_cloud_scheduling(seed=2026) -> ProblemSpec:
    rng = np.random.default_rng(seed + 2)
    n_tasks, n_vms = 20, 5
    task_mi = rng.uniform(250.0, 1600.0, size=n_tasks)
    vm_mips = np.array([180.0, 240.0, 320.0, 410.0, 520.0])
    vm_power = np.array([0.85, 1.0, 1.25, 1.55, 1.95])

    def obj(x):
        assign = np.clip(np.floor(x).astype(int), 0, n_vms - 1)
        vm_time = np.zeros(n_vms)
        for ti, vi in enumerate(assign):
            vm_time[vi] += task_mi[ti] / vm_mips[vi]
        return float(np.max(vm_time) + np.sum(vm_time * vm_power) * 0.08)

    def pop_val(x):
        assign = np.clip(np.floor(x).astype(int), 0, n_vms - 1)
        return float(np.std(np.bincount(assign, minlength=n_vms)))

    def decode(x):
        assign = np.clip(np.floor(x).astype(int), 0, n_vms - 1)
        vm_time = np.zeros(n_vms)
        for ti, vi in enumerate(assign):
            vm_time[vi] += task_mi[ti] / vm_mips[vi]
        return {
            "type": "kv",
            "Makespan (s)": round(float(np.max(vm_time)), 2),
            "Energy Penalty": round(float(np.sum(vm_time * vm_power) * 0.08), 2),
            "VM Loads (s)": ", ".join(str(round(float(v), 1)) for v in vm_time),
        }

    return ProblemSpec(
        id="cloud_scheduling",
        name="Cloud VM Task Scheduling (20 Tasks)",
        category="Scheduling",
        mode="min",
        dim=n_tasks,
        lb=np.zeros(n_tasks),
        ub=np.full(n_tasks, float(n_vms) - 1e-6),
        obj_func=obj,
        pop_value_func=pop_val,
        description="Minimize makespan and power cost assigning 20 compute tasks across 5 heterogeneous cloud VMs.",
        unit="Cost",
        decode_func=decode,
    )


def _make_bin_packing(seed=2026) -> ProblemSpec:
    rng = np.random.default_rng(seed + 3)
    n_items, bin_cap = 18, 100.0
    sizes = rng.integers(18, 66, size=n_items).astype(float)

    def pack(order):
        bins = []
        for idx in order:
            s = sizes[idx]
            placed = False
            for bi in range(len(bins)):
                if bins[bi] + s <= bin_cap:
                    bins[bi] += s
                    placed = True
                    break
            if not placed:
                bins.append(s)
        return bins

    def obj(x):
        bins = pack(np.argsort(x))
        slack = float(np.mean([(bin_cap - b) / bin_cap for b in bins]))
        return float(len(bins)) + 0.35 * slack

    def pop_val(x):
        ordered = sizes[np.argsort(x)]
        return float(np.sum(ordered[:-1] < ordered[1:]) / n_items)

    def decode(x):
        bins = pack(np.argsort(x))
        return {
            "type": "kv",
            "Containers Used": len(bins),
            "Avg Fill (%)": round(100.0 * float(np.mean(bins)) / bin_cap, 1),
            "Bin Loads": ", ".join(str(int(b)) for b in bins),
        }

    return ProblemSpec(
        id="bin_packing",
        name="Warehouse Bin Packing (18 Packages)",
        category="Packing",
        mode="min",
        dim=n_items,
        lb=np.zeros(n_items),
        ub=np.ones(n_items),
        obj_func=obj,
        pop_value_func=pop_val,
        description="Minimize containers used and unused space via Random-Key First-Fit bin packing.",
        unit="Bins",
        decode_func=decode,
    )


def _make_portfolio(seed=2026) -> ProblemSpec:
    rng = np.random.default_rng(seed + 4)
    assets = ["NVDA", "MSFT", "TCS", "INFY", "BTC-ETF", "GOLD", "US-T10Y", "RELIANCE"]
    n_assets = len(assets)
    ret = np.array([0.24, 0.16, 0.14, 0.12, 0.31, 0.08, 0.05, 0.15])
    A = rng.normal(0.0, 0.11, size=(n_assets, n_assets))
    cov = A @ A.T + np.diag([0.09, 0.04, 0.035, 0.03, 0.22, 0.018, 0.005, 0.045])
    rf = 0.04

    def obj(x):
        w = np.maximum(x, 0.0)
        w = w / (np.sum(w) + 1e-9)
        pr = float(np.dot(w, ret))
        pv = float(np.sqrt(max(1e-9, w @ cov @ w)))
        conc = 2.0 * float(np.sum(np.maximum(0.0, w - 0.45) ** 2))
        return (pr - rf) / pv - conc

    def pop_val(x):
        w = np.maximum(x, 0.0)
        w = w / (np.sum(w) + 1e-9)
        return float(np.dot(w, ret))

    def decode(x):
        w = np.maximum(x, 0.0)
        w = w / (np.sum(w) + 1e-9)
        pr = float(np.dot(w, ret))
        pv = float(np.sqrt(max(1e-9, w @ cov @ w)))
        return {
            "type": "kv",
            "Sharpe Ratio": round((pr - rf) / pv, 4),
            "Expected Return (%)": round(100.0 * pr, 2),
            "Volatility (%)": round(100.0 * pv, 2),
            "Top Allocation": f"{assets[int(np.argmax(w))]} ({round(100.0 * float(np.max(w)), 1)}%)",
        }

    return ProblemSpec(
        id="portfolio",
        name="Portfolio Sharpe Optimization (8 Assets)",
        category="Finance",
        mode="max",
        dim=n_assets,
        lb=np.zeros(n_assets),
        ub=np.ones(n_assets),
        obj_func=obj,
        pop_value_func=pop_val,
        description="Maximize risk-adjusted Markowitz Sharpe ratio across 8 assets under a 45% single-asset concentration cap.",
        unit="Sharpe",
        decode_func=decode,
    )


def _make_facility_location(seed=2026) -> ProblemSpec:
    rng = np.random.default_rng(seed + 5)
    n_zones, n_hubs, radius = 30, 3, 24.0
    centers = np.array([[22.0, 28.0], [74.0, 25.0], [52.0, 76.0]])
    zones = np.clip(np.vstack([centers[i] + rng.normal(0, 11.5, size=(10, 2)) for i in range(3)]), 2.0, 98.0)
    demands = rng.integers(15, 85, size=n_zones).astype(float)

    def obj(x):
        hubs = x.reshape(n_hubs, 2)
        min_d = np.min(np.sqrt(np.sum((zones[:, None, :] - hubs[None, :, :]) ** 2, axis=-1)), axis=1)
        return float(np.sum(demands[min_d <= radius]) + 0.15 * np.sum(demands * np.exp(-min_d / radius)))

    def pop_val(x):
        h = x.reshape(n_hubs, 2)
        return float((np.linalg.norm(h[0] - h[1]) + np.linalg.norm(h[0] - h[2]) + np.linalg.norm(h[1] - h[2])) / 3.0)

    def decode(x):
        hubs = x.reshape(n_hubs, 2)
        min_d = np.min(np.sqrt(np.sum((zones[:, None, :] - hubs[None, :, :]) ** 2, axis=-1)), axis=1)
        cov = 100.0 * float(np.sum(demands[min_d <= radius]) / np.sum(demands))
        return {
            "type": "facility",
            "covered_pct": round(cov, 1),
            "hubs": hubs.round(2).tolist(),
            "zones": zones.round(2).tolist(),
            "demands": demands.tolist(),
            "radius": radius,
        }

    return ProblemSpec(
        id="facility_location",
        name="Facility Location Coverage (3 Hubs)",
        category="Logistics",
        mode="max",
        dim=6,
        lb=np.zeros(6),
        ub=np.full(6, 100.0),
        obj_func=obj,
        pop_value_func=pop_val,
        description="Place 3 urban logistics hubs to maximize customer demand covered within a 24 km radius.",
        unit="Score",
        decode_func=decode,
    )


def _make_max_cut(seed=2026) -> ProblemSpec:
    rng = np.random.default_rng(seed + 6)
    n_nodes = 16
    W = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < 0.42:
                W[i, j] = W[j, i] = float(rng.integers(2, 15))

    def obj(x):
        s = np.where(x >= 0.0, 1.0, -1.0)
        return 0.25 * float(np.sum(W * (1.0 - np.outer(s, s))))

    def pop_val(x):
        return 1.0 - abs(float(np.mean(x >= 0.0)) - 0.5) * 2.0

    def decode(x):
        s = (x >= 0.0).astype(int)
        return {
            "type": "kv",
            "Max Cut Weight": round(obj(x), 1),
            "Partition A Size": int(np.sum(s == 1)),
            "Partition B Size": int(np.sum(s == 0)),
        }

    return ProblemSpec(
        id="max_cut",
        name="Weighted Graph Max-Cut (16 Nodes)",
        category="Graph Partition",
        mode="max",
        dim=n_nodes,
        lb=np.full(n_nodes, -2.0),
        ub=np.full(n_nodes, 2.0),
        obj_func=obj,
        pop_value_func=pop_val,
        description="Partition a 16-node weighted graph into two subsets maximizing total crossing edge weight.",
        unit="Weight",
        decode_func=decode,
    )


def _make_hybrid_ir(seed=2026) -> ProblemSpec:
    rng = np.random.default_rng(seed + 7)
    n_queries, docs_per_q = 22, 22
    max_lat = 40.0
    queries = []
    for _ in range(n_queries):
        rel = rng.choice([0, 1, 2, 3], size=docs_per_q, p=[0.48, 0.24, 0.18, 0.10]).astype(float)
        rel[0], rel[1] = 3.0, 2.0
        dl = rng.lognormal(0.0, 0.45, size=docs_per_q)
        idf = rng.uniform(1.2, 4.5, size=docs_per_q)
        tf = np.maximum(0.1, (rel + 0.6) * rng.uniform(0.7, 1.8, size=docs_per_q) * (dl ** 0.55))
        sem = np.clip(0.22 + 0.23 * rel + rng.normal(0.0, 0.14, size=docs_per_q), 0.0, 1.0)
        trust = np.clip(0.30 + 0.20 * rel + rng.normal(0.0, 0.16, size=docs_per_q), 0.0, 1.0)
        spam = np.where(rel == 0)[0][:3]
        tf[spam] = rng.uniform(9.0, 18.0, size=len(spam))
        trust[spam] = rng.uniform(0.05, 0.22, size=len(spam))
        queries.append((tf, dl, idf, sem, trust, rel))

    discounts = np.log2(np.arange(2, 12))

    def eval_ir(x):
        k1, b, wb, ws, wt = x
        wsum = wb + ws + wt + 1e-9
        nwb, nws, nwt = wb / wsum, ws / wsum, wt / wsum
        ndcgs = []
        for tf, dl, idf, sem, trust, rel in queries:
            bm25 = idf * ((tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl)))
            bm25_n = (bm25 - np.min(bm25)) / (np.max(bm25) - np.min(bm25) + 1e-9)
            order = np.argsort(nwb * bm25_n + nws * sem + nwt * trust)[::-1]
            rk = rel[order][:10]
            ik = np.sort(rel)[::-1][:10]
            dcg = float(np.sum((2.0 ** rk - 1.0) / discounts))
            idcg = float(np.sum((2.0 ** ik - 1.0) / discounts))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        mean_ndcg = float(np.mean(ndcgs))
        lat = float(12.0 + 8.5 * nwb + 34.0 * (nws ** 0.85) + 18.0 * (nwt ** 0.9) + 1.8 * (k1 / 3.0))
        viol = max(0.0, (lat - max_lat) / max_lat)
        score = mean_ndcg - 0.12 * (lat / 65.0) - 4.0 * (viol ** 2)
        return score, mean_ndcg, lat, (nwb, nws, nwt)

    def obj(x):
        return eval_ir(x)[0]

    def pop_val(x):
        return float((x[3] + x[4]) / (x[2] + x[3] + x[4] + 1e-9))

    def decode(x):
        score, ndcg, lat, (nwb, nws, nwt) = eval_ir(x)
        return {
            "type": "kv",
            "NDCG@10": round(ndcg, 4),
            "Latency (ms)": round(lat, 2),
            "BM25 k1 / b": f"{x[0]:.3f} / {x[1]:.3f}",
            "Weights (BM25/Sem/Trust)": f"{nwb:.2f} / {nws:.2f} / {nwt:.2f}",
        }

    return ProblemSpec(
        id="hybrid_ir",
        name="Hybrid IR Search Rank Tuner (5D)",
        category="Information Retrieval",
        mode="max",
        dim=5,
        lb=np.array([0.2, 0.05, 0.0, 0.0, 0.0]),
        ub=np.array([3.0, 1.0, 1.0, 1.0, 1.0]),
        obj_func=obj,
        pop_value_func=pop_val,
        description="Tune BM25 (k1, b) and hybrid semantic/trust weights maximizing NDCG@10 under a 40 ms latency budget.",
        unit="Score",
        decode_func=decode,
    )


def _make_continuous(pid, name, fn, bound, dim=10, opt_val=0.0) -> ProblemSpec:
    return ProblemSpec(
        id=pid,
        name=f"{name} ({dim}D)",
        category="Continuous",
        mode="min",
        dim=dim,
        lb=np.full(dim, -bound),
        ub=np.full(dim, bound),
        obj_func=fn,
        pop_value_func=lambda x, ov=opt_val: float(np.sqrt(np.mean((x - ov) ** 2))),
        description=f"Minimize {name} benchmark landscape in {dim} dimensions (global minimum = 0.0).",
        unit="Loss",
        decode_func=lambda x: {"type": "kv", "Loss": round(float(fn(x)), 6), "L2 Norm": round(float(np.linalg.norm(x)), 4)},
        optimum_2d=opt_val,
    )


def build_problem_suite(seed=2026) -> Dict[str, ProblemSpec]:
    items = [
        _make_knapsack(seed),
        _make_tsp(seed),
        _make_network_flow(),
        _make_cloud_scheduling(seed),
        _make_bin_packing(seed),
        _make_portfolio(seed),
        _make_facility_location(seed),
        _make_max_cut(seed),
        _make_hybrid_ir(seed),
        _make_continuous("rastrigin_10d", "Rastrigin Multimodal", rastrigin, 5.12, 10, 0.0),
        _make_continuous("ackley_10d", "Ackley Deep Funnel", ackley, 10.0, 10, 0.0),
        _make_continuous("rosenbrock_10d", "Rosenbrock Valley", rosenbrock, 5.0, 10, 1.0),
    ]
    return {p.id: p for p in items}


def build_2d_benchmarks() -> Dict[str, ProblemSpec]:
    items = [
        _make_continuous("rastrigin_2d", "Rastrigin", rastrigin, 5.12, 2, 0.0),
        _make_continuous("ackley_2d", "Ackley", ackley, 8.0, 2, 0.0),
        _make_continuous("rosenbrock_2d", "Rosenbrock", rosenbrock, 3.0, 2, 1.0),
        _make_continuous("sphere_2d", "Sphere", sphere, 6.0, 2, 0.0),
    ]
    return {p.id: p for p in items}
