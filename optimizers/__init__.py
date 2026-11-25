from optimizers.gwo_original import run_original_gwo
from optimizers.gwo_nonlinear_a import run_nonlinear_a_gwo
from optimizers.gwo_dual_criteria import run_dual_criteria_gwo
from optimizers.gwo_hybrid import run_hybrid_gwo

ALGORITHMS = {
    "original_gwo": {
        "id": "original_gwo",
        "code": "GWO-01",
        "name": "Original GWO",
        "short": "Original",
        "summary": "Linear convergence factor a(t) = 2(1 - t/T) with fitness-only Alpha, Beta, and Delta selection.",
        "a_type": "Linear",
        "leader_type": "Fitness Only",
        "file": "optimizers/gwo_original.py",
        "runner": run_original_gwo,
    },
    "nonlinear_a_gwo": {
        "id": "nonlinear_a_gwo",
        "code": "GWO-02",
        "name": "Non-Linear a GWO",
        "short": "Non-Linear a",
        "summary": "Replaces linear a with non-linear cosine decay a(t) = 2 cos((pi/2)(t/T)^1.25). Keeps fitness-only leaders.",
        "a_type": "Non-Linear Cosine",
        "leader_type": "Fitness Only",
        "file": "optimizers/gwo_nonlinear_a.py",
        "runner": run_nonlinear_a_gwo,
    },
    "dual_criteria_gwo": {
        "id": "dual_criteria_gwo",
        "code": "GWO-03",
        "name": "Dual-Criteria GWO",
        "short": "Dual-Criteria",
        "summary": "Keeps linear a(t), but selects Alpha, Beta, and Delta using initial-population anchor + dual population-value & fitness ranking.",
        "a_type": "Linear",
        "leader_type": "Population + Fitness",
        "file": "optimizers/gwo_dual_criteria.py",
        "runner": run_dual_criteria_gwo,
    },
    "hybrid_gwo": {
        "id": "hybrid_gwo",
        "code": "GWO-04",
        "name": "Combined GWO",
        "short": "Combined (Both)",
        "summary": "Combines both the non-linear cosine convergence factor a(t) and the dual-criteria population + fitness leader selection.",
        "a_type": "Non-Linear Adaptive",
        "leader_type": "Population + Fitness",
        "file": "optimizers/gwo_hybrid.py",
        "runner": run_hybrid_gwo,
    },
}

ALGO_ORDER = ["original_gwo", "nonlinear_a_gwo", "dual_criteria_gwo", "hybrid_gwo"]
