import optimizers.gwo_original as mod_original
import optimizers.gwo_nonlinear_a as mod_nonlinear_a
import optimizers.gwo_dual_criteria as mod_dual_criteria
import optimizers.gwo_hybrid as mod_hybrid

ALGORITHMS = {
    "original_gwo": {
        "id": "original_gwo",
        "code": "GWO-01",
        "name": "Original GWO",
        "short": "Original",
        "summary": "Linear convergence factor a(t) = 2(1 - t/T) with standard fitness-only Alpha, Beta, and Delta selection.",
        "a_type": "Linear",
        "leader_type": "Fitness Only",
        "file": "optimizers/gwo_original.py",
        "module": mod_original,
        "runner": mod_original.run_original_gwo,
    },
    "nonlinear_a_gwo": {
        "id": "nonlinear_a_gwo",
        "code": "GWO-02",
        "name": "Non-Linear a GWO",
        "short": "Non-Linear a",
        "summary": "Replaces linear a with non-linear cosine decay a(t) = 2 cos((pi/2)(t/T)^1.25) while keeping fitness-only leader selection.",
        "a_type": "Non-Linear Cosine",
        "leader_type": "Fitness Only",
        "file": "optimizers/gwo_nonlinear_a.py",
        "module": mod_nonlinear_a,
        "runner": mod_nonlinear_a.run_nonlinear_a_gwo,
    },
    "dual_criteria_gwo": {
        "id": "dual_criteria_gwo",
        "code": "GWO-03",
        "name": "Dual-Criteria GWO",
        "short": "Top-3 Tier Leaders",
        "summary": "Keeps linear a(t), and at Step 1 ranks Top 3 values (1st/2nd/3rd) across every column [x1..xd, f(x)] to assign Alpha, Beta, and Delta by majority vote.",
        "a_type": "Linear",
        "leader_type": "Top-3 Column Tiers (t=0)",
        "file": "optimizers/gwo_dual_criteria.py",
        "module": mod_dual_criteria,
        "runner": mod_dual_criteria.run_dual_criteria_gwo,
    },
    "hybrid_gwo": {
        "id": "hybrid_gwo",
        "code": "GWO-04",
        "name": "Combined GWO",
        "short": "Combined (Both)",
        "summary": "Combines the non-linear cosine convergence factor a(t) with Step-1 Top-3 tiered column-ranking leader selection for Alpha, Beta, and Delta.",
        "a_type": "Non-Linear Cosine",
        "leader_type": "Top-3 Column Tiers (t=0)",
        "file": "optimizers/gwo_hybrid.py",
        "module": mod_hybrid,
        "runner": mod_hybrid.run_hybrid_gwo,
    },
}

ALGO_ORDER = ["original_gwo", "nonlinear_a_gwo", "dual_criteria_gwo", "hybrid_gwo"]
