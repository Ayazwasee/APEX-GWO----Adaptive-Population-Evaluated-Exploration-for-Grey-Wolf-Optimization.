# Grey Wolf Optimization Lab (Vercel + FastAPI)

A clean, modular web application comparing **4 Grey Wolf Optimization (GWO) variants** across **12 real-world & combinatorial optimization problems** (6 Maximization + 6 Minimization), deployed on **Vercel** with a lightweight FastAPI + NumPy backend and vanilla HTML/CSS/Canvas frontend.

## The 4 GWO Algorithms
1. **Original GWO** (`optimizers/gwo_original.py`) — Standard linear convergence factor `a(t) = 2(1 - t/T)` and fitness-only Alpha/Beta/Delta selection.
2. **Non-Linear a GWO** (`optimizers/gwo_nonlinear_a.py`) — Replaces linear `a` with non-linear cosine decay `a(t) = 2 cos((pi/2)(t/T)^1.25)` while keeping fitness-only leaders.
3. **Dual-Criteria GWO** (`optimizers/gwo_dual_criteria.py`) — Keeps linear `a(t)`, but inspects the initial population (Min/Max aware) and selects Alpha/Beta/Delta using combined Population Value + Fitness ranking.
4. **Combined GWO** (`optimizers/gwo_hybrid.py`) — Combines both the non-linear adaptive convergence factor `a(t)` and the dual-criteria Population + Fitness leader selection.

## Local Development
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Open `http://localhost:8000`.

## Deploy to Vercel
1. Push this folder to a GitHub repository.
2. Import the repository in the Vercel Dashboard (Vercel auto-detects `main.py` as FastAPI and serves `public/` via CDN).
3. Or deploy via CLI:
```bash
vercel --prod
```
