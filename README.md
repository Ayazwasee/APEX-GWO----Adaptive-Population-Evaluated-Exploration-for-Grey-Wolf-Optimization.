gwo_vercel_app/
├── main.py                         # FastAPI entrypoint auto-detected by Vercel
├── vercel.json                     # Vercel Python function config (maxDuration: 60s)
├── requirements.txt                # Lightweight deps only: fastapi, uvicorn, numpy
├── .gitignore
├── README.md
├── optimizers/
│   ├── __init__.py                 # 4-algorithm registry
│   ├── common.py                   # Shared data structures & rank normalization
│   ├── gwo_original.py             # Algorithm 1 (clean code, 0 comments)
│   ├── gwo_nonlinear_a.py          # Algorithm 2 (clean code, 0 comments)
│   ├── gwo_dual_criteria.py        # Algorithm 3 (clean code, 0 comments)
│   └── gwo_hybrid.py               # Algorithm 4 (clean code, 0 comments)
├── environments/
│   ├── __init__.py
│   └── problem_suite.py            # 12 Min/Max problems + 4 2D surfaces (pure NumPy)
└── public/
    ├── index.html                  # Clean editorial UI (Light default + Dark toggle)
    ├── styles.css                  # Minimal monochrome hairline layout, zero AI fluff
    └── app.js                      # High-DPI Canvas charts, 2D swarm player & Show Code modal
