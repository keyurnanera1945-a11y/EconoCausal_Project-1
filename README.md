# EconoCausal

Causal ML project estimating the individual treatment effect (ITE) of discounts/promotional
emails on customer conversion, using Double Machine Learning (DML), and optimally
allocating a limited discount budget to the customers it will actually persuade.

## Stack
- EconML (Double Machine Learning)
- DoWhy (causal DAG + refutation tests)
- SciPy (budget-constrained allocation)
- FastAPI (backend, coming Week 4)
- React + Plotly.js (dashboard, coming Week 3-4)

## Dataset
Hillstrom Email Marketing dataset (place as `data/hillstrom.csv`).
Randomized email campaign — customers randomly assigned to receive an email or not,
outcome = whether they converted (purchased).

## Run the pipeline
```bash
pip install -r requirements.txt
python src/pipeline.py
```

Outputs land in `outputs/`:
- `ite_scores.csv` — per-customer individual treatment effect estimates
- `qini_curve.csv` — data to plot the uplift/Qini curve
- `allocation_table.csv` — optimized discount allocation given a budget
- `run_summary.txt` — naive baseline, DoWhy estimate, refutation test results, allocation summary

## Project docs
See `PLAN.md` for the full week-by-week roadmap and `PROGRESS.md` for the session log.
