# 📈 EconoCausal: Double Machine Learning & Prescriptive Uplift Engine

> **Causal Machine Learning platform for estimating Individual Treatment Effects (ITE) and solving budget-constrained promotional discount allocation.** Built with **EconML**, **DoWhy**, **FastAPI**, and **React + Plotly**.

---

## 🎯 Problem Overview

Traditional propensity models predict $P(Y=1 \mid X)$, identifying customers who are most likely to convert. However, this often wastes budget on **"Sure Things"** (customers who would buy anyway without a discount) and neglects **"Sleeping Dogs"** (customers whom marketing alienates).

**EconoCausal** solves this by estimating the true **Individual Treatment Effect (ITE / Uplift)**:
$$\tau_i = \mathbb{E}[Y_i(1) - Y_i(0) \mid X_i]$$
We then optimize a constrained marketing budget by prioritizing **Persuadables**—customers whose purchase behavior is positively caused by the promotion.

---

## 🏗️ System Architecture

```
                                [ Hillstrom Dataset (64k Customers) ]
                                                  │
                ┌─────────────────────────────────┴─────────────────────────────────┐
                ▼                                                                   ▼
       [ 1. DoWhy Causal DAG ]                                           [ 2. Statistical Drift Detector ]
   - Backdoor Identification                                          - 2-Sample Kolmogorov-Smirnov (KS)
   - 9 Confounder Adjustments                                         - Population Stability Index (PSI)
                │                                                                   │
                ▼                                                                   ▼
     [ 3. EconML Double ML ]                                             [ Live Alerting Matrix ]
   - Cross-Fitting Residualization                                       (Economic Downturn / Web Surge)
   - Heterogeneous ITE ($\tau_i$) Estimation
                │
                ▼
  [ 4. Refutation Audit (Falsification) ]
   - Random Common Cause ($p = 0.86$)
   - Placebo Treatment ($p = 0.92$)
   - Data Subset Validation ($p = 0.84$)
                │
                ▼
  [ 5. SciPy Prescriptive Optimizer ]
   - Knapsack/Greedy ITE-per-dollar allocation
   - Outputs optimal customer prescriptions
                │
        ┌───────┴───────────────────────┐
        ▼                               ▼
[ FastAPI REST API (:8008) ]   [ React 19 + Plotly Dashboard (:5173) ]
- `/api/summary`               - Interactive Qini / Uplift Curve
- `/api/qini`                  - Dynamic Budget Simulation Matrix
- `/api/optimize`              - Real-Time Drift Quality Monitor
- `/api/drift/simulate`        - Causal DAG & Audit Diagnostics
```

---

## 📊 Empirical Results & Causal Audit

Evaluated on the 64,000-observation Hillstrom Email Marketing trial:

| Metric / Test | Value | Interpretation |
| :--- | :--- | :--- |
| **Naive Baseline Effect** | `+0.00495` (+0.495%) | Raw difference in conversion rates between groups |
| **DoWhy Backdoor Estimate** | `+0.00494` (+0.494%) | Unconfounded causal lift after conditioning on 9 confounders |
| **EconML DML Mean ITE ($\overline{\tau}$)** | `+0.00511` (+0.511%) | Mean of heterogeneous treatment effects (+3.18% relative difference) |
| **Random Common Cause Test** | $p = 0.86$ (**PASS**) | Estimate unchanged after injecting random noise confounder |
| **Placebo Treatment Test** | $p = 0.92$ (**PASS**) | Estimated effect drops to $\approx 0$ when treatment is randomized |
| **Data Subset Validation** | $p = 0.84$ (**PASS**) | Estimate remains stable across sub-samples |
| **Budget Optimization ($5,000)** | **7.28** net conversions | Targets top 500 customers at **~2.8x efficiency** vs. blanket campaigns |

---

## 📁 Repository Structure

```
EconoCausal_Project-1/
├── data/
│   └── hillstrom.csv              # Hillstrom marketing dataset (64k rows)
├── src/
│   ├── pipeline.py                # End-to-end DML training, audit & allocation pipeline
│   ├── drift.py                   # 2-Sample KS & PSI statistical data drift engine
│   ├── api.py                     # FastAPI REST server with CORS & dynamic simulation
│   └── plot_qini.py               # Lightweight Matplotlib visualizer
├── outputs/
│   ├── ite_scores.csv             # Per-customer ITE scores (64k rows)
│   ├── qini_curve.csv             # Cumulative uplift curve coordinates
│   ├── allocation_table.csv       # Prescribed discount dispatch matrix
│   ├── causal_dag.png             # Visual DAG exported via DoWhy
│   └── qini_plot.png              # Qini curve plot
├── frontend/                      # Vite + React 19 + Plotly.js Dashboard
│   ├── src/
│   │   ├── pages/
│   │   │   ├── QiniCurvePage.jsx      # Interactive Qini uplift curve & cutoff slider
│   │   │   ├── AllocationPage.jsx     # Prescriptive discount budget simulator
│   │   │   └── DriftMonitorPage.jsx   # Live KS + PSI data drift auditing
│   │   ├── App.jsx                # Main navigation & model audit views
│   │   └── index.css              # Modern glassmorphic dark design system
│   └── package.json
└── requirements.txt               # Python causal ML dependencies
```

---

## 🚀 Quick Start Guide

### 1. Python ML Pipeline & FastAPI Backend

```bash
# 1. Activate environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the full causal pipeline end-to-end
python src/pipeline.py

# 3. Launch FastAPI backend
python -m uvicorn src.api:app --host 127.0.0.1 --port 8008 --reload
```
API docs available at: `http://127.0.0.1:8008/docs`

---

### 2. Frontend React Dashboard

```bash
cd frontend
npm install
npm run dev
```
Open **`http://127.0.0.1:5173/`** in your browser to interact with the dashboard.

---

## 🔌 REST API Endpoints

- `GET /api/health` — API health and model status.
- `GET /api/summary` — Full audit metrics, backdoor estimate, and refutation results.
- `GET /api/qini` — Sampled Qini curve coordinates for high-performance chart rendering.
- `POST /api/optimize` — Dynamic Knapsack optimization taking `{ "budget": 5000, "cost_per_discount": 10 }`.
- `GET /api/drift/simulate?shift_type=economic_downturn` — Live statistical drift evaluation (KS test $p$-values & PSI).
- `POST /api/drift/audit` — Evaluates custom incoming customer JSON batches.

---

## 📜 License
MIT License. Built for rigorous causal data science and portfolio demonstrations.
