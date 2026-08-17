# EconoCausal — Dynamic Pricing via Double Machine Learning

## What This Project Is (one-liner)
Estimate the *causal* effect of discounts on individual customers (not just correlation),
then optimally allocate a limited discount budget to the customers who are actually
persuadable — using Double Machine Learning (DML) + optimization + a dashboard.

## Domain
Causal AI / Economics — a rare, high-differentiation portfolio piece (almost nobody
at student level does causal inference; most do standard predictive ML).

---

## Tech Stack
- **Causal Inference:** Python, EconML (Microsoft), DoWhy (Microsoft)
- **Base ML models for DML:** Random Forest / LightGBM
- **Optimization:** SciPy (linear/convex allocation under budget constraint)
- **Backend API:** FastAPI
- **Frontend:** React + Plotly.js (Uplift/Qini curves, allocation table)
- **Dataset:** Kaggle Criteo Uplift Modeling Dataset (primary) or Hillstrom Email
  Marketing dataset (lighter/simpler, good first pass)

---

## Dataset Options (ranked by ease)
1. **Hillstrom Email Marketing Dataset** — ~64k rows, 3 treatment groups
   (no email / men's email / women's email), small, fast to iterate on, classic
   for uplift modeling tutorials. BEST STARTING POINT.
2. **Criteo Uplift Modeling Dataset** — ~25M rows, real ad campaign data,
   gold standard, industry-grade. Use this once the pipeline works end-to-end
   on Hillstrom, to prove it scales.
3. EconML/CausalML also has built-in *synthetic data generators*
   (`econml.data.dgps`) — useful for controlled testing/debugging where you
   know the "true" treatment effect and can verify your model recovers it.

---

## Core Concepts You Need To Understand (in order)
1. **Confounders vs Treatment vs Outcome** — a confounder affects both whether
   someone got the discount AND whether they'd buy anyway (e.g. loyal customers
   are both more likely to get targeted promos and more likely to buy regardless).
2. **Correlation vs Causation** — why plain regression/XGBoost on "got discount"
   as a feature gives you a *biased* effect estimate.
3. **Propensity Score** — probability a customer receives treatment given their
   features; used to correct for non-random assignment in observational data.
4. **Double/Debiased Machine Learning (DML)** — trains two ML models
   (one predicting outcome from features, one predicting treatment from features),
   removes their predictable parts, then estimates the treatment effect from
   what's left (the "residuals"). This is what cancels out confounding bias.
5. **Individual Treatment Effect (ITE) / CATE** — the DML output: per-customer
   estimate of "how much did/would the treatment change their outcome."
6. **Qini / Uplift Curves** — how you evaluate an uplift model (different from
   normal ML accuracy metrics like AUC).
7. **Refutation Tests (DoWhy)** — sanity checks that your causal estimate isn't
   just noise or a modeling artifact (e.g. add a random confounder and see if
   the effect estimate stays stable).

---

## Week-wise Development Plan

### Week 1 — Causal Foundations + Scaffolding
**Causal ML side:**
- Load Hillstrom dataset, explore treatment/outcome columns
- Define the causal DAG in DoWhy (identify Confounders, Treatment, Outcome)
- Understand and verify the DAG assumptions

**Frontend side:**
- Init React app
- Build upload view for campaign data
- Build budget-constraint input view

/ MILESTONE: You can explain your DAG out loud — which vars are confounders and why

### Week 2 — DML Model Training
**Causal ML side:**
- Train EconML's `LinearDML` / `CausalForestDML` using Random Forest or
  LightGBM as base estimators
- Estimate ITE per customer
- Sanity-check: does ITE distribution make sense (not all zero, not all identical)?

**Frontend side:**
- Integrate Plotly.js
- Build Qini curve + Uplift curve rendering (compare model vs random targeting)

/ MILESTONE: You have a working ITE estimate per customer + a chart proving it beats random

### Mid-Project Review — Causal Audit
- Run DoWhy Refutation Tests (placebo treatment, random common cause,
  data subset validation) to prove the estimate is robust, not noise
- Frontend: validate dashboard can filter/chart thousands of ITE scores smoothly

/ MILESTONE: You can defend your model's validity with actual refutation test results
  (this is the part that will impress reviewers/interviewers most — most student
  projects skip validation entirely)

### Week 3 — Prescriptive Optimization
**Causal ML side:**
- Write SciPy optimization: maximize total predicted revenue/retention subject
  to total discount budget constraint (e.g. $5,000)
- Output: a per-customer "prescription" — who gets what discount

**Frontend side:**
- Build Allocation Matrix table UI showing final prescription per customer

/ MILESTONE: End-to-end — raw data in, optimal discount table out

### Week 4 — API, Packaging, Polish
**Causal ML side:**
- Wrap in FastAPI REST endpoints
- Build a simple data-drift detector (warns if new data looks statistically
  different from training data — customer behavior changed)

**Frontend side:**
- Natural language summaries (e.g. "This strategy saves $4,200 vs blanket targeting")
- Final polish pass

/ MILESTONE: Fully working demo-able product with API + dashboard + narrative

---

## Scale-Up Step (after Week 4, optional but recommended for portfolio strength)
Swap Hillstrom → Criteo dataset to prove it works on large-scale real ad data,
not just a toy dataset. This is what separates a "tutorial project" from
"production-grade skill demonstration."

---

## Current Status
- [ ] Week 1 not started
- Next session: start with dataset download + EDA + DAG definition

## Notes for Prem
- You prefer understanding before writing code — each week above starts with
  the CONCEPT before the CODE. Ask for Socratic walkthroughs on DML/DAG before
  touching EconML syntax.
- Consolidate code into minimal clean cells per session, not fragmented steps.
