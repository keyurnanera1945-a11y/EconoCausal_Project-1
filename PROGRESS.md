# EconoCausal Project Progress

## Week 1: Data Preparation & Exploratory Data Analysis
- Loaded and cleaned Hillstrom Email Marketing dataset (64,000 customers).
- Defined treatment (`segment != 'No E-Mail'`), outcome (`conversion`), and customer features.
- Set up project structure and baseline data pipeline in `src/pipeline.py`.

## Week 2: Double Machine Learning & Uplift Modeling
- Implemented EconML `LinearDML` model for Individual Treatment Effect (ITE) estimation.
- Generated real ITE scores saved to `outputs/ite_scores.csv`.
- Validated model using DoWhy refutation tests (Placebo, Random Common Cause, Data Subset).
- Computed Qini curve and plot saved to `outputs/qini_curve.csv` and `outputs/qini_plot.png`.
- Built initial React frontend with Qini curve visualization page and baseline allocation table.

## Week 3 Complete: SciPy LP Optimization, Discount Tiers, & Live Matrix UI
- **SciPy LP Formal Optimizer (`optimize_allocation_lp`):**
  - Formulated linear programming solver using `scipy.optimize.linprog` with the HiGHS solver to maximize total expected ITE across all customers subject to budget constraint $c_{\text{treatment}} \cdot \sum x_i \le \text{budget}$.
  - Negated ITE values in the objective vector to perform maximization.
  - Side-by-side comparison verified that LP exact solution matches greedy baseline for homogeneous costs ($5,000 budget $\rightarrow$ 500 customers targeted, $5,000 spend, +5.8459 expected conversion gain).

- **Dynamic Discount Tier Pricing (`optimize_allocation_tiered`):**
  - Extended optimization to support 3 dynamic discount tiers:
    - **Low ($5):** $1.0\times$ ITE multiplier (base uplift)
    - **Medium ($10):** $1.3\times$ ITE multiplier (moderate incentive boost)
    - **High ($20):** $1.6\times$ ITE multiplier (high incentive boost)
  - Modeled LP over $N \times K$ variables with constraint $\sum_t x_{i,t} \le 1$ per customer.
  - Output saved to `outputs/allocation_table_tiered.csv` and `frontend/public/data/allocation_table_tiered.csv`.
  - Tiered optimization summary at $5,000 budget: 537 customers targeted, $4,995 total spend, **+8.5996 expected conversion gain** (+47.1% gain over flat rate), breaking down into 105 Low ($5) and 432 Medium ($10) discount assignments.

- **Full Allocation Matrix UI & Live Budget Slider:**
  - Extended React frontend page (`AllocationPage.jsx`) to load `allocation_table_tiered.csv`.
  - Paginated table at **25 rows per page** with Next/Previous controls and page counters.
  - Required columns displayed: Rank, Customer ITE Score ($\tau_i$), Assigned Discount Tier ($5 Low / $10 Mid / $20 High), Expected Incremental Conversion, Past Spend ($), and Segment.
  - Clickable column headers supporting ascending & descending sorting with indicator icons (`▲`, `▼`, `↕`).
  - Interactive budget slider ($1,000 – $20,000) with real-time client-side re-filtering and live updating KPI cards.
