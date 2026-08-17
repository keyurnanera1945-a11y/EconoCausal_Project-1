"""
EconoCausal — Full Pipeline
Double Machine Learning for estimating causal effect of discounts/emails
on customer conversion, using the Hillstrom Email Marketing dataset.

Pipeline stages:
1. Load & clean data
2. Define Treatment / Outcome / Confounders
3. Naive (biased) baseline estimate
4. DoWhy causal DAG + identification
5. EconML Double Machine Learning -> Individual Treatment Effects (ITE)
6. DoWhy refutation tests (validity checks)
7. Uplift/Qini evaluation
8. SciPy budget-constrained optimization (discount allocation)
9. Save all outputs (model, ITE scores, allocation table, plots)
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "hillstrom.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------------
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardize column names in case of casing differences
    df.columns = [c.strip().lower() for c in df.columns]

    # Simplify treatment: any email vs no email
    df["treatment"] = (df["segment"] != "No E-Mail").astype(int)

    # One-hot encode categorical confounders
    df = pd.get_dummies(df, columns=["zip_code", "channel"], drop_first=True)

    return df


# ---------------------------------------------------------------------
# 2. DEFINE COLUMN ROLES
# ---------------------------------------------------------------------
OUTCOME_COL = "conversion"
TREATMENT_COL = "treatment"

BASE_CONFOUNDERS = [
    "recency", "history", "mens", "womens", "newbie",
]


def get_confounder_cols(df: pd.DataFrame) -> list:
    onehot_cols = [c for c in df.columns if c.startswith("zip_code_") or c.startswith("channel_")]
    return BASE_CONFOUNDERS + onehot_cols


# ---------------------------------------------------------------------
# 3. NAIVE (BIASED) BASELINE
# ---------------------------------------------------------------------
def naive_baseline(df: pd.DataFrame) -> dict:
    """Simple group-mean comparison — ignores confounding, shown for contrast."""
    grp = df.groupby(TREATMENT_COL)[OUTCOME_COL].mean()
    naive_effect = grp.get(1, np.nan) - grp.get(0, np.nan)
    return {
        "conversion_rate_treated": grp.get(1, np.nan),
        "conversion_rate_control": grp.get(0, np.nan),
        "naive_effect_estimate": naive_effect,
    }


# ---------------------------------------------------------------------
# 4. CAUSAL DAG (DoWhy)
# ---------------------------------------------------------------------
def build_causal_model(df: pd.DataFrame):
    from dowhy import CausalModel

    confounders = get_confounder_cols(df)

    model = CausalModel(
        data=df,
        treatment=TREATMENT_COL,
        outcome=OUTCOME_COL,
        common_causes=confounders,
    )
    return model


def identify_and_estimate_naive_dowhy(model):
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
    )
    return identified_estimand, estimate


# ---------------------------------------------------------------------
# 5. DOUBLE MACHINE LEARNING (EconML) -> ITE
# ---------------------------------------------------------------------
def train_dml_model(df: pd.DataFrame):
    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    confounders = get_confounder_cols(df)

    Y = df[OUTCOME_COL].values
    T = df[TREATMENT_COL].values
    X = df[confounders].values  # heterogeneity features (for CATE)
    W = df[confounders].values  # controls (for nuisance models)

    model_y = RandomForestRegressor(n_estimators=200, min_samples_leaf=20, random_state=RANDOM_STATE)
    model_t = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=RANDOM_STATE)

    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        random_state=RANDOM_STATE,
    )
    dml.fit(Y, T, X=X, W=W)

    ite = dml.effect(X)  # individual treatment effect per row
    df = df.copy()
    df["ite"] = ite

    return dml, df


# ---------------------------------------------------------------------
# 6. REFUTATION TESTS (DoWhy) — validity checks
# ---------------------------------------------------------------------
def run_refutation_tests(model, identified_estimand, estimate):
    results = {}

    try:
        refute_random = model.refute_estimate(
            identified_estimand, estimate, method_name="random_common_cause"
        )
        results["random_common_cause"] = str(refute_random)
    except Exception as e:
        results["random_common_cause"] = f"FAILED: {e}"

    try:
        refute_placebo = model.refute_estimate(
            identified_estimand, estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
        )
        results["placebo_treatment"] = str(refute_placebo)
    except Exception as e:
        results["placebo_treatment"] = f"FAILED: {e}"

    try:
        refute_subset = model.refute_estimate(
            identified_estimand, estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.8,
        )
        results["data_subset"] = str(refute_subset)
    except Exception as e:
        results["data_subset"] = f"FAILED: {e}"

    return results


# ---------------------------------------------------------------------
# 7. UPLIFT / QINI EVALUATION
# ---------------------------------------------------------------------
def compute_qini_curve(df: pd.DataFrame, ite_col: str = "ite"):
    """
    Returns dataframe with cumulative incremental gains sorted by
    predicted ITE, for plotting a Qini curve.
    """
    d = df.sort_values(ite_col, ascending=False).reset_index(drop=True)
    d["cum_treated"] = (d[TREATMENT_COL] == 1).cumsum()
    d["cum_control"] = (d[TREATMENT_COL] == 0).cumsum()
    d["cum_outcome_treated"] = ((d[TREATMENT_COL] == 1) * d[OUTCOME_COL]).cumsum()
    d["cum_outcome_control"] = ((d[TREATMENT_COL] == 0) * d[OUTCOME_COL]).cumsum()

    n_treated_total = max((d[TREATMENT_COL] == 1).sum(), 1)
    n_control_total = max((d[TREATMENT_COL] == 0).sum(), 1)

    d["qini"] = (
        d["cum_outcome_treated"]
        - d["cum_outcome_control"] * (n_treated_total / n_control_total)
    )
    return d[["qini"]].reset_index().rename(columns={"index": "rank"})


# ---------------------------------------------------------------------
# 8. BUDGET-CONSTRAINED OPTIMIZATION (SciPy)
# ---------------------------------------------------------------------
def optimize_allocation(df: pd.DataFrame, budget: float, cost_per_treatment: float = 10.0):
    """
    Greedy/LP-style allocation: rank customers by ITE per dollar spent,
    allocate discount budget to highest-uplift customers first until
    budget is exhausted. (Simple, explainable version of the SciPy
    linear program — can be swapped for scipy.optimize.linprog for a
    strict LP formulation later.)
    """
    d = df.copy()
    d["ite_per_dollar"] = d["ite"] / cost_per_treatment
    d = d.sort_values("ite_per_dollar", ascending=False).reset_index(drop=True)

    d["cum_cost"] = (d.index + 1) * cost_per_treatment
    d["allocated"] = (d["cum_cost"] <= budget).astype(int)

    n_allocated = d["allocated"].sum()
    total_spend = n_allocated * cost_per_treatment
    expected_gain = d.loc[d["allocated"] == 1, "ite"].sum()

    summary = {
        "budget": budget,
        "customers_targeted": int(n_allocated),
        "total_spend": total_spend,
        "expected_incremental_conversions": expected_gain,
    }
    return d, summary


# ---------------------------------------------------------------------
# 9. FULL PIPELINE RUNNER
# ---------------------------------------------------------------------
def run_full_pipeline(budget: float = 5000.0):
    print("[1/8] Loading data...")
    df = load_data()

    print("[2/8] Naive baseline...")
    naive = naive_baseline(df)
    print(naive)

    print("[3/8] Building DoWhy causal model...")
    model = build_causal_model(df)
    identified_estimand, dowhy_estimate = identify_and_estimate_naive_dowhy(model)
    print("DoWhy backdoor linear estimate:", dowhy_estimate.value)

    print("[4/8] Training EconML DML model (this may take a minute)...")
    dml_model, df_with_ite = train_dml_model(df)
    print("ITE summary stats:")
    print(df_with_ite["ite"].describe())

    print("[5/8] Running refutation tests...")
    refutations = run_refutation_tests(model, identified_estimand, dowhy_estimate)
    for k, v in refutations.items():
        print(f"--- {k} ---")
        print(v[:500] if isinstance(v, str) else v)

    print("[6/8] Computing Qini curve...")
    qini_df = compute_qini_curve(df_with_ite)
    qini_df.to_csv(OUTPUT_DIR / "qini_curve.csv", index=False)

    print("[7/8] Running budget optimization...")
    alloc_df, alloc_summary = optimize_allocation(df_with_ite, budget=budget)
    print(alloc_summary)

    print("[8/8] Saving outputs...")
    df_with_ite.to_csv(OUTPUT_DIR / "ite_scores.csv", index=False)
    alloc_df.to_csv(OUTPUT_DIR / "allocation_table.csv", index=False)

    with open(OUTPUT_DIR / "run_summary.txt", "w") as f:
        f.write("=== NAIVE BASELINE ===\n")
        f.write(str(naive) + "\n\n")
        f.write("=== DOWHY BACKDOOR ESTIMATE ===\n")
        f.write(str(dowhy_estimate.value) + "\n\n")
        f.write("=== REFUTATION TESTS ===\n")
        for k, v in refutations.items():
            f.write(f"--- {k} ---\n{v}\n\n")
        f.write("=== BUDGET OPTIMIZATION SUMMARY ===\n")
        f.write(str(alloc_summary) + "\n")

    print("Done. Outputs saved to:", OUTPUT_DIR)
    return {
        "naive": naive,
        "dowhy_estimate": dowhy_estimate.value,
        "refutations": refutations,
        "alloc_summary": alloc_summary,
    }


if __name__ == "__main__":
    run_full_pipeline(budget=5000.0)
