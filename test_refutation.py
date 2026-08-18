import sys

sys.path.append(".")


from src.preprocessing import (
    load_data,
    prepare_data
)

from src.refutation import (
    run_all_refutation_tests
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/hillstrom.csv"


print("=" * 70)
print("ECONOCAUSAL - DO WHY REFUTATION TEST")
print("=" * 70)


# ============================================================
# LOAD ORIGINAL DATA
# ============================================================

print("\nLoading dataset...")

df = load_data(
    DATA_PATH
)

print(
    f"Original dataset shape: {df.shape}"
)


# ============================================================
# USE SAME PREPROCESSING AS WEEK 1 / WEEK 2
# ============================================================

print("\nApplying standard EconoCausal preprocessing...")

df_clean = prepare_data(
    df
)

print(
    f"Cleaned dataset shape: {df_clean.shape}"
)


# ============================================================
# VERIFY EXPECTED SIZE
# ============================================================

print(
    "\nChecking dataset consistency..."
)

print(
    f"Rows used by Week 1 / Week 2 / Refutation: "
    f"{len(df_clean):,}"
)


# ============================================================
# RUN REFUTATION
# ============================================================

print(
    "\nRunning DoWhy causal audit..."
)

results = run_all_refutation_tests(
    df_clean
)


# ============================================================
# ORIGINAL EFFECT
# ============================================================

print("\n")
print("=" * 60)
print("ORIGINAL CAUSAL EFFECT")
print("=" * 60)

print(
    f"Original causal effect: "
    f"{results['original_effect']:.8f}"
)


# ============================================================
# DATASET SIZE
# ============================================================

print("\n")
print("=" * 60)
print("DATASET CONSISTENCY CHECK")
print("=" * 60)

print(
    f"Dataset rows used: "
    f"{results['dataset_rows']:,}"
)

print(
    f"Dataset columns used: "
    f"{results['dataset_shape'][1]}"
)


# ============================================================
# REFUTATION RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("REFUTATION TEST RESULTS")
print("=" * 60)


for test_name, result in results[
    "results"
].items():

    print("\n")
    print(
        f"TEST: {test_name}"
    )

    print("-" * 50)

    if "error" in result:

        print(
            f"ERROR: {result['error']}"
        )

        continue

    print(
        "Estimated effect:",
        result["estimated_effect"]
    )

    print(
        "New effect:",
        result["new_effect"]
    )

    print(
        "P-value:",
        result["p_value"]
    )


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("CAUSAL AUDIT COMPLETED")
print("=" * 70)

print(
    "\nAll refutation tests used the SAME cleaned dataset "
    "as the main EconoCausal pipeline."
)