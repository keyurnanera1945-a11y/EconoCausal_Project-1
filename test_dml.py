from src.preprocessing import (
    load_data,
    prepare_data,
    get_feature_columns
)

from src.dml_model import (
    train_dml_model,
    create_ite_table,
    get_ite_summary
)

from src.uplift import (
    calculate_uplift_curve,
    calculate_qini_curve,
    calculate_qini_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/hillstrom.csv"


print("=" * 70)
print("ECONOCAUSAL - WEEK 2 DML TEST")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\n1. Loading dataset...")

df = load_data(
    DATA_PATH
)

print(
    f"Original shape: {df.shape}"
)


# ============================================================
# PREPROCESSING
# ============================================================

print("\n2. Preprocessing...")

df_clean = prepare_data(
    df
)

print(
    f"Cleaned shape: {df_clean.shape}"
)


# ============================================================
# FEATURES
# ============================================================

features = get_feature_columns(
    df_clean
)

print("\n3. Features:")

for feature in features:
    print(
        f"   - {feature}"
    )


# ============================================================
# TRAIN DML
# ============================================================

print("\n4. Training EconML CausalForestDML...")

(
    model,
    X_train,
    X_test,
    T_train,
    T_test,
    Y_train,
    Y_test
) = train_dml_model(
    df_clean,
    features=features
)

print(
    "\nDML training completed successfully!"
)

print(
    f"Training samples: {len(X_train):,}"
)

print(
    f"Testing samples: {len(X_test):,}"
)


# ============================================================
# CREATE ITE TABLE
# ============================================================

print("\n5. Calculating Individual Treatment Effects...")

ite_df = create_ite_table(
    df_clean,
    model,
    features=features
)

print(
    "ITE calculation completed!"
)


# ============================================================
# SUMMARY
# ============================================================

summary = get_ite_summary(
    ite_df
)

print("\n" + "=" * 70)
print("ITE SUMMARY")
print("=" * 70)

print(
    f"Customers:              {summary['customers']:,}"
)

print(
    f"Mean ITE:               {summary['mean_ite']:.6f}"
)

print(
    f"Median ITE:             {summary['median_ite']:.6f}"
)

print(
    f"Minimum ITE:            {summary['min_ite']:.6f}"
)

print(
    f"Maximum ITE:            {summary['max_ite']:.6f}"
)

print(
    f"Positive ITE:            {summary['positive_ite']:,}"
)

print(
    f"Negative ITE:            {summary['negative_ite']:,}"
)

print(
    f"Persuadables:            {summary['persuadables']:,}"
)

print(
    f"Strong Persuadables:     {summary['strong_persuadables']:,}"
)


# ============================================================
# TOP CUSTOMERS
# ============================================================

print("\n6. Top 10 customers by ITE:")

top_customers = (
    ite_df
    .sort_values(
        "ITE",
        ascending=False
    )
    .head(10)
)

print(
    top_customers[
        [
            "ITE",
            "ITE_Percentage",
            "customer_type"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# UPLIFT
# ============================================================

print("\n7. Calculating uplift curve...")

uplift_df = calculate_uplift_curve(
    ite_df
)

print(
    "Uplift curve calculated!"
)


# ============================================================
# QINI
# ============================================================

print("\n8. Calculating Qini curve...")

qini_df = calculate_qini_curve(
    ite_df
)

qini_score = calculate_qini_score(
    qini_df
)

print(
    f"Qini score: {qini_score:.4f}"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("WEEK 2 DML TEST COMPLETED SUCCESSFULLY")
print("=" * 70)