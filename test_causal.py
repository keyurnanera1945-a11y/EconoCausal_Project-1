from src.preprocessing import (
    load_data,
    prepare_data,
    get_feature_columns
)

from src.causal_model import (
    build_causal_model,
    identify_effect,
    estimate_causal_effect
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = "data/hillstrom.csv"


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("=" * 60)
print("ECONOCAUSAL - WEEK 1 CAUSAL ANALYSIS")
print("=" * 60)

print("\nLoading dataset...")

df = load_data(DATA_PATH)

print("Original dataset shape:")
print(df.shape)


# --------------------------------------------------
# Prepare Dataset
# --------------------------------------------------

print("\nPreparing dataset...")

df_clean = prepare_data(df)

print("Cleaned dataset shape:")
print(df_clean.shape)


# --------------------------------------------------
# Select Confounders
# --------------------------------------------------

features = get_feature_columns(df_clean)

print("\nConfounders:")
for feature in features:
    print("-", feature)


# --------------------------------------------------
# Treatment and Outcome
# --------------------------------------------------

print("\nTreatment:")
print("0 = No E-Mail")
print("1 = E-Mail")

print("\nOutcome:")
print("0 = No Conversion")
print("1 = Conversion")


# --------------------------------------------------
# Build Causal Model
# --------------------------------------------------

print("\nBuilding DoWhy causal model...")

model = build_causal_model(
    df=df_clean,
    treatment="treatment",
    outcome="outcome",
    common_causes=features
)

print("Causal model created successfully.")


# --------------------------------------------------
# Identify Causal Effect
# --------------------------------------------------

print("\nIdentifying causal effect...")

identified_estimand = identify_effect(model)

print("\nIdentified Estimand:")
print(identified_estimand)


# --------------------------------------------------
# Estimate Causal Effect
# --------------------------------------------------

print("\nEstimating causal effect...")

estimate = estimate_causal_effect(
    model,
    identified_estimand
)


# --------------------------------------------------
# Display Result
# --------------------------------------------------

print("\n" + "=" * 60)
print("CAUSAL EFFECT RESULT")
print("=" * 60)

print("\nEstimated causal effect:")

print(estimate.value)

print("\nInterpretation:")

effect = estimate.value

if effect > 0:
    print(
        f"Email treatment is estimated to increase "
        f"conversion by approximately {effect:.4f}."
    )

elif effect < 0:
    print(
        f"Email treatment is estimated to decrease "
        f"conversion by approximately {abs(effect):.4f}."
    )

else:
    print(
        "The estimated causal effect is approximately zero."
    )


print("\nWeek 1 causal analysis completed.")