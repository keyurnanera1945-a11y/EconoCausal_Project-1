from src.preprocessing import (
    load_data,
    prepare_data,
    get_feature_columns
)

DATA_PATH = "data/hillstrom.csv"

df = load_data(DATA_PATH)

print("Original shape:")
print(df.shape)

df_clean = prepare_data(df)

print("\nCleaned shape:")
print(df_clean.shape)

print("\nTreatment distribution:")
print(df_clean["treatment"].value_counts())

print("\nOutcome distribution:")
print(df_clean["outcome"].value_counts())

features = get_feature_columns(df_clean)

print("\nConfounders:")
print(features)

print("\nPrepared data:")
print(
    df_clean[
        features + ["treatment", "outcome"]
    ].head()
)