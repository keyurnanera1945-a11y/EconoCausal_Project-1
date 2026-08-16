import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load the Hillstrom marketing dataset.
    """
    df = pd.read_csv(path)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare Hillstrom data for causal analysis.
    """

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows with missing treatment/outcome
    df = df.dropna(subset=["segment", "conversion"])

    # Binary treatment:
    # 0 = No Email
    # 1 = Any Email
    df["treatment"] = (
        df["segment"] != "No E-Mail"
    ).astype(int)

    # Binary outcome
    df["outcome"] = df["conversion"].astype(int)

    return df


def get_feature_columns(df: pd.DataFrame):
    """
    Select customer characteristics used as confounders.
    """

    features = [
        "recency",
        "history",
        "mens",
        "womens",
        "newbie",
    ]

    # Keep only columns actually available
    features = [
        col for col in features
        if col in df.columns
    ]

    return features