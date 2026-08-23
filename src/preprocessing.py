import pandas as pd


# ============================================================
# DATASET CONFIGURATION
# ============================================================

REQUIRED_COLUMNS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
]

FEATURE_COLUMNS = [
    "recency",
    "history",
    "mens",
    "womens",
    "newbie",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Load the Hillstrom marketing dataset.

    Parameters
    ----------
    path : str
        Path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    if not path:
        raise ValueError("Dataset path cannot be empty.")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df


# ============================================================
# DATASET SCHEMA VALIDATION
# ============================================================

def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that the Hillstrom dataset contains
    all required columns.

    Raises
    ------
    ValueError
        If required columns are missing.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{missing_columns}"
        )


# ============================================================
# DATA QUALITY REPORT
# ============================================================

def get_data_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate a basic data quality report.

    Returns
    -------
    dict
        Dataset shape, duplicate count, missing values,
        and numeric-column information.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing_values = (
        df.isnull()
        .sum()
        .to_dict()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    numeric_columns = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": duplicate_rows,
        "missing_values": missing_values,
        "numeric_columns": numeric_columns,
    }


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare Hillstrom data for causal analysis.

    Processing steps:
        1. Validate dataset schema.
        2. Remove duplicate rows.
        3. Remove rows with missing treatment/outcome.
        4. Create binary treatment.
        5. Create binary outcome.
        6. Validate treatment and outcome values.

    Returns
    -------
    pd.DataFrame
        Cleaned and prepared dataframe.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    validate_schema(df)

    df = df.copy()

    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    df = df.drop_duplicates()

    # --------------------------------------------------------
    # Remove rows with missing treatment/outcome
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "segment",
            "conversion",
        ]
    )

    # --------------------------------------------------------
    # Binary treatment
    #
    # 0 = No Email
    # 1 = Any Email
    # --------------------------------------------------------

    df["treatment"] = (
        df["segment"] != "No E-Mail"
    ).astype(int)

    # --------------------------------------------------------
    # Binary outcome
    #
    # 0 = No conversion
    # 1 = Conversion
    # --------------------------------------------------------

    df["outcome"] = (
        df["conversion"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Validate treatment
    # --------------------------------------------------------

    valid_treatment_values = {
        0,
        1,
    }

    treatment_values = set(
        df["treatment"]
        .dropna()
        .unique()
    )

    if not treatment_values.issubset(
        valid_treatment_values
    ):
        raise ValueError(
            "Treatment column contains "
            f"invalid values: {treatment_values}"
        )

    # --------------------------------------------------------
    # Validate outcome
    # --------------------------------------------------------

    outcome_values = set(
        df["outcome"]
        .dropna()
        .unique()
    )

    if not outcome_values.issubset(
        valid_treatment_values
    ):
        raise ValueError(
            "Outcome column contains "
            f"invalid values: {outcome_values}"
        )

    return df


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list:
    """
    Select customer characteristics used as
    confounders in the causal model.

    Returns only columns that are available
    in the supplied dataframe.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    features = [
        column
        for column in FEATURE_COLUMNS
        if column in df.columns
    ]

    if not features:
        raise ValueError(
            "No causal feature columns were found "
            "in the dataframe."
        )

    return features


# ============================================================
# TREATMENT SUMMARY
# ============================================================

def get_treatment_summary(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate treatment-group statistics.

    Returns
    -------
    dict
        Counts and proportions for treated and
        untreated customers.
    """

    if "treatment" not in df.columns:
        raise ValueError(
            "Treatment column not found. "
            "Run prepare_data() first."
        )

    total = len(df)

    if total == 0:
        return {
            "total_customers": 0,
            "treated_customers": 0,
            "control_customers": 0,
            "treatment_rate": 0.0,
        }

    treated = int(
        (df["treatment"] == 1).sum()
    )

    control = int(
        (df["treatment"] == 0).sum()
    )

    return {
        "total_customers": total,
        "treated_customers": treated,
        "control_customers": control,
        "treatment_rate": treated / total,
    }


# ============================================================
# OUTCOME SUMMARY
# ============================================================

def get_outcome_summary(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate conversion statistics.
    """

    if "outcome" not in df.columns:
        raise ValueError(
            "Outcome column not found. "
            "Run prepare_data() first."
        )

    total = len(df)

    if total == 0:
        return {
            "total_customers": 0,
            "converted_customers": 0,
            "conversion_rate": 0.0,
        }

    converted = int(
        (df["outcome"] == 1).sum()
    )

    return {
        "total_customers": total,
        "converted_customers": converted,
        "conversion_rate": converted / total,
    }