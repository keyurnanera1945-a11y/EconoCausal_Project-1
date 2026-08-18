import pandas as pd
from dowhy import CausalModel


# ============================================================
# REFUTATION CONFIGURATION
# ============================================================

TREATMENT = "treatment"
OUTCOME = "outcome"

CONFOUNDERS = [
    "recency",
    "history",
    "mens",
    "womens",
    "newbie"
]


# ============================================================
# VALIDATE CLEANED DATA
# ============================================================

def validate_refutation_data(df):
    """
    Validate that the refutation module receives the SAME
    cleaned dataframe used by Week 1 and Week 2.
    """

    required_columns = [
        "recency",
        "history",
        "mens",
        "womens",
        "newbie",
        "treatment",
        "outcome"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    if df.empty:

        raise ValueError(
            "The cleaned dataset is empty."
        )

    return True


# ============================================================
# BUILD DOWHY MODEL
# ============================================================

def build_refutation_model(df):

    validate_refutation_data(df)

    graph = """
    digraph {

        recency -> treatment;
        recency -> outcome;

        history -> treatment;
        history -> outcome;

        mens -> treatment;
        mens -> outcome;

        womens -> treatment;
        womens -> outcome;

        newbie -> treatment;
        newbie -> outcome;

        treatment -> outcome;
    }
    """

    model = CausalModel(
        data=df,
        treatment=TREATMENT,
        outcome=OUTCOME,
        graph=graph
    )

    return model


# ============================================================
# ORIGINAL CAUSAL EFFECT
# ============================================================

def estimate_original_effect(model):

    identified_estimand = model.identify_effect(
        proceed_when_unidentifiable=True
    )

    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )

    return identified_estimand, estimate


# ============================================================
# SAFE ATTRIBUTE EXTRACTION
# ============================================================

def get_result_value(result, attribute):

    try:

        value = getattr(
            result,
            attribute,
            None
        )

        if callable(value):

            value = value()

        return value

    except Exception:

        return None


# ============================================================
# RANDOM COMMON CAUSE
# ============================================================

def random_common_cause_test(
    model,
    identified_estimand,
    estimate
):

    result = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="random_common_cause"
    )

    return {
        "test": "random_common_cause",

        "estimated_effect":
            get_result_value(
                result,
                "estimated_effect"
            ),

        "new_effect":
            get_result_value(
                result,
                "new_effect"
            ),

        "p_value":
            get_result_value(
                result,
                "p_value"
            ),

        "result": result
    }


# ============================================================
# PLACEBO TREATMENT
# ============================================================

def placebo_treatment_test(
    model,
    identified_estimand,
    estimate
):

    result = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter"
    )

    return {
        "test": "placebo_treatment",

        "estimated_effect":
            get_result_value(
                result,
                "estimated_effect"
            ),

        "new_effect":
            get_result_value(
                result,
                "new_effect"
            ),

        "p_value":
            get_result_value(
                result,
                "p_value"
            ),

        "result": result
    }


# ============================================================
# DATA SUBSET
# ============================================================

def data_subset_test(
    model,
    identified_estimand,
    estimate
):

    result = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="data_subset_refuter",
        subset_fraction=0.9
    )

    return {
        "test": "data_subset",

        "estimated_effect":
            get_result_value(
                result,
                "estimated_effect"
            ),

        "new_effect":
            get_result_value(
                result,
                "new_effect"
            ),

        "p_value":
            get_result_value(
                result,
                "p_value"
            ),

        "result": result
    }


# ============================================================
# RUN ALL REFUTATION TESTS
# ============================================================

def run_all_refutation_tests(df):
    """
    IMPORTANT:

    This function receives the ALREADY CLEANED dataframe.

    It does NOT load hillstrom.csv again.

    Therefore Week 1, Week 2 and Refutation all use exactly
    the same customer population.
    """

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_refutation_data(df)

    # --------------------------------------------------------
    # Make a copy
    # --------------------------------------------------------

    refutation_df = df.copy()

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    refutation_df = refutation_df[
        CONFOUNDERS
        + [
            TREATMENT,
            OUTCOME
        ]
    ].dropna().copy()

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_refutation_model(
        refutation_df
    )

    # --------------------------------------------------------
    # Identify effect
    # --------------------------------------------------------

    identified_estimand, estimate = (
        estimate_original_effect(model)
    )

    original_effect = float(
        estimate.value
    )

    # --------------------------------------------------------
    # Results container
    # --------------------------------------------------------

    results = {}

    # ========================================================
    # TEST 1
    # ========================================================

    try:

        results[
            "random_common_cause"
        ] = random_common_cause_test(
            model,
            identified_estimand,
            estimate
        )

    except Exception as e:

        results[
            "random_common_cause"
        ] = {
            "test":
                "random_common_cause",

            "error":
                str(e)
        }

    # ========================================================
    # TEST 2
    # ========================================================

    try:

        results[
            "placebo_treatment"
        ] = placebo_treatment_test(
            model,
            identified_estimand,
            estimate
        )

    except Exception as e:

        results[
            "placebo_treatment"
        ] = {
            "test":
                "placebo_treatment",

            "error":
                str(e)
        }

    # ========================================================
    # TEST 3
    # ========================================================

    try:

        results[
            "data_subset"
        ] = data_subset_test(
            model,
            identified_estimand,
            estimate
        )

    except Exception as e:

        results[
            "data_subset"
        ] = {
            "test":
                "data_subset",

            "error":
                str(e)
        }

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "original_effect":
            original_effect,

        "dataset_shape":
            refutation_df.shape,

        "dataset_rows":
            len(refutation_df),

        "results":
            results
    }