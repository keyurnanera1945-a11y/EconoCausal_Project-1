from dowhy import CausalModel


def build_causal_model(
    df,
    treatment="treatment",
    outcome="outcome",
    common_causes=None
):
    """
    Build a DoWhy causal model.

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared dataset.

    treatment : str
        Treatment column.

    outcome : str
        Outcome column.

    common_causes : list
        Confounding/customer feature columns.

    Returns
    -------
    CausalModel
    """

    if common_causes is None:
        common_causes = []

    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=common_causes
    )

    return model


def identify_effect(model):
    """
    Identify the causal effect using DoWhy.
    """

    identified_estimand = model.identify_effect(
        proceed_when_unidentifiable=True
    )

    return identified_estimand


def estimate_causal_effect(
    model,
    identified_estimand
):
    """
    Estimate the causal effect using
    a backdoor adjustment method.
    """

    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )

    return estimate


def run_causal_analysis(
    df,
    treatment="treatment",
    outcome="outcome",
    common_causes=None
):
    """
    Complete causal analysis:

    1. Build causal model
    2. Identify causal effect
    3. Estimate causal effect

    Returns
    -------
    model
    identified_estimand
    estimate
    """

    model = build_causal_model(
        df=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=common_causes
    )

    identified_estimand = identify_effect(model)

    estimate = estimate_causal_effect(
        model,
        identified_estimand
    )

    return (
        model,
        identified_estimand,
        estimate
    )