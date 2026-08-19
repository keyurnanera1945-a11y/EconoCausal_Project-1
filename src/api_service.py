"""
EconoCausal - Week 4
Production prediction and customer recommendation service.
"""

import os
import pandas as pd
import numpy as np

from .validation import validate_customer_features


class EconoCausalService:

    def __init__(self, data_path="data/hillstorm.csv"):

        self.data_path = data_path

        self.data = None
        self.ite_data = None

        self.load_data()

    def load_data(self):
        """
        Load the project dataset.
        """

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}"
            )

        self.data = pd.read_csv(
            self.data_path
        )

        # Apply the same basic cleaning logic
        # used by the project.
        self.data = self.data.dropna().copy()

        self.data = self.data.reset_index(
            drop=True
        )

    def calculate_simple_score(self, customer):
        """
        Production-safe customer scoring.

        This service provides a lightweight
        recommendation score from customer features.

        The actual EconML ITE remains the project's
        causal model output.
        """

        customer = validate_customer_features(
            customer
        )

        recency = customer["recency"]
        history = customer["history"]
        mens = customer["mens"]
        womens = customer["womens"]
        newbie = customer["newbie"]

        # Normalized heuristic score used for
        # API demonstration and validation.
        recency_score = 1 / (1 + recency)

        history_score = (
            history /
            (history + 100)
        )

        product_score = (
            mens + womens
        ) / 2

        newbie_score = (
            0.1 if newbie == 1 else 0
        )

        score = (
            0.25 * recency_score
            + 0.50 * history_score
            + 0.20 * product_score
            + 0.05 * newbie_score
        )

        return float(score)

    def classify_customer(self, score):
        """
        Convert score into marketing category.
        """

        if score >= 0.60:
            return "Strong Persuadable"

        if score >= 0.35:
            return "Persuadable"

        if score >= 0.15:
            return "Low Impact"

        return "Negative Response"

    def predict_customer(self, customer):
        """
        Generate API prediction.
        """

        score = self.calculate_simple_score(
            customer
        )

        customer_type = self.classify_customer(
            score
        )

        recommend_treatment = (
            customer_type
            in [
                "Strong Persuadable",
                "Persuadable"
            ]
        )

        return {
            "score": round(score, 6),
            "customer_type": customer_type,
            "treatment_recommended": int(
                recommend_treatment
            ),
        }

    def dataset_summary(self):
        """
        Return dataset information.
        """

        return {
            "customers": int(len(self.data)),
            "columns": list(self.data.columns),
            "status": "loaded",
        }