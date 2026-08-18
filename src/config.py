import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class PipelineConfig:
    raw_data_path: str
    processed_dir: str
    feature_cols: List[str]
    treatment_col: str
    outcome_col: str
    secondary_outcomes: List[str]
    impute_cols: List[str]
    impute_strategy: str
    drop_missing: bool
    remove_duplicates: bool
    test_size: float
    random_state: int
    results_dir: str
    tables_dir: str
    summary_file: str

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        path = Path(yaml_path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            path = project_root / yaml_path

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        prep = data.get("preprocessing", {})
        imp = prep.get("imputation", {})
        split = prep.get("split", {})

        return cls(
            raw_data_path=data["dataset"]["raw_data_path"],
            processed_dir=data["dataset"]["processed_dir"],
            feature_cols=data["features"]["feature_cols"],
            treatment_col=data["features"]["treatment_col"],
            outcome_col=data["features"]["outcome_col"],
            secondary_outcomes=data["features"].get("secondary_outcomes", []),
            impute_cols=imp.get("cols", []),
            impute_strategy=imp.get("strategy", "dropna"),
            drop_missing=prep.get("drop_missing", True),
            remove_duplicates=prep.get("remove_duplicates", True),
            test_size=split.get("test_size", 0.20),
            random_state=split.get("random_state", 42),
            results_dir=data["outputs"]["results_dir"],
            tables_dir=data["outputs"]["tables_dir"],
            summary_file=data["outputs"]["summary_file"],
        )


def load_config(config_path: str = "config/config.yaml") -> PipelineConfig:
    return PipelineConfig.from_yaml(config_path)
