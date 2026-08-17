import argparse
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Criteo Uplift Machine Learning Data Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML pipeline configuration file"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional path to dataset CSV file (overrides config raw_data_path)"
    )

    args = parser.parse_args()

    print("==================================================")
    print("      CRITEO UPLIFT ML DATA PIPELINE RUNNER       ")
    print("==================================================")

    results = run_pipeline(config_path=args.config, data_path=args.data)
    print("\nSummary Metrics:")
    print(results["summary_df"].to_string(index=False))


if __name__ == "__main__":
    main()
