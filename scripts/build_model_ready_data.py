"""
Phase 4: Build Model-Ready Data

This script builds model-ready datasets by:
1. Creating item-level criteria vectors from ABSA train chunks
2. Merging criteria vectors into train/val/test/val_warm/test_warm splits
3. Building train-only user/item ID mappings
4. Creating encoded model-ready datasets
5. Validating merged and model-ready outputs

Outputs:
    data/processed/item_criteria_vectors_train.parquet

    data/processed/train_with_item_criteria.parquet
    data/processed/val_with_item_criteria.parquet
    data/processed/test_with_item_criteria.parquet
    data/processed/val_warm_with_item_criteria.parquet
    data/processed/test_warm_with_item_criteria.parquet

    data/processed/model_ready/user2idx.parquet
    data/processed/model_ready/item2idx.parquet

    data/processed/model_ready/train_model.parquet
    data/processed/model_ready/val_model.parquet
    data/processed/model_ready/test_model.parquet
    data/processed/model_ready/val_warm_model.parquet
    data/processed/model_ready/test_warm_model.parquet
    data/processed/model_ready/model_ready_summary.csv
"""

import argparse
import logging
from pathlib import Path

import yaml

from src.data.model_ready_data import build_model_ready_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def log_dataframe(title: str, df) -> None:
    """Pretty-print pandas DataFrame to logger."""
    if df is not None:
        logger.info("\n%s:", title)
        logger.info("\n%s\n", df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build model-ready datasets for training"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML file",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate datasets after building (default: True)",
    )

    parser.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="Skip validation steps",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PHASE 4: BUILD MODEL-READY DATA")
    logger.info("=" * 80)

    try:
        config = load_config(args.config)

        processed_dir = config["data"].get("processed_dir", "data/processed")

        logger.info("Config loaded from: %s", args.config)
        logger.info("Processing directory: %s", processed_dir)
        logger.info("Validation enabled: %s", args.validate)

        results = build_model_ready_data(
            config=config,
            validate=args.validate,
        )

        logger.info("\n" + "=" * 80)
        logger.info("PHASE 4 COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        log_dataframe(
            "Item Criteria Coverage",
            results.get("item_criteria_coverage"),
        )

        log_dataframe(
            "Merged Dataset Validation Results",
            results.get("validation_results"),
        )

        log_dataframe(
            "ID Mapping Stats",
            results.get("mapping_stats"),
        )

        log_dataframe(
            "Model-ready Stats",
            results.get("model_ready_stats"),
        )

        log_dataframe(
            "Overlap Stats",
            results.get("overlap_stats"),
        )

        logger.info("Output files:")
        logger.info("  - data/processed/item_criteria_vectors_train.parquet")
        logger.info("  - data/processed/train_with_item_criteria.parquet")
        logger.info("  - data/processed/val_with_item_criteria.parquet")
        logger.info("  - data/processed/test_with_item_criteria.parquet")
        logger.info("  - data/processed/val_warm_with_item_criteria.parquet")
        logger.info("  - data/processed/test_warm_with_item_criteria.parquet")
        logger.info("  - data/processed/model_ready/user2idx.parquet")
        logger.info("  - data/processed/model_ready/item2idx.parquet")
        logger.info("  - data/processed/model_ready/train_model.parquet")
        logger.info("  - data/processed/model_ready/val_model.parquet")
        logger.info("  - data/processed/model_ready/test_model.parquet")
        logger.info("  - data/processed/model_ready/val_warm_model.parquet")
        logger.info("  - data/processed/model_ready/test_warm_model.parquet")
        logger.info("  - data/processed/model_ready/model_ready_summary.csv")

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    main()
