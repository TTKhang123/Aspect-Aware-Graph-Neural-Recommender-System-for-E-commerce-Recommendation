"""
Phase 2: Data Processing Pipeline

This file is used to run the pipeline in ../src/data/data_loader.py

This file defines the main pipeline for processing Amazon review and metadata data.
It includes functions to load configuration from a YAML file, run the data processing pipeline,
and handle command-line arguments for flexibility in execution.
The pipeline processes review and metadata files in chunks,
allowing for efficient handling of large datasets.

Output:
    ../data/temp_metadata (contain chunked parquet files for cleaned metadata),
    ../data/temp_reviews (contain chunked parquet files for cleaned reviews),

    ../data/processed/reviews_dedup.parquet (deduplicated reviews parquet file),
    ../data/processed/reviews_filtered_min5.parquet (filtered reviews from reviews_dedup.parquet),
    ../data/processed/item_clean_min5.parquet (filtered metadata parquet file),

    ../data/processed/reviews_merged_min5.parquet
    (merge item_clean_min5.parquet and reviews_filtered_min5.parquet on asin)

    ../data/processed/train.parquet
    ../data/processed/val.parquet
    ../data/processed/test.parquet
    ../data/processed/val_warm.parquet
    ../data/processed/test_warm.parquet

    CSV files for checking (all located in ../data/processed/):
"""

import argparse
import logging
from pathlib import Path

import yaml

from src.data.data_loader import AmazonDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(
    config_path: str,
    review_limit: int | None = None,
    metadata_limit: int | None = None,
    review_chunk_size: int = 300_000,
    metadata_chunk_size: int = 100_000,
    clear_temp: bool = True,
):
    config = load_config(config_path)

    loader = AmazonDataLoader(config)

    loader.run_pipeline(
        review_chunk_size=review_chunk_size,
        metadata_chunk_size=metadata_chunk_size,
        review_limit=review_limit,
        metadata_limit=metadata_limit,
        clear_temp=clear_temp,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Amazon data processing pipeline")

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML file",
    )

    parser.add_argument(
        "--review-limit",
        type=int,
        default=None,
        help="Limit number of review rows for testing",
    )

    parser.add_argument(
        "--metadata-limit",
        type=int,
        default=None,
        help="Limit number of metadata rows for testing",
    )

    parser.add_argument(
        "--review-chunk-size",
        type=int,
        default=300_000,
        help="Review chunk size",
    )

    parser.add_argument(
        "--metadata-chunk-size",
        type=int,
        default=100_000,
        help="Metadata chunk size",
    )

    parser.add_argument(
        "--no-clear-temp",
        action="store_true",
        help="Do not clear old temp parquet chunks before running",
    )

    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        review_limit=args.review_limit,
        metadata_limit=args.metadata_limit,
        review_chunk_size=args.review_chunk_size,
        metadata_chunk_size=args.metadata_chunk_size,
        clear_temp=not args.no_clear_temp,
    )
