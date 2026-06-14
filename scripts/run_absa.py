"""
Phase 3: Aspect-Based Sentiment Analysis (ABSA)
"""

import argparse
from pathlib import Path

import yaml
import torch

from src.models.absa import HybridABSA, build_default_absa_config


def load_config(config_path: Path | None):
    if config_path is None or not config_path.exists():
        print("Using default ABSA config...")
        return build_default_absa_config()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def print_system_info():
    print("=" * 60)

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("=" * 60)


def validate_paths(input_path: Path, output_dir: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Run ABSA module on parquet dataset")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to YAML config file",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/train.parquet"),
        help="Input parquet file path",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/absa_train_chunks"),
        help="Directory to write ABSA outputs",
    )

    parser.add_argument(
        "--text-col", type=str, default="review_text", help="Review text column name"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20000,
        help="Chunk size for parquet processing",
    )

    parser.add_argument(
        "--mode", choices=["chunks", "single"], default="chunks", help="Output mode"
    )

    args = parser.parse_args()

    print_system_info()

    validate_paths(args.input, args.output_dir)

    config = load_config(args.config)

    print("\nLoading ABSA model...")
    absa = HybridABSA(config)

    print("\nRunning ABSA pipeline...")
    print("Input:", args.input)
    print("Output dir:", args.output_dir)
    print("Mode:", args.mode)
    print("Chunk size:", args.chunk_size)

    if args.mode == "chunks":
        absa.process_parquet_to_chunks(
            input_path=args.input,
            output_dir=args.output_dir,
            text_col=args.text_col,
            chunk_size=args.chunk_size,
        )

    else:
        output_file = args.output_dir / "absa_output.parquet"

        absa.process_parquet_file(
            input_path=args.input,
            output_path=output_file,
            text_col=args.text_col,
            chunk_size=args.chunk_size,
        )

    print("\nABSA pipeline completed successfully.")


if __name__ == "__main__":
    main()
