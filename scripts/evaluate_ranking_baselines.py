from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict

import duckdb
import numpy as np
import pandas as pd
import torch
import yaml

from src.models.mcf import (
    MatrixFactorization,
    NeuralCollaborativeFiltering,
    DeepFMRecommender,
)

from src.training.ranking_evaluator import (
    evaluate_ranking_model,
    load_eval_data,
    load_train_user_items,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_model_stats(model_ready_dir: Path) -> Dict[str, float]:
    con = duckdb.connect()

    train_path = str(model_ready_dir / "train_model.parquet").replace("\\", "/")
    user_map_path = str(model_ready_dir / "user2idx.parquet").replace("\\", "/")
    item_map_path = str(model_ready_dir / "item2idx.parquet").replace("\\", "/")

    stats = con.execute(f"""
        SELECT
            (SELECT COUNT(*) FROM read_parquet('{user_map_path}')) AS n_users,
            (SELECT COUNT(*) FROM read_parquet('{item_map_path}')) AS n_items,
            AVG(rating) AS global_mean,
            MIN(rating) AS min_rating,
            MAX(rating) AS max_rating
        FROM read_parquet('{train_path}')
    """).df().iloc[0]

    return {
        "n_users": int(stats["n_users"]),
        "n_items": int(stats["n_items"]),
        "global_mean": float(stats["global_mean"]),
        "min_rating": float(stats["min_rating"]),
        "max_rating": float(stats["max_rating"]),
    }


def build_deepfm_item_dense_matrix(
    train_path: Path,
    n_items: int,
) -> np.ndarray:
    criteria_cols = [
        "item_quality",
        "item_value",
        "item_design",
        "item_usability",
        "item_durability",
    ]
    mask_cols = [
        "mask_quality",
        "mask_value",
        "mask_design",
        "mask_usability",
        "mask_durability",
    ]
    dense_cols = criteria_cols + mask_cols

    df = pd.read_parquet(train_path, columns=["item_idx"] + dense_cols)

    df[criteria_cols] = (df[criteria_cols].astype("float32") - 3.0) / 2.0
    df[mask_cols] = df[mask_cols].astype("float32")

    item_df = df.groupby("item_idx")[dense_cols].mean().astype("float32")

    item_dense = np.zeros((n_items, len(dense_cols)), dtype=np.float32)

    valid_items = item_df.index.to_numpy(np.int64)
    valid_items = valid_items[(valid_items >= 0) & (valid_items < n_items)]

    item_dense[valid_items] = item_df.loc[valid_items, dense_cols].to_numpy(
        dtype=np.float32
    )

    return item_dense


def build_model(model_cfg: dict, stats: dict):
    model_type = model_cfg["model_type"]

    if model_type == "mf":
        return MatrixFactorization(
            n_users=stats["n_users"],
            n_items=stats["n_items"],
            embedding_dim=model_cfg.get("embedding_dim", 64),
            global_mean=stats["global_mean"],
            min_rating=stats["min_rating"],
            max_rating=stats["max_rating"],
        )

    if model_type == "ncf":
        return NeuralCollaborativeFiltering(
            n_users=stats["n_users"],
            n_items=stats["n_items"],
            embedding_dim=model_cfg.get("embedding_dim", 64),
            hidden_dims=model_cfg.get("hidden_dims", [128, 64, 32]),
            dropout=model_cfg.get("dropout", 0.2),
            global_mean=stats["global_mean"],
            min_rating=stats["min_rating"],
            max_rating=stats["max_rating"],
        )

    if model_type == "deepfm":
        return DeepFMRecommender(
            n_users=stats["n_users"],
            n_items=stats["n_items"],
            dense_dim=model_cfg.get("dense_dim", 10),
            embedding_dim=model_cfg.get("embedding_dim", 32),
            hidden_dims=model_cfg.get("hidden_dims", [256, 128, 64]),
            dropout=model_cfg.get("dropout", 0.2),
            global_mean=stats["global_mean"],
            min_rating=stats["min_rating"],
            max_rating=stats["max_rating"],
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def evaluate_one_model(cfg: dict, model_key: str) -> Dict[str, dict]:
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    paths = cfg["paths"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    metrics_cfg = cfg["metrics"]
    model_cfg = cfg["models"][model_key]

    model_ready_dir = Path("data/processed/model_ready")
    output_dir = Path("outputs/ranking_baselines") / model_key
    ckpt_path = output_dir / "best_model.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    train_path = model_ready_dir / paths.get("train_file", "train_model.parquet")

    val_warm_path = model_ready_dir / "val_warm_model.parquet"
    test_warm_path = model_ready_dir / "test_warm_model.parquet"

    val_full_path = model_ready_dir / "val_model.parquet"
    test_full_path = model_ready_dir / "test_model.parquet"

    stats = get_model_stats(model_ready_dir)
    device = choose_device(train_cfg.get("device", "cuda"))

    logger.info("=" * 80)
    logger.info(f"Evaluating ranking baseline: {model_key}")
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Stats: {stats}")
    logger.info("=" * 80)

    model = build_model(model_cfg, stats).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    model_type = model_cfg["model_type"]
    positive_threshold = float(data_cfg.get("positive_threshold", 4.0))

    train_user_items = load_train_user_items(train_path)

    item_dense_matrix = None
    if model_type == "deepfm":
        item_dense_matrix = build_deepfm_item_dense_matrix(
            train_path=train_path,
            n_items=stats["n_items"],
        )

    k_values = metrics_cfg.get("top_k", [5, 10, 20])
    max_eval_users = data_cfg.get("max_eval_users", 2000)
    num_negatives_eval = int(data_cfg.get("num_negatives_eval", 100))
    filter_seen_items = bool(data_cfg.get("filter_seen_items", True))

    eval_splits = {
        "val_warm": val_warm_path,
        "test_warm": test_warm_path,
        "val_full": val_full_path,
        "test_full": test_full_path,
    }

    final = {}

    for split_name, split_path in eval_splits.items():
        logger.info(f"Evaluating split: {split_name} | {split_path}")

        positives = load_eval_data(
            split_path,
            positive_threshold=positive_threshold,
        )

        final[split_name] = evaluate_ranking_model(
            model=model,
            model_type=model_type,
            eval_positives_by_user=positives,
            train_user_items=train_user_items,
            n_items=stats["n_items"],
            device=device,
            k_values=k_values,
            num_negatives=num_negatives_eval,
            max_eval_users=max_eval_users,
            filter_seen_items=filter_seen_items,
            seed=seed + hash(split_name) % 10000,
            item_dense_matrix=item_dense_matrix,
        )

    save_path = output_dir / "metrics_full_eval.json"
    save_json(final, save_path)

    logger.info(f"Saved full evaluation metrics to: {save_path}")

    test_full = final.get("test_full", {})
    logger.info(
        f"{model_key} test_full | "
        f"ndcg@10={test_full.get('ndcg@10')} | "
        f"recall@10={test_full.get('recall@10')} | "
        f"hit@10={test_full.get('hit@10')}"
    )

    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="mf_bpr, ncf_bpr, deepfm_bpr, or all",
    )

    args, _ = parser.parse_known_args()

    root_cfg = load_yaml(args.config)
    cfg = root_cfg["ranking_baselines"]

    available = list(cfg["models"].keys())

    if args.model == "all":
        selected = available
    else:
        if args.model not in available:
            raise ValueError(f"Unknown model={args.model}. Available={available}")
        selected = [args.model]

    all_results = {}

    for model_key in selected:
        result = evaluate_one_model(cfg, model_key)
        all_results[model_key] = result

    summary_path = (
        Path(cfg["paths"]["output_dir"]) / "ranking_baseline_full_eval_summary.json"
    )
    save_json(all_results, summary_path)

    logger.info("=" * 80)
    logger.info("FULL RANKING BASELINE EVALUATION SUMMARY")
    logger.info("=" * 80)

    for model_key, result in all_results.items():
        for split_name in ["val_warm", "test_warm", "val_full", "test_full"]:
            m = result.get(split_name, {})
            logger.info(
                f"{model_key} | {split_name} | "
                f"ndcg@10={m.get('ndcg@10')} | "
                f"recall@10={m.get('recall@10')} | "
                f"hit@10={m.get('hit@10')}"
            )


if __name__ == "__main__":
    main()
