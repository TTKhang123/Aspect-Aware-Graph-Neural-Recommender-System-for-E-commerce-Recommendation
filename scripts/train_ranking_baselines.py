from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, Tuple

import duckdb
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.models.mcf import (
    MatrixFactorization,
    NeuralCollaborativeFiltering,
    DeepFMRecommender,
)
from src.training.ranking_trainer import (
    BPRInteractionDataset,
    BPRDeepFMDataset,
    train_one_epoch_bpr,
    train_one_epoch_deepfm_bpr,
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


# def get_model_stats(model_ready_dir: Path) -> Dict[str, float]:
#     con = duckdb.connect()
#     train_path = str(model_ready_dir / "train_model.parquet").replace("\\", "/")

#     stats = con.execute(f"""
#         SELECT
#             MAX(user_idx) + 1 AS n_users,
#             MAX(item_idx) + 1 AS n_items,
#             AVG(rating) AS global_mean,
#             MIN(rating) AS min_rating,
#             MAX(rating) AS max_rating
#         FROM read_parquet('{train_path}')
#         """).df().iloc[0]

#     return {
#         "n_users": int(stats["n_users"]),
#         "n_items": int(stats["n_items"]),
#         "global_mean": float(stats["global_mean"]),
#         "min_rating": float(stats["min_rating"]),
#         "max_rating": float(stats["max_rating"]),
#     }


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

    df = pd.read_parquet(
        train_path,
        columns=["item_idx"] + dense_cols,
    )

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


def build_model(
    model_key: str,
    model_cfg: dict,
    stats: dict,
):
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

    raise ValueError(f"Unsupported ranking baseline model_type: {model_type}")


def train_one_model(
    cfg: dict,
    model_key: str,
) -> Dict[str, dict]:
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    paths = cfg["paths"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    metrics_cfg = cfg["metrics"]
    model_cfg = cfg["models"][model_key]

    model_ready_dir = Path("/data/processed/model_ready")
    output_dir = Path(paths["output_dir"]) / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = model_ready_dir / paths.get("train_file", "train_model.parquet")
    val_path = model_ready_dir / paths.get("val_file", "val_warm_model.parquet")
    test_path = model_ready_dir / paths.get("test_file", "test_warm_model.parquet")

    stats = get_model_stats(model_ready_dir)

    device = choose_device(train_cfg.get("device", "cuda"))

    logger.info("=" * 80)
    logger.info(f"Training ranking baseline: {model_key}")
    logger.info(f"Device: {device}")
    logger.info(f"Stats: {stats}")
    logger.info("=" * 80)

    model = build_model(model_key, model_cfg, stats).to(device)

    model_type = model_cfg["model_type"]
    positive_threshold = float(data_cfg.get("positive_threshold", 4.0))

    if model_type == "deepfm":
        train_dataset = BPRDeepFMDataset(
            parquet_path=train_path,
            n_items=stats["n_items"],
            positive_threshold=positive_threshold,
            seed=seed,
        )
        item_dense_matrix = build_deepfm_item_dense_matrix(
            train_path=train_path,
            n_items=stats["n_items"],
        )
    else:
        train_dataset = BPRInteractionDataset(
            parquet_path=train_path,
            n_items=stats["n_items"],
            positive_threshold=positive_threshold,
            seed=seed,
        )
        item_dense_matrix = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 8192)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )

    val_positives = load_eval_data(
        val_path,
        positive_threshold=positive_threshold,
    )
    test_positives = load_eval_data(
        test_path,
        positive_threshold=positive_threshold,
    )
    train_user_items = load_train_user_items(train_path)

    k_values = metrics_cfg.get("top_k", [5, 10, 20])
    max_eval_users = data_cfg.get("max_eval_users", 2000)
    num_negatives_eval = int(data_cfg.get("num_negatives_eval", 100))
    filter_seen_items = bool(data_cfg.get("filter_seen_items", True))

    best_metric = -1.0
    best_path = output_dir / "best_model.pt"
    history = []
    bad_epochs = 0
    patience = int(train_cfg.get("patience", 5))

    for epoch in range(1, int(train_cfg.get("epochs", 30)) + 1):
        logger.info(f"Epoch {epoch}")

        if model_type == "deepfm":
            train_loss = train_one_epoch_deepfm_bpr(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                grad_clip=train_cfg.get("grad_clip_norm", 5.0),
            )
        else:
            train_loss = train_one_epoch_bpr(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                grad_clip=train_cfg.get("grad_clip_norm", 5.0),
            )

        val_metrics = evaluate_ranking_model(
            model=model,
            model_type=model_type,
            eval_positives_by_user=val_positives,
            train_user_items=train_user_items,
            n_items=stats["n_items"],
            device=device,
            k_values=k_values,
            num_negatives=num_negatives_eval,
            max_eval_users=max_eval_users,
            filter_seen_items=filter_seen_items,
            seed=seed + epoch,
            item_dense_matrix=item_dense_matrix,
        )

        monitor = float(val_metrics.get("ndcg@10", val_metrics.get("recall@10", 0.0)))

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val": val_metrics,
            }
        )
        save_json({"history": history}, output_dir / "history.json")

        logger.info(
            f"train_loss={train_loss:.6f} | "
            f"val_ndcg@10={val_metrics.get('ndcg@10'):.6f} | "
            f"val_recall@10={val_metrics.get('recall@10'):.6f}"
        )

        if monitor > best_metric:
            best_metric = monitor
            bad_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "model_key": model_key,
                    "model_type": model_type,
                    "stats": stats,
                    "best_val_metric": best_metric,
                },
                best_path,
            )

            logger.info(f"Saved best checkpoint: {best_path}")
        else:
            bad_epochs += 1
            logger.info(f"No improvement: {bad_epochs}/{patience}")

            if bad_epochs >= patience:
                logger.info("Early stopping triggered.")
                break

    logger.info("Loading best checkpoint for final test evaluation.")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    final = {}

    final["val_warm"] = evaluate_ranking_model(
        model=model,
        model_type=model_type,
        eval_positives_by_user=val_positives,
        train_user_items=train_user_items,
        n_items=stats["n_items"],
        device=device,
        k_values=k_values,
        num_negatives=num_negatives_eval,
        max_eval_users=max_eval_users,
        filter_seen_items=filter_seen_items,
        seed=seed + 1000,
        item_dense_matrix=item_dense_matrix,
    )

    final["test_warm"] = evaluate_ranking_model(
        model=model,
        model_type=model_type,
        eval_positives_by_user=test_positives,
        train_user_items=train_user_items,
        n_items=stats["n_items"],
        device=device,
        k_values=k_values,
        num_negatives=num_negatives_eval,
        max_eval_users=max_eval_users,
        filter_seen_items=filter_seen_items,
        seed=seed + 2000,
        item_dense_matrix=item_dense_matrix,
    )

    save_json(final, output_dir / "metrics.json")

    logger.info(f"Final metrics saved to {output_dir / 'metrics.json'}")
    logger.info(str(final))

    return final


def main():
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

    # args = parser.parse_args()
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
        result = train_one_model(cfg, model_key)
        all_results[model_key] = result

        save_json(
            all_results,
            Path(cfg["paths"]["output_dir"]) / "ranking_baseline_summary.json",
        )

    logger.info("=" * 80)
    logger.info("RANKING BASELINE SUMMARY")
    logger.info("=" * 80)

    for model_key, result in all_results.items():
        test = result.get("test_warm", {})
        logger.info(
            f"{model_key} | "
            f"test_ndcg@10={test.get('ndcg@10')} | "
            f"test_recall@10={test.get('recall@10')} | "
            f"test_hit@10={test.get('hit@10')}"
        )


if __name__ == "__main__":
    main()
