##################### cr_neumf

import argparse
import logging
from pathlib import Path

import duckdb
import torch
import yaml
from torch.utils.data import DataLoader

# from src.models.mcf import MatrixFactorization, NeuralCollaborativeFiltering
# from src.training.trainer import ParquetRatingDataset, MFTrainer

from src.models.mcf import (
    MatrixFactorization,
    NeuralCollaborativeFiltering,
    MultiCriteriaMatrixFactorizationBaseline,
    DeepFMRecommender,
    MCDCF,
    CriteriaResidualNeuMF,
)

from src.training.trainer import (
    ParquetRatingDataset,
    ParquetMultiCriteriaDataset,
    MFTrainer,
    MCMFTrainer,
    MCDCFTrainer,
    CriteriaResidualNeuMFTrainer,
    DeepFMTrainer,
    ParquetDeepFMDataset,
    ParquetCriteriaResidualNeuMFDataset,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_stats(model_ready_dir: Path):
    con = duckdb.connect()
    train_path = str(model_ready_dir / "train_model.parquet").replace("\\", "/")

    stats = con.execute(f"""
    SELECT
        COUNT(DISTINCT user_idx) AS n_users,
        COUNT(DISTINCT item_idx) AS n_items,
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


def build_model(model_name: str, model_cfg: dict, stats: dict):
    if model_name == "mf":
        return MatrixFactorization(
            n_users=stats["n_users"],
            n_items=stats["n_items"],
            embedding_dim=model_cfg["embedding_dim"],
            global_mean=stats["global_mean"],
            min_rating=stats["min_rating"],
            max_rating=stats["max_rating"],
        )

    if model_name == "ncf":
        return NeuralCollaborativeFiltering(
            n_users=stats["n_users"],
            n_items=stats["n_items"],
            embedding_dim=model_cfg["embedding_dim"],
            hidden_dims=model_cfg.get("hidden_dims", [128, 64, 32]),
            dropout=model_cfg.get("dropout", 0.2),
            global_mean=stats["global_mean"],
            min_rating=stats["min_rating"],
            max_rating=stats["max_rating"],
        )

    if model_name == "mcmf":
        return MultiCriteriaMatrixFactorizationBaseline(
            n_users=stats["n_users"],
            n_items=stats["n_items"],
            n_criteria=model_cfg.get("n_criteria", 5),
            embedding_dim=model_cfg["embedding_dim"],
            global_mean=stats["global_mean"],
            min_rating=stats["min_rating"],
            max_rating=stats["max_rating"],
        )

    if model_name == "deepfm":
        return DeepFMRecommender(
            n_users=stats["n_users"],
            n_items=stats["n_items"],
            dense_dim=model_cfg.get("dense_dim", 16),
            embedding_dim=model_cfg["embedding_dim"],
            hidden_dims=model_cfg.get("hidden_dims", [256, 128, 64]),
            dropout=model_cfg.get("dropout", 0.2),
            global_mean=stats["global_mean"],
            min_rating=stats["min_rating"],
            max_rating=stats["max_rating"],
        )

    if model_name == "mcdcf":
        return MCDCF(
            n_users=stats["n_users"],
            n_items=stats["n_items"],
            n_criteria=model_cfg.get("n_criteria", 5),
            embedding_dim=model_cfg["embedding_dim"],
            hidden_dims=model_cfg.get("hidden_dims", [256, 128, 64]),
            ncf_hidden_dims=model_cfg.get("ncf_hidden_dims", [128, 64, 32]),
            attention_dim=model_cfg.get("attention_dim", 64),
            dropout=model_cfg.get("dropout", 0.3),
            global_mean=stats["global_mean"],
            min_rating=stats["min_rating"],
            max_rating=stats["max_rating"],
        )

    if model_name == "cr_neumf":
        return CriteriaResidualNeuMF(
            n_users=stats["n_users"],
            n_items=stats["n_items"],
            dense_dim=model_cfg.get("dense_dim", 16),
            n_criteria=model_cfg.get("n_criteria", 5),
            embedding_dim=model_cfg.get("embedding_dim", 64),
            gmf_dim=model_cfg.get("gmf_dim", 64),
            mlp_hidden_dims=model_cfg.get("mlp_hidden_dims", [256, 128, 64]),
            criteria_hidden_dim=model_cfg.get("criteria_hidden_dim", 64),
            dropout=model_cfg.get("dropout", 0.3),
            global_mean=stats["global_mean"],
            min_rating=stats["min_rating"],
            max_rating=stats["max_rating"],
            alpha_init=model_cfg.get("alpha_init", 0.05),
        )

    raise ValueError(f"Unsupported model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Train recommender baseline model")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/kaggle/input/datasets/mumomu/config17/config.yaml"),
        help="Path to config YAML",
    )

    parser.add_argument(
        "--model",
        choices=["mf", "ncf", "mcmf", "deepfm", "mcdcf", "cr_neumf"],
        default="deepfm",  ################################################################# mf ncf mcmf(bo) deepfm cr_neumf
    )

    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    config = load_config(args.config)

    data_cfg = config["data"]
    training_cfg = config["training"]
    model_cfg = config[args.model]

    model_ready_dir = Path("/kaggle/input/datasets/mumomu/model-ready/model_ready")
    save_dir = Path(training_cfg["save_dir"]) / f"{args.model}_baseline"
    save_dir.mkdir(parents=True, exist_ok=True)

    if training_cfg.get("device", "cpu") == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info("=" * 80)
    logger.info(f"Training baseline: {args.model.upper()}")
    logger.info("=" * 80)
    logger.info(f"Using device: {device}")

    train_path = model_ready_dir / "train_model.parquet"
    val_path = model_ready_dir / "val_warm_model.parquet"
    test_path = model_ready_dir / "test_warm_model.parquet"

    stats = get_model_stats(model_ready_dir)
    logger.info(f"Dataset stats: {stats}")

    logger.info("Loading datasets...")

    if args.model in ["mcmf", "mcdcf"]:
        dataset_cls = ParquetMultiCriteriaDataset
    elif args.model == "deepfm":
        dataset_cls = ParquetDeepFMDataset
    elif args.model in ["cgnmf", "cr_neumf"]:
        dataset_cls = ParquetCriteriaResidualNeuMFDataset
    else:
        dataset_cls = ParquetRatingDataset

    train_dataset = dataset_cls(train_path)
    val_dataset = dataset_cls(val_path)
    test_dataset = dataset_cls(test_path)

    # train_dataset = ParquetRatingDataset(train_path)
    # val_dataset = ParquetRatingDataset(val_path)
    # test_dataset = ParquetRatingDataset(test_path)

    logger.info(f"Train samples: {len(train_dataset):,}")
    logger.info(f"Val samples: {len(val_dataset):,}")
    logger.info(f"Test samples: {len(test_dataset):,}")

    batch_size = model_cfg["batch_size"]
    num_workers = training_cfg.get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"Building {args.model.upper()} model...")

    model = build_model(
        model_name=args.model,
        model_cfg=model_cfg,
        stats=stats,
    )

    logger.info(model)

    if args.model == "mcmf":
        trainer = MCMFTrainer(
            model=model,
            device=device,
            lr=model_cfg["learning_rate"],
            weight_decay=model_cfg["weight_decay"],
            lambda_overall=model_cfg.get("lambda_overall", 1.0),
            lambda_criteria=model_cfg.get("lambda_criteria", 0.3),
            save_dir=save_dir,
        )

    elif args.model == "mcdcf":
        trainer = MCDCFTrainer(
            model=model,
            device=device,
            lr=model_cfg["learning_rate"],
            weight_decay=model_cfg["weight_decay"],
            lambda_overall=model_cfg.get("lambda_overall", 1.0),
            lambda_criteria=model_cfg.get("lambda_criteria", 0.3),
            save_dir=save_dir,
        )
    elif args.model == "cr_neumf":
        trainer = CriteriaResidualNeuMFTrainer(
            model=model,
            device=device,
            lr=model_cfg["learning_rate"],
            weight_decay=model_cfg["weight_decay"],
            lambda_overall=model_cfg.get("lambda_overall", 1.0),
            lambda_criteria=model_cfg.get("lambda_criteria", 0.01),
            save_dir=save_dir,
        )

    elif args.model == "deepfm":
        trainer = DeepFMTrainer(
            model=model,
            device=device,
            lr=model_cfg["learning_rate"],
            weight_decay=model_cfg["weight_decay"],
            save_dir=save_dir,
        )

    else:
        trainer = MFTrainer(
            model=model,
            device=device,
            lr=model_cfg["learning_rate"],
            weight_decay=model_cfg["weight_decay"],
            save_dir=save_dir,
        )

    logger.info("=" * 80)
    logger.info("START TRAINING")
    logger.info("=" * 80)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=model_cfg["epochs"],
        patience=model_cfg.get(
            "patience",
            training_cfg.get("early_stopping_patience", 3),
        ),
    )

    # checkpoint_name = "best_mf.pt"
    # checkpoint_path = save_dir / checkpoint_name

    if args.model == "mcmf":
        checkpoint_name = "best_mcmf.pt"
    elif args.model == "mcdcf":
        checkpoint_name = "best_mcdcf.pt"
    elif args.model == "deepfm":
        checkpoint_name = "best_deepfm.pt"
    elif args.model == "cr_neumf":
        checkpoint_name = "best_cr_neumf.pt"
    else:
        checkpoint_name = "best_mf.pt"

    checkpoint_path = save_dir / checkpoint_name

    logger.info(f"Loading best checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    logger.info("=" * 80)
    logger.info("FINAL TEST EVALUATION")
    logger.info("=" * 80)

    test_metrics = trainer.evaluate(
        test_loader,
        split_name="test",
    )

    logger.info(f"Test metrics: {test_metrics}")
    from src.training.trainer import predict_and_save

    pred_dir = Path("predictions") / args.model
    pred_dir.mkdir(parents=True, exist_ok=True)

    predict_and_save(
        trainer=trainer,
        data_loader=val_loader,
        save_path=pred_dir / "val_predictions.npz",
        model_type=args.model,
    )

    predict_and_save(
        trainer=trainer,
        data_loader=test_loader,
        save_path=pred_dir / "test_predictions.npz",
        model_type=args.model,
    )

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
