import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

from src.training.evaluator import rating_metrics

logger = logging.getLogger(__name__)


class ParquetRatingDataset(Dataset):
    """
    Dataset for MF baseline.

    Required columns:
        user_idx
        item_idx
        rating
    """

    def __init__(self, parquet_path: str | Path):
        self.parquet_path = Path(parquet_path)

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        df = pd.read_parquet(
            self.parquet_path,
            columns=["user_idx", "item_idx", "rating"],
        )

        self.user_idx = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.item_idx = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.rating = torch.tensor(df["rating"].values, dtype=torch.float32)

        del df

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        return {
            "user_idx": self.user_idx[idx],
            "item_idx": self.item_idx[idx],
            "rating": self.rating[idx],
        }


class MFTrainer:
    def __init__(
        self,
        model,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        save_dir: str | Path = "models/mf_baseline",
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.loss_fn = torch.nn.MSELoss()

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_rmse": [],
            "val_mae": [],
        }

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()

        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")

        for batch in pbar:
            user_idx = batch["user_idx"].to(self.device)
            item_idx = batch["item_idx"].to(self.device)
            rating = batch["rating"].to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(user_idx, item_idx)
            loss = self.loss_fn(pred, rating)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(
        self, data_loader: DataLoader, split_name: str = "val"
    ) -> Dict[str, float]:
        self.model.eval()

        all_preds: List[float] = []
        all_targets: List[float] = []

        for batch in tqdm(data_loader, desc=f"[{split_name}]"):
            user_idx = batch["user_idx"].to(self.device)
            item_idx = batch["item_idx"].to(self.device)
            rating = batch["rating"].to(self.device)

            pred = self.model(user_idx, item_idx)

            all_preds.append(pred.detach().cpu().numpy())
            all_targets.append(rating.detach().cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        return rating_metrics(all_preds, all_targets)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        patience: int = 3,
    ):
        best_val_rmse = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader, split_name="val")

            self.history["train_loss"].append(train_loss)
            self.history["val_rmse"].append(val_metrics["rmse"])
            self.history["val_mae"].append(val_metrics["mae"])

            logger.info(
                f"Epoch {epoch} | "
                f"train_loss={train_loss:.4f} | "
                f"val_rmse={val_metrics['rmse']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f}"
            )

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_rmse": val_metrics["rmse"],
                        "val_mae": val_metrics["mae"],
                    },
                    # self.save_dir / "best_mf.pt",
                    self.save_dir
                    / (
                        "best_bias.pt"
                        if self.model.__class__.__name__ == "BiasOnlyRecommender"
                        else "best_mf.pt"
                    ),
                )

                # logger.info(f"Saved best MF model: RMSE={best_val_rmse:.4f}")
                model_label = self.model.__class__.__name__
                logger.info(f"Saved best {model_label} model: RMSE={best_val_rmse:.4f}")

            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

        return self.history


class ParquetMultiCriteriaDataset(Dataset):
    """
    Dataset for Multi-Criteria MF.

    Required columns:
        user_idx, item_idx, rating
        item_quality, item_value, item_design, item_usability, item_durability
        mask_quality, mask_value, mask_design, mask_usability, mask_durability
    """

    def __init__(self, parquet_path):
        self.parquet_path = Path(parquet_path)

        cols = [
            "user_idx",
            "item_idx",
            "rating",
            "item_quality",
            "item_value",
            "item_design",
            "item_usability",
            "item_durability",
            "mask_quality",
            "mask_value",
            "mask_design",
            "mask_usability",
            "mask_durability",
        ]

        df = pd.read_parquet(self.parquet_path, columns=cols)

        self.user_idx = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.item_idx = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.rating = torch.tensor(df["rating"].values, dtype=torch.float32)

        self.criteria = torch.tensor(
            df[
                [
                    "item_quality",
                    "item_value",
                    "item_design",
                    "item_usability",
                    "item_durability",
                ]
            ].values,
            dtype=torch.float32,
        )

        self.criteria_mask = torch.tensor(
            df[
                [
                    "mask_quality",
                    "mask_value",
                    "mask_design",
                    "mask_usability",
                    "mask_durability",
                ]
            ].values,
            dtype=torch.float32,
        )

        del df

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        return {
            "user_idx": self.user_idx[idx],
            "item_idx": self.item_idx[idx],
            "rating": self.rating[idx],
            "criteria": self.criteria[idx],
            "criteria_mask": self.criteria_mask[idx],
        }


class MCMFTrainer:
    def __init__(
        self,
        model,
        device,
        lr=1e-3,
        weight_decay=1e-5,
        lambda_overall=1.0,
        lambda_criteria=0.3,
        save_dir="models/mcmf_baseline",
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        from src.models.mcf import MaskedMultiCriteriaLoss

        self.loss_fn = MaskedMultiCriteriaLoss(
            lambda_overall=lambda_overall,
            lambda_criteria=lambda_criteria,
        )

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_rmse": [],
            "val_mae": [],
            "val_criteria_rmse": [],
        }

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()

        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")

        for batch in pbar:
            user_idx = batch["user_idx"].to(self.device)
            item_idx = batch["item_idx"].to(self.device)
            rating = batch["rating"].to(self.device)
            criteria = batch["criteria"].to(self.device)
            criteria_mask = batch["criteria_mask"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(user_idx, item_idx)

            loss, loss_dict = self.loss_fn(
                outputs=outputs,
                rating=rating,
                criteria_targets=criteria,
                criteria_masks=criteria_mask,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss_dict['total']:.4f}",
                    "overall": f"{loss_dict['overall']:.4f}",
                    "criteria": f"{loss_dict['criteria']:.4f}",
                }
            )

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self, data_loader, split_name="val"):
        self.model.eval()

        all_preds = []
        all_targets = []

        criteria_preds_all = []
        criteria_targets_all = []
        criteria_masks_all = []

        for batch in tqdm(data_loader, desc=f"[{split_name}]"):
            user_idx = batch["user_idx"].to(self.device)
            item_idx = batch["item_idx"].to(self.device)
            rating = batch["rating"].to(self.device)
            criteria = batch["criteria"].to(self.device)
            criteria_mask = batch["criteria_mask"].to(self.device)

            outputs = self.model(user_idx, item_idx)

            all_preds.append(outputs["overall"].detach().cpu().numpy())
            all_targets.append(rating.detach().cpu().numpy())

            criteria_preds_all.append(outputs["criteria"].detach().cpu().numpy())
            criteria_targets_all.append(criteria.detach().cpu().numpy())
            criteria_masks_all.append(criteria_mask.detach().cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        metrics = rating_metrics(all_preds, all_targets)

        cp = np.concatenate(criteria_preds_all)
        ct = np.concatenate(criteria_targets_all)
        cm = np.concatenate(criteria_masks_all)

        se = ((cp - ct) ** 2) * cm
        denom = np.maximum(cm.sum(), 1.0)
        criteria_rmse = float(np.sqrt(se.sum() / denom))

        metrics["criteria_rmse"] = criteria_rmse

        return metrics

    def train(self, train_loader, val_loader, epochs=10, patience=3):
        best_val_rmse = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader, split_name="val")

            self.history["train_loss"].append(train_loss)
            self.history["val_rmse"].append(val_metrics["rmse"])
            self.history["val_mae"].append(val_metrics["mae"])
            self.history["val_criteria_rmse"].append(val_metrics["criteria_rmse"])

            logger.info(
                f"Epoch {epoch} | "
                f"train_loss={train_loss:.4f} | "
                f"val_rmse={val_metrics['rmse']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f} | "
                f"val_criteria_rmse={val_metrics['criteria_rmse']:.4f}"
            )

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_rmse": val_metrics["rmse"],
                        "val_mae": val_metrics["mae"],
                        "val_criteria_rmse": val_metrics["criteria_rmse"],
                    },
                    self.save_dir / "best_mcmf.pt",
                )

                logger.info(f"Saved best MCMF model: RMSE={best_val_rmse:.4f}")

            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

        return self.history


class ParquetDeepFMDataset(Dataset):
    """
    Dataset for DeepFM baseline.
    """

    def __init__(self, parquet_path):
        self.parquet_path = Path(parquet_path)

        cols = [
            "user_idx",
            "item_idx",
            "rating",
            "item_quality",
            "item_value",
            "item_design",
            "item_usability",
            "item_durability",
            "mask_quality",
            "mask_value",
            "mask_design",
            "mask_usability",
            "mask_durability",
            "support_quality",
            "support_value",
            "support_design",
            "support_usability",
            "support_durability",
            "item_support",
        ]

        df = pd.read_parquet(self.parquet_path, columns=cols)

        self.user_idx = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.item_idx = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.rating = torch.tensor(df["rating"].values, dtype=torch.float32)

        # dense_cols = [
        #     "item_quality",
        #     "item_value",
        #     "item_design",
        #     "item_usability",
        #     "item_durability",
        #     "mask_quality",
        #     "mask_value",
        #     "mask_design",
        #     "mask_usability",
        #     "mask_durability",
        #     "support_quality",
        #     "support_value",
        #     "support_design",
        #     "support_usability",
        #     "support_durability",
        #     "item_support",
        # ]

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

        df[criteria_cols] = (df[criteria_cols].astype("float32") - 3.0) / 2.0
        df[mask_cols] = df[mask_cols].astype("float32")

        self.dense_features = torch.tensor(
            df[dense_cols].values,
            dtype=torch.float32,
        )

        del df

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        return {
            "user_idx": self.user_idx[idx],
            "item_idx": self.item_idx[idx],
            "dense_features": self.dense_features[idx],
            "rating": self.rating[idx],
        }


class DeepFMTrainer:
    def __init__(
        self,
        model,
        device,
        lr=1e-3,
        weight_decay=1e-5,
        save_dir="models/deepfm_baseline",
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.loss_fn = torch.nn.MSELoss()

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_rmse": [],
            "val_mae": [],
        }

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()

        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")

        for batch in pbar:
            user_idx = batch["user_idx"].to(self.device)
            item_idx = batch["item_idx"].to(self.device)
            dense_features = batch["dense_features"].to(self.device)
            rating = batch["rating"].to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(user_idx, item_idx, dense_features)
            loss = self.loss_fn(pred, rating)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self, data_loader, split_name="val"):
        self.model.eval()

        all_preds = []
        all_targets = []

        for batch in tqdm(data_loader, desc=f"[{split_name}]"):
            user_idx = batch["user_idx"].to(self.device)
            item_idx = batch["item_idx"].to(self.device)
            dense_features = batch["dense_features"].to(self.device)
            rating = batch["rating"].to(self.device)

            pred = self.model(user_idx, item_idx, dense_features)

            all_preds.append(pred.detach().cpu().numpy())
            all_targets.append(rating.detach().cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        return rating_metrics(all_preds, all_targets)

    def train(self, train_loader, val_loader, epochs=10, patience=3):
        best_val_rmse = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader, split_name="val")

            self.history["train_loss"].append(train_loss)
            self.history["val_rmse"].append(val_metrics["rmse"])
            self.history["val_mae"].append(val_metrics["mae"])

            logger.info(
                f"Epoch {epoch} | "
                f"train_loss={train_loss:.4f} | "
                f"val_rmse={val_metrics['rmse']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f}"
            )

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_rmse": val_metrics["rmse"],
                        "val_mae": val_metrics["mae"],
                    },
                    self.save_dir / "best_deepfm.pt",
                )

                logger.info(f"Saved best DeepFM model: RMSE={best_val_rmse:.4f}")

            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break
        return self.history


class MCDCFTrainer:
    def __init__(
        self,
        model,
        device,
        lr=1e-3,
        weight_decay=1e-5,
        lambda_overall=1.0,
        lambda_criteria=0.3,
        save_dir="models/mcdcf",
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        from src.models.mcf import MCDCFLoss

        self.loss_fn = MCDCFLoss(
            lambda_overall=lambda_overall,
            lambda_criteria=lambda_criteria,
        )

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_rmse": [],
            "val_mae": [],
            "val_criteria_rmse": [],
        }

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()

        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")

        for batch in pbar:
            user_idx = batch["user_idx"].to(self.device)
            item_idx = batch["item_idx"].to(self.device)
            rating = batch["rating"].to(self.device)
            criteria = batch["criteria"].to(self.device)
            criteria_mask = batch["criteria_mask"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(user_idx, item_idx)

            loss, loss_dict = self.loss_fn(
                outputs=outputs,
                rating=rating,
                criteria_targets=criteria,
                criteria_masks=criteria_mask,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss_dict['total']:.4f}",
                    "overall": f"{loss_dict['overall']:.4f}",
                    "criteria": f"{loss_dict['criteria']:.4f}",
                }
            )

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self, data_loader, split_name="val"):
        self.model.eval()

        all_preds = []
        all_targets = []

        criteria_preds_all = []
        criteria_targets_all = []
        criteria_masks_all = []

        attention_all = []

        for batch in tqdm(data_loader, desc=f"[{split_name}]"):
            user_idx = batch["user_idx"].to(self.device)
            item_idx = batch["item_idx"].to(self.device)
            rating = batch["rating"].to(self.device)
            criteria = batch["criteria"].to(self.device)
            criteria_mask = batch["criteria_mask"].to(self.device)

            outputs = self.model(user_idx, item_idx)

            all_preds.append(outputs["overall"].detach().cpu().numpy())
            all_targets.append(rating.detach().cpu().numpy())

            criteria_preds_all.append(outputs["criteria"].detach().cpu().numpy())
            criteria_targets_all.append(criteria.detach().cpu().numpy())
            criteria_masks_all.append(criteria_mask.detach().cpu().numpy())

            attention_all.append(outputs["attention"].detach().cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        metrics = rating_metrics(all_preds, all_targets)

        cp = np.concatenate(criteria_preds_all)
        ct = np.concatenate(criteria_targets_all)
        cm = np.concatenate(criteria_masks_all)

        se = ((cp - ct) ** 2) * cm
        denom = np.maximum(cm.sum(), 1.0)

        metrics["criteria_rmse"] = float(np.sqrt(se.sum() / denom))

        attention = np.concatenate(attention_all)
        mean_attention = attention.mean(axis=0)

        criteria_names = [
            "quality",
            "value",
            "design",
            "usability",
            "durability",
        ]

        for name, value in zip(criteria_names, mean_attention):
            metrics[f"attn_{name}"] = float(value)

        return metrics

    def train(self, train_loader, val_loader, epochs=10, patience=3):
        best_val_rmse = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader, split_name="val")

            self.history["train_loss"].append(train_loss)
            self.history["val_rmse"].append(val_metrics["rmse"])
            self.history["val_mae"].append(val_metrics["mae"])
            self.history["val_criteria_rmse"].append(val_metrics["criteria_rmse"])

            logger.info(
                f"Epoch {epoch} | "
                f"train_loss={train_loss:.4f} | "
                f"val_rmse={val_metrics['rmse']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f} | "
                f"val_criteria_rmse={val_metrics['criteria_rmse']:.4f} | "
                f"attn=["
                f"q={val_metrics['attn_quality']:.3f}, "
                f"v={val_metrics['attn_value']:.3f}, "
                f"d={val_metrics['attn_design']:.3f}, "
                f"u={val_metrics['attn_usability']:.3f}, "
                f"dur={val_metrics['attn_durability']:.3f}]"
            )

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_rmse": val_metrics["rmse"],
                        "val_mae": val_metrics["mae"],
                        "val_criteria_rmse": val_metrics["criteria_rmse"],
                        "attention": {
                            "quality": val_metrics["attn_quality"],
                            "value": val_metrics["attn_value"],
                            "design": val_metrics["attn_design"],
                            "usability": val_metrics["attn_usability"],
                            "durability": val_metrics["attn_durability"],
                        },
                    },
                    self.save_dir / "best_mcdcf.pt",
                )

                logger.info(f"Saved best MC-DCF model: RMSE={best_val_rmse:.4f}")

            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

        return self.history


class ParquetCriteriaResidualNeuMFDataset(Dataset):
    """
    Dataset for Criteria-Residual NeuMF.

    Required:
        user_idx, item_idx, rating
        dense_features: criteria + mask + support
        criteria targets
        criteria masks
    """

    def __init__(self, parquet_path):
        self.parquet_path = Path(parquet_path)

        cols = [
            "user_idx",
            "item_idx",
            "rating",
            "item_quality",
            "item_value",
            "item_design",
            "item_usability",
            "item_durability",
            "mask_quality",
            "mask_value",
            "mask_design",
            "mask_usability",
            "mask_durability",
            # "support_quality",
            # "support_value",
            # "support_design",
            # "support_usability",
            # "support_durability",
            # "item_support",
        ]

        df = pd.read_parquet(self.parquet_path, columns=cols)

        self.user_idx = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.item_idx = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.rating = torch.tensor(df["rating"].values, dtype=torch.float32)

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

        # dense_cols = [
        #     "item_quality",
        #     "item_value",
        #     "item_design",
        #     "item_usability",
        #     "item_durability",
        #     "mask_quality",
        #     "mask_value",
        #     "mask_design",
        #     "mask_usability",
        #     "mask_durability",
        #     "support_quality",
        #     "support_value",
        #     "support_design",
        #     "support_usability",
        #     "support_durability",
        #     "item_support",
        # ]
        dense_cols = criteria_cols + mask_cols

        # df[criteria_cols] = (df[criteria_cols].astype("float32") - 3.0) / 2.0
        df[mask_cols] = df[mask_cols].astype("float32")

        ###################################

        # self.criteria = torch.tensor(
        #     df[criteria_cols].values,
        #     dtype=torch.float32,
        # )

        # self.criteria_mask = torch.tensor(
        #     df[mask_cols].values,
        #     dtype=torch.float32,
        # )

        # self.dense_features = torch.tensor(
        #     df[dense_cols].values,
        #     dtype=torch.float32,
        # )

        # del df
        #######################################

        # criteria target giữ bản gốc 1–5
        self.criteria = torch.tensor(
            df[criteria_cols].values.astype("float32"),
            dtype=torch.float32,
        )

        self.criteria_mask = torch.tensor(
            df[mask_cols].values.astype("float32"),
            dtype=torch.float32,
        )

        # chỉ normalize cho dense_features
        df_dense = df[dense_cols].copy()
        df_dense[criteria_cols] = (
            df_dense[criteria_cols].astype("float32") - 3.0
        ) / 2.0
        df_dense[mask_cols] = df_dense[mask_cols].astype("float32")

        self.dense_features = torch.tensor(
            df_dense[dense_cols].values,
            dtype=torch.float32,
        )
        del df, df_dense

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        return {
            "user_idx": self.user_idx[idx],
            "item_idx": self.item_idx[idx],
            "rating": self.rating[idx],
            "criteria": self.criteria[idx],
            "criteria_mask": self.criteria_mask[idx],
            "dense_features": self.dense_features[idx],
        }


class CriteriaResidualNeuMFTrainer:
    def __init__(
        self,
        model,
        device,
        lr=5e-4,
        weight_decay=1e-5,
        lambda_overall=1.0,
        lambda_criteria=0.01,
        save_dir="models/cr_neumf_baseline",
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        from src.models.mcf import CriteriaResidualNeuMFLoss

        self.loss_fn = CriteriaResidualNeuMFLoss(
            lambda_overall=lambda_overall,
            lambda_criteria=lambda_criteria,
        )

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_rmse": [],
            "val_mae": [],
            "val_criteria_rmse": [],
        }

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()

        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")

        for batch in pbar:
            user_idx = batch["user_idx"].to(self.device)
            item_idx = batch["item_idx"].to(self.device)
            dense_features = batch["dense_features"].to(self.device)
            rating = batch["rating"].to(self.device)
            criteria = batch["criteria"].to(self.device)
            criteria_mask = batch["criteria_mask"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                user_idx=user_idx,
                item_idx=item_idx,
                dense_features=dense_features,
            )

            loss, loss_dict = self.loss_fn(
                outputs=outputs,
                rating=rating,
                criteria_targets=criteria,
                criteria_masks=criteria_mask,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            alpha_value = outputs["criteria_alpha"].detach().cpu().item()

            pbar.set_postfix(
                {
                    "loss": f"{loss_dict['total']:.4f}",
                    "overall": f"{loss_dict['overall']:.4f}",
                    "criteria": f"{loss_dict['criteria']:.4f}",
                    "alpha": f"{alpha_value:.4f}",
                }
            )

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self, data_loader, split_name="val"):
        self.model.eval()

        all_preds = []
        all_targets = []

        criteria_preds_all = []
        criteria_targets_all = []
        criteria_masks_all = []

        for batch in tqdm(data_loader, desc=f"[{split_name}]"):
            user_idx = batch["user_idx"].to(self.device)
            item_idx = batch["item_idx"].to(self.device)
            dense_features = batch["dense_features"].to(self.device)
            rating = batch["rating"].to(self.device)
            criteria = batch["criteria"].to(self.device)
            criteria_mask = batch["criteria_mask"].to(self.device)

            outputs = self.model(
                user_idx=user_idx,
                item_idx=item_idx,
                dense_features=dense_features,
            )

            all_preds.append(outputs["overall"].detach().cpu().numpy())
            all_targets.append(rating.detach().cpu().numpy())

            criteria_preds_all.append(outputs["criteria"].detach().cpu().numpy())
            criteria_targets_all.append(criteria.detach().cpu().numpy())
            criteria_masks_all.append(criteria_mask.detach().cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        metrics = rating_metrics(all_preds, all_targets)

        cp = np.concatenate(criteria_preds_all)
        ct = np.concatenate(criteria_targets_all)
        cm = np.concatenate(criteria_masks_all)

        se = ((cp - ct) ** 2) * cm
        denom = np.maximum(cm.sum(), 1.0)

        metrics["criteria_rmse"] = float(np.sqrt(se.sum() / denom))
        # metrics["criteria_alpha"] = float(
        #     self.model.criteria_alpha.detach().cpu().item()
        # )
        # alpha = 0.1 * torch.sigmoid(self.model.criteria_alpha_raw)
        alpha = 0.02 * torch.sigmoid(self.model.criteria_alpha_raw)

        metrics["criteria_alpha"] = float(alpha.detach().cpu().item())

        return metrics

    def train(self, train_loader, val_loader, epochs=10, patience=3):
        best_val_rmse = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader, split_name="val")

            self.history["train_loss"].append(train_loss)
            self.history["val_rmse"].append(val_metrics["rmse"])
            self.history["val_mae"].append(val_metrics["mae"])
            self.history["val_criteria_rmse"].append(val_metrics["criteria_rmse"])

            logger.info(
                f"Epoch {epoch} | "
                f"train_loss={train_loss:.4f} | "
                f"val_rmse={val_metrics['rmse']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f} | "
                f"val_criteria_rmse={val_metrics['criteria_rmse']:.4f} | "
                f"alpha={val_metrics['criteria_alpha']:.4f}"
            )

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                patience_counter = 0

                # torch.save(
                #     {
                #         "epoch": epoch,
                #         "model_state_dict": self.model.state_dict(),
                #         "optimizer_state_dict": self.optimizer.state_dict(),
                #         "val_rmse": val_metrics["rmse"],
                #         "val_mae": val_metrics["mae"],
                #         "val_criteria_rmse": val_metrics["criteria_rmse"],
                #         "criteria_alpha": val_metrics["criteria_alpha"],
                #     },
                #     self.save_dir / "best_cr_neumf.pt",
                # )
                save_path = self.save_dir / "best_cr_neumf.pt"
                tmp_path = str(save_path) + ".tmp"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_rmse": val_metrics["rmse"],
                        "val_mae": val_metrics["mae"],
                        "val_criteria_rmse": val_metrics["criteria_rmse"],
                        "criteria_alpha": val_metrics["criteria_alpha"],
                    },
                    tmp_path,
                )
                os.replace(tmp_path, save_path)

                logger.info(
                    f"Saved best Criteria-Residual NeuMF model: "
                    f"RMSE={best_val_rmse:.4f}"
                )

            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

        return self.history


@torch.no_grad()
def predict_and_save(trainer, data_loader, save_path: str | Path, model_type: str):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    trainer.model.eval()

    all_preds = []
    all_targets = []

    for batch in tqdm(data_loader, desc=f"[predict] {save_path.name}"):
        user_idx = batch["user_idx"].to(trainer.device)
        item_idx = batch["item_idx"].to(trainer.device)
        rating = batch["rating"].to(trainer.device)

        if model_type in ["bias", "mf", "ncf"]:
            pred = trainer.model(user_idx, item_idx)

        elif model_type == "deepfm":
            dense_features = batch["dense_features"].to(trainer.device)
            pred = trainer.model(user_idx, item_idx, dense_features)

        elif model_type == "cr_neumf":
            dense_features = batch["dense_features"].to(trainer.device)
            outputs = trainer.model(user_idx, item_idx, dense_features)
            pred = outputs["overall"]

        elif model_type in ["mcmf", "mcdcf"]:
            outputs = trainer.model(user_idx, item_idx)
            pred = outputs["overall"]

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        all_preds.append(pred.detach().cpu().numpy())
        all_targets.append(rating.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    all_preds = np.clip(all_preds, 1.0, 5.0)

    np.savez(
        save_path,
        pred=all_preds,
        target=all_targets,
    )


# print("done")
