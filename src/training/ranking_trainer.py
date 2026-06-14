import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset


class BPRInteractionDataset(Dataset):
    def __init__(self, parquet_path, n_items, positive_threshold=4.0, seed=42):
        df = pd.read_parquet(
            parquet_path,
            columns=["user_idx", "item_idx", "rating"],
        )

        df = df[df["rating"] >= positive_threshold].copy()

        self.users = df["user_idx"].astype("int64").to_numpy()
        self.pos_items = df["item_idx"].astype("int64").to_numpy()
        self.n_items = int(n_items)

        self.user_pos = {}
        for u, i in zip(self.users, self.pos_items):
            self.user_pos.setdefault(int(u), set()).add(int(i))

        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.users)

    def sample_negative(self, user):
        positives = self.user_pos.get(int(user), set())
        while True:
            neg = int(self.rng.integers(0, self.n_items))
            if neg not in positives:
                return neg

    def __getitem__(self, idx):
        u = int(self.users[idx])
        pos = int(self.pos_items[idx])
        neg = self.sample_negative(u)

        return {
            "user_idx": torch.tensor(u, dtype=torch.long),
            "pos_item_idx": torch.tensor(pos, dtype=torch.long),
            "neg_item_idx": torch.tensor(neg, dtype=torch.long),
        }


class BPRDeepFMDataset(Dataset):
    def __init__(self, parquet_path, n_items, positive_threshold=4.0, seed=42):
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

        df = pd.read_parquet(parquet_path, columns=cols)

        df = df.dropna(subset=["user_idx", "item_idx", "rating"]).copy()

        df[criteria_cols] = (df[criteria_cols].astype("float32") - 3.0) / 2.0
        df[mask_cols] = df[mask_cols].astype("float32")

        # Build item-level dense feature matrix from ALL rows in train file.
        # This is needed so negative items also have correct dense features.
        item_dense_df = df.groupby("item_idx")[dense_cols].mean().astype("float32")

        self.n_items = int(n_items)

        self.item_dense_matrix = np.zeros(
            (self.n_items, len(dense_cols)),
            dtype=np.float32,
        )

        valid_items = item_dense_df.index.to_numpy(np.int64)
        valid_items = valid_items[(valid_items >= 0) & (valid_items < self.n_items)]

        self.item_dense_matrix[valid_items] = item_dense_df.loc[
            valid_items,
            dense_cols,
        ].to_numpy(dtype=np.float32)

        # BPR training uses only positive interactions.
        pos_df = df[df["rating"] >= positive_threshold].copy()

        self.users = pos_df["user_idx"].astype("int64").to_numpy()
        self.pos_items = pos_df["item_idx"].astype("int64").to_numpy()

        self.user_pos = {}
        for u, i in zip(self.users, self.pos_items):
            self.user_pos.setdefault(int(u), set()).add(int(i))

        self.rng = np.random.default_rng(seed)

        del df, pos_df, item_dense_df

    def __len__(self):
        return len(self.users)

    def sample_negative(self, user):
        positives = self.user_pos.get(int(user), set())

        while True:
            neg = int(self.rng.integers(0, self.n_items))
            if neg not in positives:
                return neg

    def __getitem__(self, idx):
        u = int(self.users[idx])
        pos = int(self.pos_items[idx])
        neg = self.sample_negative(u)

        pos_dense = self.item_dense_matrix[pos]
        neg_dense = self.item_dense_matrix[neg]

        return {
            "user_idx": torch.tensor(u, dtype=torch.long),
            "pos_item_idx": torch.tensor(pos, dtype=torch.long),
            "neg_item_idx": torch.tensor(neg, dtype=torch.long),
            "pos_dense": torch.tensor(pos_dense, dtype=torch.float32),
            "neg_dense": torch.tensor(neg_dense, dtype=torch.float32),
        }


def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))


def train_one_epoch_bpr(model, loader, optimizer, device, grad_clip=5.0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        users = batch["user_idx"].to(device)
        pos = batch["pos_item_idx"].to(device)
        neg = batch["neg_item_idx"].to(device)

        pos_scores = model(users, pos)
        neg_scores = model(users, neg)

        loss = bpr_loss(pos_scores, neg_scores)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_one_epoch_deepfm_bpr(model, loader, optimizer, device, grad_clip=5.0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        users = batch["user_idx"].to(device)
        pos = batch["pos_item_idx"].to(device)
        neg = batch["neg_item_idx"].to(device)

        pos_dense = batch["pos_dense"].to(device)
        neg_dense = batch["neg_dense"].to(device)

        pos_scores = model(users, pos, pos_dense)
        neg_scores = model(users, neg, neg_dense)

        loss = bpr_loss(pos_scores, neg_scores)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    return total_loss / max(n_batches, 1)
