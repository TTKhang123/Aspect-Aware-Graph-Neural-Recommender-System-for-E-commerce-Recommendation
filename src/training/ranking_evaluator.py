from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# def load_eval_data(
#     parquet_path: str | Path,
#     positive_threshold: float = 4.0,
# ) -> Dict[int, np.ndarray]:
#     df = pd.read_parquet(
#         parquet_path,
#         columns=["user_idx", "item_idx", "rating"],
#     )

#     df = df.dropna(subset=["user_idx", "item_idx", "rating"]).copy()
#     df = df[df["rating"] >= positive_threshold].copy()

#     positives_by_user = {}

#     for u, g in df.groupby("user_idx", sort=False):
#         positives_by_user[int(u)] = g["item_idx"].astype("int64").to_numpy()

#     return positives_by_user


def load_eval_data(
    parquet_path: str | Path,
    positive_threshold: float = 4.0,
) -> Dict[int, dict]:
    cols = [
        "user_idx",
        "item_idx",
        "rating",
        "is_train_user",
        "is_train_item",
        "is_warm_row",
    ]

    df = pd.read_parquet(parquet_path, columns=cols)
    df = df.dropna(subset=["user_idx", "item_idx", "rating"]).copy()
    df = df[df["rating"] >= positive_threshold].copy()

    out = {}

    for u, g in df.groupby("user_idx", sort=False):
        out[int(u)] = {
            "items": g["item_idx"].astype("int64").to_numpy(),
            "is_train_user": int(g["is_train_user"].iloc[0]),
            "item_is_train": g["is_train_item"].astype("int64").to_numpy(),
        }

    return out


def load_train_user_items(parquet_path: str | Path) -> Dict[int, set]:
    df = pd.read_parquet(
        parquet_path,
        columns=["user_idx", "item_idx"],
    )

    df = df.dropna(subset=["user_idx", "item_idx"]).copy()

    user_items = defaultdict(set)

    for u, i in zip(df["user_idx"].values, df["item_idx"].values):
        user_items[int(u)].add(int(i))

    return dict(user_items)


def hit_at_k(ranked_items: np.ndarray, positives: set, k: int) -> float:
    return float(any(i in positives for i in ranked_items[:k]))


def precision_at_k(ranked_items: np.ndarray, positives: set, k: int) -> float:
    if k <= 0:
        return 0.0
    return float(sum(i in positives for i in ranked_items[:k]) / k)


def recall_at_k(ranked_items: np.ndarray, positives: set, k: int) -> float:
    if len(positives) == 0:
        return 0.0
    return float(sum(i in positives for i in ranked_items[:k]) / len(positives))


def mrr_at_k(ranked_items: np.ndarray, positives: set, k: int) -> float:
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in positives:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_items: np.ndarray, positives: set, k: int) -> float:
    dcg = 0.0

    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in positives:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(positives), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return float(dcg / idcg)


@torch.no_grad()
def evaluate_ranking_model(
    model: torch.nn.Module,
    model_type: str,
    eval_positives_by_user: Dict[int, np.ndarray],
    train_user_items: Dict[int, set],
    n_items: int,
    device: torch.device,
    k_values: List[int],
    num_negatives: int = 100,
    max_eval_users: Optional[int] = 2000,
    filter_seen_items: bool = True,
    seed: int = 42,
    item_dense_matrix: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed)

    users = list(eval_positives_by_user.keys())

    # if max_eval_users is not None:
    #     users = users[: int(max_eval_users)]
    if max_eval_users is not None and len(users) > max_eval_users:
        rng = np.random.default_rng(seed)
        users = rng.choice(
            users,
            size=int(max_eval_users),
            replace=False,
        ).tolist()

    metrics = defaultdict(list)
    recommended_items_by_k = {k: [] for k in k_values}
    metrics_by_group = defaultdict(lambda: defaultdict(list))

    for user in tqdm(users, desc="eval ranking"):
        # positives = set(int(i) for i in eval_positives_by_user[user])
        user_record = eval_positives_by_user[user]
        positive_items_arr = user_record["items"]
        positives = set(int(i) for i in positive_items_arr)

        is_train_user = bool(user_record["is_train_user"])
        item_train_flags = user_record["item_is_train"]

        if is_train_user and item_train_flags.mean() == 1.0:
            group = "warm_user_warm_item"

        elif is_train_user and item_train_flags.mean() < 1.0:
            group = "warm_user_cold_or_mixed_item"

        elif (not is_train_user) and item_train_flags.mean() == 1.0:
            group = "cold_user_warm_item"

        else:
            group = "cold_user_cold_or_mixed_item"

        if len(positives) == 0:
            continue

        seen = train_user_items.get(int(user), set()) if filter_seen_items else set()

        candidate_items = set(positives)

        while len(candidate_items) < len(positives) + num_negatives:
            neg = int(rng.integers(0, n_items))
            if neg in candidate_items:
                continue
            if neg in seen:
                continue
            candidate_items.add(neg)

        candidate_items = np.array(list(candidate_items), dtype=np.int64)

        users_t = torch.full(
            (len(candidate_items),),
            int(user),
            dtype=torch.long,
            device=device,
        )
        items_t = torch.tensor(candidate_items, dtype=torch.long, device=device)

        if model_type == "deepfm":
            if item_dense_matrix is None:
                raise ValueError(
                    "item_dense_matrix is required for DeepFM ranking eval."
                )

            dense_np = item_dense_matrix[candidate_items]
            dense_t = torch.tensor(dense_np, dtype=torch.float32, device=device)
            scores = model(users_t, items_t, dense_t)
        else:
            scores = model(users_t, items_t)

        scores_np = scores.detach().cpu().numpy()
        ranked_idx = np.argsort(-scores_np)
        ranked_items = candidate_items[ranked_idx]

        for k in k_values:

            hit = hit_at_k(ranked_items, positives, k)
            precision = precision_at_k(ranked_items, positives, k)
            recall = recall_at_k(ranked_items, positives, k)
            mrr = mrr_at_k(ranked_items, positives, k)
            ndcg = ndcg_at_k(ranked_items, positives, k)

            metrics[f"hit@{k}"].append(hit)
            metrics[f"precision@{k}"].append(precision)
            metrics[f"recall@{k}"].append(recall)
            metrics[f"mrr@{k}"].append(mrr)
            metrics[f"ndcg@{k}"].append(ndcg)

            metrics_by_group[group][f"hit@{k}"].append(hit)
            metrics_by_group[group][f"precision@{k}"].append(precision)
            metrics_by_group[group][f"recall@{k}"].append(recall)
            metrics_by_group[group][f"mrr@{k}"].append(mrr)
            metrics_by_group[group][f"ndcg@{k}"].append(ndcg)

            recommended_items_by_k[k].extend(ranked_items[:k].tolist())

    final = {}

    for key, values in metrics.items():
        final[key] = float(np.mean(values)) if values else 0.0

    ####
    final["groups"] = {}
    for group_name, group_metrics in metrics_by_group.items():
        final["groups"][group_name] = {}
        for metric_name, values in group_metrics.items():

            final["groups"][group_name][metric_name] = (
                float(np.mean(values)) if len(values) > 0 else 0.0
            )

    for k in k_values:
        rec_items = recommended_items_by_k[k]
        final[f"coverage@{k}"] = (
            float(len(set(rec_items)) / n_items) if len(rec_items) > 0 else 0.0
        )

    final["n_eval_users"] = float(len(users))

    return final
