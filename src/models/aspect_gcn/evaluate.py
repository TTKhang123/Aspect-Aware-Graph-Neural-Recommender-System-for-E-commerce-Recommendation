from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .data import DataBundle, SplitData
from .metrics import (
    aggregate_metric_dicts,
    coverage_at_k,
    diversity_at_k,
    novelty_at_k,
    ranking_metrics_for_user,
)
from .model import LightGCNAspectRecommender


def _sample_eval_candidates(
    user: int,
    true_items: np.ndarray,
    n_items: int,
    train_seen: Dict[int, np.ndarray],
    num_negatives: int,
    rng: np.random.Generator,
    filter_seen: bool,
) -> np.ndarray:
    true_items = np.asarray(true_items, dtype=np.int64)
    true_set = set(int(i) for i in true_items)
    seen_set = (
        set(int(i) for i in train_seen.get(int(user), np.array([], dtype=np.int64)))
        if filter_seen
        else set()
    )
    blocked = true_set | seen_set
    negatives = []
    max_trials = max(num_negatives * 50, 1000)
    trials = 0
    while len(negatives) < num_negatives and trials < max_trials:
        item = int(rng.integers(0, n_items))
        if item not in blocked:
            negatives.append(item)
            blocked.add(item)
        trials += 1
    if len(negatives) < num_negatives:
        # Fallback: deterministic fill. This is slower only in dense edge cases.
        for item in range(n_items):
            if item not in blocked:
                negatives.append(item)
                blocked.add(item)
                if len(negatives) >= num_negatives:
                    break
    return np.concatenate([true_items, np.asarray(negatives, dtype=np.int64)])


def _user_group(user: int, items: np.ndarray, warm_users: set, warm_items: set) -> str:
    is_warm_user = int(user) in warm_users
    item_warm_ratio = (
        np.mean([int(i) in warm_items for i in items]) if len(items) else 0.0
    )
    if is_warm_user and item_warm_ratio == 1.0:
        return "warm_user_warm_item"
    if is_warm_user and item_warm_ratio < 1.0:
        return "warm_user_mixed_or_cold_item"
    if (not is_warm_user) and item_warm_ratio == 1.0:
        return "cold_user_warm_item"
    return "cold_user_mixed_or_cold_item"


@torch.no_grad()
def evaluate_split(
    model: LightGCNAspectRecommender,
    split: SplitData,
    bundle: DataBundle,
    norm_adj: torch.Tensor,
    item_features_t: torch.Tensor,
    item_feature_mask_t: torch.Tensor,
    device: torch.device,
    k_values: Sequence[int],
    max_eval_users: Optional[int] = 2000,
    num_negatives: int = 100,
    filter_seen_items: bool = True,
    seed: int = 42,
) -> Dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    users = list(split.positives_by_user.keys())
    if max_eval_users is not None and len(users) > int(max_eval_users):
        users = list(rng.choice(users, size=int(max_eval_users), replace=False))

    warm_users = set(int(u) for u in bundle.warm_users)
    warm_items = set(int(i) for i in bundle.warm_items)

    rows: List[Dict[str, float]] = []
    rows_by_group: Dict[str, List[Dict[str, float]]] = {}
    ranked_lists: List[np.ndarray] = []

    for user in tqdm(users, desc=f"eval {split.name}"):
        true_items = split.positives_by_user[int(user)]
        if len(true_items) == 0:
            continue
        candidates = _sample_eval_candidates(
            user=int(user),
            true_items=true_items,
            n_items=bundle.n_items,
            train_seen=bundle.train_user_pos,
            num_negatives=num_negatives,
            rng=rng,
            filter_seen=filter_seen_items,
        )

        u_t = torch.tensor([int(user)], dtype=torch.long, device=device)
        cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
        scores = model.full_sort_scores(
            users=u_t,
            candidate_items=cand_t,
            norm_adj=norm_adj,
            item_features=item_features_t,
            item_feature_mask=item_feature_mask_t,
        ).squeeze(0)
        order = torch.argsort(scores, descending=True).detach().cpu().numpy()
        ranked_items = candidates[order]
        ranked_lists.append(ranked_items)

        metric_row = ranking_metrics_for_user(
            ranked_items, set(int(i) for i in true_items), k_values
        )
        rows.append(metric_row)
        group = _user_group(int(user), true_items, warm_users, warm_items)
        rows_by_group.setdefault(group, []).append(metric_row)

    out = aggregate_metric_dicts(rows)
    for k in k_values:
        out[f"coverage@{k}"] = coverage_at_k(ranked_lists, bundle.n_items, k)
        out[f"novelty@{k}"] = novelty_at_k(
            ranked_lists, bundle.item_popularity, len(bundle.train.users), k
        )
        out[f"diversity@{k}"] = diversity_at_k(ranked_lists, bundle.item_features, k)

    for group, group_rows in rows_by_group.items():
        group_metrics = aggregate_metric_dicts(group_rows)
        for key, value in group_metrics.items():
            out[f"{group}/{key}"] = value

    out["n_eval_users"] = float(len(users))
    return out
