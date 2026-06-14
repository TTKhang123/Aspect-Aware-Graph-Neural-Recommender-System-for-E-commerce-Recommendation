from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    rel = relevance[:k].astype(np.float32)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float(np.sum(rel * discounts))


def ranking_metrics_for_user(
    ranked_items: np.ndarray, true_items: set, k_values: Sequence[int]
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if len(true_items) == 0:
        for k in k_values:
            out[f"recall@{k}"] = 0.0
            out[f"precision@{k}"] = 0.0
            out[f"hit@{k}"] = 0.0
            out[f"ndcg@{k}"] = 0.0
            out[f"mrr@{k}"] = 0.0
        return out

    true_count = len(true_items)
    for k in k_values:
        topk = ranked_items[:k]
        hits = np.array(
            [1 if int(i) in true_items else 0 for i in topk], dtype=np.float32
        )
        n_hits = float(hits.sum())
        out[f"recall@{k}"] = n_hits / max(true_count, 1)
        out[f"precision@{k}"] = n_hits / k
        out[f"hit@{k}"] = 1.0 if n_hits > 0 else 0.0

        ideal_hits = np.ones(min(true_count, k), dtype=np.float32)
        idcg = dcg_at_k(ideal_hits, k)
        out[f"ndcg@{k}"] = dcg_at_k(hits, k) / idcg if idcg > 0 else 0.0

        rr = 0.0
        for rank, h in enumerate(hits, start=1):
            if h > 0:
                rr = 1.0 / rank
                break
        out[f"mrr@{k}"] = rr
    return out


def coverage_at_k(all_ranked: List[np.ndarray], n_items: int, k: int) -> float:
    rec_items = set()
    for arr in all_ranked:
        rec_items.update(int(i) for i in arr[:k])
    return len(rec_items) / max(n_items, 1)


def novelty_at_k(
    all_ranked: List[np.ndarray],
    item_popularity: np.ndarray,
    n_train_interactions: int,
    k: int,
) -> float:
    vals = []
    denom = max(n_train_interactions, 1)
    for arr in all_ranked:
        for item in arr[:k]:
            pop = (
                int(item_popularity[int(item)])
                if int(item) < len(item_popularity)
                else 0
            )
            prob = max(pop / denom, 1e-12)
            vals.append(-np.log2(prob))
    return float(np.mean(vals)) if vals else 0.0


def diversity_at_k(
    all_ranked: List[np.ndarray], item_features: np.ndarray, k: int
) -> float:
    """Mean pairwise cosine distance among items in each top-K list."""
    if item_features is None or item_features.size == 0:
        return 0.0
    divs = []
    for arr in all_ranked:
        items = arr[:k]
        if len(items) < 2:
            continue
        feats = item_features[items]
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        valid = norms.squeeze(-1) > 1e-8
        feats = feats[valid]
        if len(feats) < 2:
            continue
        feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)
        sim = feats @ feats.T
        tri = sim[np.triu_indices_from(sim, k=1)]
        divs.append(float(np.mean(1.0 - tri)))
    return float(np.mean(divs)) if divs else 0.0


def aggregate_metric_dicts(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
