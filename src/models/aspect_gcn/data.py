from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SplitData:
    name: str
    users: np.ndarray
    items: np.ndarray
    ratings: np.ndarray
    positives_by_user: Dict[int, np.ndarray]


@dataclass
class DataBundle:
    n_users: int
    n_items: int
    train: SplitData
    val: SplitData
    test: SplitData
    val_full: Optional[SplitData]
    test_full: Optional[SplitData]
    train_user_pos: Dict[int, np.ndarray]
    item_features: np.ndarray
    item_feature_mask: np.ndarray
    item_popularity: np.ndarray
    warm_users: np.ndarray
    warm_items: np.ndarray
    cold_users: np.ndarray
    cold_items: np.ndarray
    feature_names: List[str]


# class BPRDataset(Dataset):
#     def __init__(
#         self,
#         users: np.ndarray,
#         pos_items: np.ndarray,
#         train_user_pos: Dict[int, np.ndarray],
#         n_items: int,
#         num_negatives: int = 1,
#         seed: int = 42,
#     ) -> None:
#         self.users = users.astype(np.int64)
#         self.pos_items = pos_items.astype(np.int64)
#         self.train_user_pos = train_user_pos
#         self.n_items = int(n_items)
#         self.num_negatives = int(num_negatives)
#         self.rng = np.random.default_rng(seed)

#     def __len__(self) -> int:
#         return len(self.users)

#     def _sample_negative(self, user: int) -> int:
#         positives = self.train_user_pos.get(int(user))
#         # Rejection sampling is usually fine for sparse implicit-feedback data.
#         while True:
#             neg = int(self.rng.integers(0, self.n_items))
#             if positives is None or neg not in positives:
#                 return neg


#     def __getitem__(self, idx: int):
#         user = int(self.users[idx])
#         pos = int(self.pos_items[idx])
#         neg = self._sample_negative(user)
#         return (
#             torch.tensor(user, dtype=torch.long),
#             torch.tensor(pos, dtype=torch.long),
#             torch.tensor(neg, dtype=torch.long),
#         )
class BPRDataset(Dataset):
    def __init__(
        self,
        users: np.ndarray,
        pos_items: np.ndarray,
        train_user_pos: Dict[int, np.ndarray],
        n_items: int,
        num_negatives: int = 1,
        seed: int = 42,
        negative_items: Optional[np.ndarray] = None,
    ) -> None:
        self.users = users.astype(np.int64)
        self.pos_items = pos_items.astype(np.int64)
        self.train_user_pos = train_user_pos
        self.n_items = int(n_items)
        self.num_negatives = int(num_negatives)
        self.rng = np.random.default_rng(seed)

        if negative_items is None:
            self.negative_items = np.arange(self.n_items, dtype=np.int64)
        else:
            self.negative_items = np.asarray(negative_items, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.users)

    def _sample_negative(self, user: int) -> int:
        positives = set(
            self.train_user_pos.get(int(user), np.array([], dtype=np.int64))
        )

        while True:
            neg = int(self.rng.choice(self.negative_items))
            if neg not in positives:
                return neg

    def __getitem__(self, idx: int):
        user = int(self.users[idx])
        pos = int(self.pos_items[idx])
        neg = self._sample_negative(user)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


def _read_parquet(
    path: Path, columns: Optional[List[str]] = None, max_rows: Optional[int] = None
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    df = pd.read_parquet(path, columns=columns)
    if max_rows is not None:
        df = df.head(int(max_rows)).copy()
    return df


def _safe_columns(path: Path) -> List[str]:
    return list(pd.read_parquet(path, columns=[]).columns)


def _available_columns(path: Path) -> List[str]:
    # pyarrow metadata would be cheaper, but this works reliably with pandas/pyarrow.
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).schema.names


def _select_columns(path: Path, needed: Sequence[str]) -> List[str]:
    available = set(_available_columns(path))
    return [c for c in needed if c in available]


def _build_split(
    name: str,
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    rating_col: str,
    positive_threshold: float,
) -> SplitData:
    # Existing val/test full files may contain NULL user_idx/item_idx for true cold rows.
    # This model needs integer ids. To evaluate true cold rows, first rebuild global
    # user/item maps over train+val+test or provide encoded cold ids.
    # df = df.dropna(subset=[user_col, item_col, rating_col]).copy()
    users = df[user_col].astype("int64").to_numpy(np.int64)
    items = df[item_col].astype("int64").to_numpy(np.int64)
    ratings = df[rating_col].astype("float32").to_numpy(np.float32)

    pos_mask = ratings >= positive_threshold
    pos_df = pd.DataFrame({"u": users[pos_mask], "i": items[pos_mask]})
    positives_by_user: Dict[int, np.ndarray] = {}
    if len(pos_df) > 0:
        for u, g in pos_df.groupby("u", sort=False):
            positives_by_user[int(u)] = g["i"].to_numpy(np.int64)

    return SplitData(
        name=name,
        users=users,
        items=items,
        ratings=ratings,
        positives_by_user=positives_by_user,
    )


def _dict_sets_to_arrays(d: Dict[int, set]) -> Dict[int, np.ndarray]:
    return {int(u): np.fromiter(items, dtype=np.int64) for u, items in d.items()}


def _make_train_user_pos(users: np.ndarray, items: np.ndarray) -> Dict[int, np.ndarray]:
    tmp: Dict[int, set] = {}
    for u, i in zip(users, items):
        tmp.setdefault(int(u), set()).add(int(i))
    return _dict_sets_to_arrays(tmp)


def _load_optional_split(
    path: Path,
    name: str,
    user_col: str,
    item_col: str,
    rating_col: str,
    positive_threshold: float,
    base_cols: List[str],
) -> Optional[SplitData]:
    if not path.exists():
        return None
    cols = _select_columns(path, base_cols)
    df = _read_parquet(path, columns=cols)
    return _build_split(name, df, user_col, item_col, rating_col, positive_threshold)


def infer_n_entities(
    dfs: List[pd.DataFrame], user_col: str, item_col: str
) -> Tuple[int, int]:
    max_u = 0
    max_i = 0
    for df in dfs:
        if len(df) == 0:
            continue
        valid = df.dropna(subset=[user_col, item_col])
        if len(valid) == 0:
            continue
        max_u = max(max_u, int(valid[user_col].max()))
        max_i = max(max_i, int(valid[item_col].max()))
    return max_u + 1, max_i + 1


def build_item_features(
    train_df: pd.DataFrame,
    all_dfs: List[pd.DataFrame],
    n_items: int,
    item_col: str,
    score_cols: Sequence[str],
    mask_cols: Sequence[str],
    support_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build one static semantic feature vector per item.

    Priority:
    1. Use train rows only to avoid future leakage.
    2. Aggregate repeated item rows by mean for scores/masks and max/mean for supports.
    3. Fill missing columns with 0.
    """
    df = train_df.copy()
    all_feature_cols: List[str] = []

    for c in score_cols:
        if c not in df.columns:
            df[c] = 0.0
        all_feature_cols.append(c)

    for c in mask_cols:
        if c not in df.columns:
            df[c] = 0.0
        all_feature_cols.append(c)

    for c in support_cols:
        if c not in df.columns:
            df[c] = 0.0
        # log1p support to prevent large-count domination.
        df[c] = np.log1p(df[c].astype(np.float32))
        all_feature_cols.append(c)

    if len(all_feature_cols) == 0:
        features = np.zeros((n_items, 1), dtype=np.float32)
        mask = np.zeros((n_items,), dtype=np.float32)
        return features, mask, ["dummy_feature"]

    agg = df.groupby(item_col, sort=False)[all_feature_cols].mean()
    features = np.zeros((n_items, len(all_feature_cols)), dtype=np.float32)
    valid_items = agg.index.to_numpy(np.int64)
    valid_items = valid_items[(valid_items >= 0) & (valid_items < n_items)]
    features[valid_items] = agg.loc[valid_items, all_feature_cols].to_numpy(np.float32)

    # Feature mask = 1 if item has any non-zero semantic signal.
    feature_mask = (np.abs(features).sum(axis=1) > 0).astype(np.float32)

    # Normalize non-zero feature columns using train item statistics only.
    nonzero_rows = feature_mask > 0
    if nonzero_rows.any():
        mean = features[nonzero_rows].mean(axis=0, keepdims=True)
        std = features[nonzero_rows].std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        features[nonzero_rows] = (features[nonzero_rows] - mean) / std
        features[~nonzero_rows] = 0.0

    return (
        features.astype(np.float32),
        feature_mask.astype(np.float32),
        all_feature_cols,
    )


########
def _get_ablation_cfg(cfg: dict) -> dict:
    """
    Return ablation flags for the current run.

    Expected cfg format:
      cfg["active_ablation"] = "lightgcn_only"
      cfg["ablations"][active_ablation] = {...}

    If no ablation is provided, default = use all features.
    """
    active_name = cfg.get("active_ablation")

    if active_name is None:
        return {
            "use_aspect_scores": True,
            "use_aspect_masks": True,
            "use_aspect_supports": True,
        }

    ablations = cfg.get("ablation", {})
    if active_name not in ablations:
        raise KeyError(
            f"active_ablation='{active_name}' not found in cfg['ablations']. "
            f"Available: {list(ablations.keys())}"
        )

    return ablations[active_name]


def _resolve_aspect_columns(cfg: dict) -> Tuple[List[str], List[str], List[str]]:
    """
    Select aspect feature columns according to the current ablation setting.
    """
    cols_cfg = cfg["columns"]
    ablation_cfg = _get_ablation_cfg(cfg)

    score_cols = (
        list(cols_cfg.get("aspect_scores", []))
        if ablation_cfg.get("use_aspect_scores", True)
        else []
    )

    mask_cols = (
        list(cols_cfg.get("aspect_masks", []))
        if ablation_cfg.get("use_aspect_masks", True)
        else []
    )

    support_cols = (
        list(cols_cfg.get("aspect_supports", []))
        if ablation_cfg.get("use_aspect_supports", True)
        else []
    )

    return score_cols, mask_cols, support_cols


#######


def load_data_bundle(cfg: dict) -> DataBundle:
    paths = cfg["paths"]
    cols_cfg = cfg["columns"]
    data_cfg = cfg["data"]

    root = Path("data/processed/model_ready")
    train_path = root / paths["train_file"]
    val_path = root / paths["val_file"]
    test_path = root / paths["test_file"]
    val_full_path = root / paths.get("val_full_file", "val_model.parquet")
    test_full_path = root / paths.get("test_full_file", "test_model.parquet")

    user_col = cols_cfg["user"]
    item_col = cols_cfg["item"]
    rating_col = cols_cfg["rating"]

    # score_cols = cols_cfg.get("aspect_scores", [])
    # mask_cols = cols_cfg.get("aspect_masks", [])
    # support_cols = cols_cfg.get("aspect_supports", [])
    ###
    score_cols, mask_cols, support_cols = _resolve_aspect_columns(cfg)
    active_ablation = cfg.get("active_ablation", "default_all_features")
    print(f"[Data] active_ablation = {active_ablation}")
    print(f"[Data] score_cols   = {score_cols}")
    print(f"[Data] mask_cols    = {mask_cols}")
    print(f"[Data] support_cols = {support_cols}")
    ####

    base_cols = [user_col, item_col, rating_col]
    feature_cols = list(score_cols) + list(mask_cols) + list(support_cols)
    needed_cols = base_cols + feature_cols

    max_train_rows = data_cfg.get("max_train_rows")
    train_cols = _select_columns(train_path, needed_cols)
    val_cols = _select_columns(val_path, base_cols + feature_cols)
    test_cols = _select_columns(test_path, base_cols + feature_cols)

    train_df = _read_parquet(train_path, columns=train_cols, max_rows=max_train_rows)
    val_df = _read_parquet(val_path, columns=val_cols)
    test_df = _read_parquet(test_path, columns=test_cols)

    extra_for_shape = [train_df, val_df, test_df]
    optional_dfs_for_shape = []
    for p in [val_full_path, test_full_path]:
        if p.exists():
            opt_cols = _select_columns(p, base_cols)
            optional_dfs_for_shape.append(_read_parquet(p, columns=opt_cols))
    extra_for_shape.extend(optional_dfs_for_shape)

    n_users, n_items = infer_n_entities(extra_for_shape, user_col, item_col)

    positive_threshold = float(data_cfg.get("positive_threshold", 4.0))
    train_split = _build_split(
        "train", train_df, user_col, item_col, rating_col, positive_threshold
    )
    val_split = _build_split(
        "val", val_df, user_col, item_col, rating_col, positive_threshold
    )
    test_split = _build_split(
        "test", test_df, user_col, item_col, rating_col, positive_threshold
    )

    val_full = _load_optional_split(
        val_full_path,
        "val_full",
        user_col,
        item_col,
        rating_col,
        positive_threshold,
        base_cols,
    )
    test_full = _load_optional_split(
        test_full_path,
        "test_full",
        user_col,
        item_col,
        rating_col,
        positive_threshold,
        base_cols,
    )

    train_user_pos = _make_train_user_pos(train_split.users, train_split.items)

    item_popularity = np.zeros(n_items, dtype=np.int64)
    for i in train_split.items:
        if 0 <= int(i) < n_items:
            item_popularity[int(i)] += 1

    item_features, item_feature_mask, feature_names = build_item_features(
        train_df=train_df,
        all_dfs=[train_df, val_df, test_df],
        n_items=n_items,
        item_col=item_col,
        score_cols=score_cols,
        mask_cols=mask_cols,
        support_cols=support_cols,
    )

    warm_users = np.unique(train_split.users)
    warm_items = np.unique(train_split.items)
    all_users = np.arange(n_users, dtype=np.int64)
    all_items = np.arange(n_items, dtype=np.int64)
    cold_users = np.setdiff1d(all_users, warm_users, assume_unique=False)
    cold_items = np.setdiff1d(all_items, warm_items, assume_unique=False)

    return DataBundle(
        n_users=n_users,
        n_items=n_items,
        train=train_split,
        val=val_split,
        test=test_split,
        val_full=val_full,
        test_full=test_full,
        train_user_pos=train_user_pos,
        item_features=item_features,
        item_feature_mask=item_feature_mask,
        item_popularity=item_popularity,
        warm_users=warm_users,
        warm_items=warm_items,
        cold_users=cold_users,
        cold_items=cold_items,
        feature_names=feature_names,
    )


print("donê")
