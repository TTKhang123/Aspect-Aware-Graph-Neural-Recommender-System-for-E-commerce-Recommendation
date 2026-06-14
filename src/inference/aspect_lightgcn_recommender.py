from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd
import torch

from src.models.aspect_gcn.data import load_data_bundle
from src.models.aspect_gcn.graph import build_normalized_adj
from src.models.aspect_gcn.model import LightGCNAspectRecommender
from src.models.aspect_gcn.utils import load_yaml, choose_device

from src.inference.content_lsh import RandomHyperplaneLSH

ASPECT_COLS = [
    "item_quality",
    "item_value",
    "item_design",
    "item_usability",
    "item_durability",
]

MASK_COLS = [
    "mask_quality",
    "mask_value",
    "mask_design",
    "mask_usability",
    "mask_durability",
]

SUPPORT_COLS = [
    "support_quality",
    "support_value",
    "support_design",
    "support_usability",
    "support_durability",
    "item_support",
]


@dataclass
class Recommendation:
    rank: int
    item_idx: int
    item_id: str
    score: float
    title: Optional[str]
    brand: Optional[str]
    aspects: Dict[str, float]
    explanation: str


class AspectLightGCNInference:
    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda",
        active_ablation: str = "lightgcn_aspects_masks_support",  ### change if u want
        metadata_path: Optional[str | Path] = None,
        lsh_index_path: Optional[str | Path] = None,
    ):
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = choose_device(device)

        self.lsh_index_path = Path(lsh_index_path) if lsh_index_path else None
        self.user_lsh = None

        if self.lsh_index_path is not None and self.lsh_index_path.exists():
            self.user_lsh = RandomHyperplaneLSH.load(self.lsh_index_path)

        root_cfg = load_yaml(self.config_path)
        self.cfg = root_cfg["aspect_lightgcn"]
        self.cfg["active_ablation"] = active_ablation

        self.model_ready_dir = Path(
            self.cfg["paths"].get("model_ready_dir", "data/processed/model_ready")
        )

        self.user_map_path = self.model_ready_dir / "user2idx.parquet"
        self.item_map_path = self.model_ready_dir / "item2idx.parquet"
        self.train_path = self.model_ready_dir / self.cfg["paths"].get(
            "train_file", "train_model.parquet"
        )

        self.metadata_path = Path(metadata_path) if metadata_path else None

        self.con = duckdb.connect()

        self._load_mappings()
        self._load_bundle()
        self._load_model()
        self._load_optional_metadata()

    def _load_mappings(self) -> None:
        user_map = pd.read_parquet(self.user_map_path)
        item_map = pd.read_parquet(self.item_map_path)

        self.user2idx = dict(zip(user_map["user_id"], user_map["user_idx"]))
        self.idx2user = dict(zip(user_map["user_idx"], user_map["user_id"]))

        self.item2idx = dict(zip(item_map["item_id"], item_map["item_idx"]))
        self.idx2item = dict(zip(item_map["item_idx"], item_map["item_id"]))

    def _load_bundle(self) -> None:
        self.bundle = load_data_bundle(self.cfg)

        self.norm_adj = build_normalized_adj(
            n_users=self.bundle.n_users,
            n_items=self.bundle.n_items,
            users=self.bundle.train.users,
            items=self.bundle.train.items,
            device=self.device,
        )

        self.item_features_t = torch.tensor(
            self.bundle.item_features,
            dtype=torch.float32,
            device=self.device,
        )

        self.item_feature_mask_t = torch.tensor(
            self.bundle.item_feature_mask,
            dtype=torch.float32,
            device=self.device,
        )

    def _load_model(self) -> None:
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)

        model_cfg = self.cfg["model"]

        self.model = LightGCNAspectRecommender(
            n_users=self.bundle.n_users,
            n_items=self.bundle.n_items,
            item_feature_dim=self.bundle.item_features.shape[1],
            embedding_dim=int(model_cfg.get("embedding_dim", 64)),
            n_layers=int(model_cfg.get("n_layers", 3)),
            aspect_hidden_dim=int(model_cfg.get("aspect_hidden_dim", 128)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            fusion=model_cfg.get("fusion", "gated_residual"),
            use_user_bias=bool(model_cfg.get("use_user_bias", True)),
            use_item_bias=bool(model_cfg.get("use_item_bias", True)),
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _load_optional_metadata(self) -> None:
        self.metadata = None

        if self.metadata_path is None or not self.metadata_path.exists():
            return

        try:
            meta = pd.read_parquet(self.metadata_path)
            if "item_id" in meta.columns:
                self.metadata = meta.set_index("item_id")
        except Exception:
            self.metadata = None

    def _get_item_metadata(self, item_id: str) -> Dict[str, Optional[str]]:
        if self.metadata is None or item_id not in self.metadata.index:
            return {
                "title": None,
                "brand": None,
            }

        row = self.metadata.loc[item_id]

        return {
            "title": (
                str(row["title"]) if "title" in row and pd.notna(row["title"]) else None
            ),
            "brand": (
                str(row["brand"]) if "brand" in row and pd.notna(row["brand"]) else None
            ),
        }

    def _get_seen_items(self, user_idx: int) -> set[int]:
        seen = self.bundle.train_user_pos.get(int(user_idx))
        if seen is None:
            return set()
        return set(int(i) for i in seen)

    def _get_item_aspects(self, item_idx: int) -> Dict[str, float]:
        train_path = str(self.train_path).replace("\\", "/")

        result = self.con.execute(f"""
            SELECT
                AVG(item_quality) AS item_quality,
                AVG(item_value) AS item_value,
                AVG(item_design) AS item_design,
                AVG(item_usability) AS item_usability,
                AVG(item_durability) AS item_durability
            FROM read_parquet('{train_path}')
            WHERE item_idx = {int(item_idx)}
            """).df()

        if len(result) == 0:
            return {c: 3.0 for c in ASPECT_COLS}

        row = result.iloc[0]

        return {c: float(row[c]) if pd.notna(row[c]) else 3.0 for c in ASPECT_COLS}

    def _get_user_aspect_profile(self, user_idx: int) -> Dict[str, float]:
        train_path = str(self.train_path).replace("\\", "/")

        result = self.con.execute(f"""
            SELECT
                AVG(item_quality) AS item_quality,
                AVG(item_value) AS item_value,
                AVG(item_design) AS item_design,
                AVG(item_usability) AS item_usability,
                AVG(item_durability) AS item_durability
            FROM read_parquet('{train_path}')
            WHERE user_idx = {int(user_idx)}
              AND rating >= 4.0
            """).df()

        if len(result) == 0:
            return {c: 3.0 for c in ASPECT_COLS}

        row = result.iloc[0]

        return {c: float(row[c]) if pd.notna(row[c]) else 3.0 for c in ASPECT_COLS}

    def _build_explanation(
        self,
        user_profile: Dict[str, float],
        item_aspects: Dict[str, float],
    ) -> str:
        diffs = []

        for c in ASPECT_COLS:
            aspect_name = c.replace("item_", "")
            item_val = item_aspects.get(c, 3.0)
            user_val = user_profile.get(c, 3.0)
            diffs.append((aspect_name, item_val, user_val, item_val - user_val))

        # Ưu tiên aspect cao nhất của item
        diffs = sorted(diffs, key=lambda x: x[1], reverse=True)
        top = diffs[:3]

        parts = []
        for name, item_val, user_val, _ in top:
            parts.append(f"{name}={item_val:.2f}")

        return "Recommended because this item has strong " + ", ".join(parts)

    def find_users_for_cold_item(
        self,
        quality: float,
        value: float,
        design: float,
        usability: float,
        durability: float,
        top_k_users: int = 50,
    ) -> pd.DataFrame:
        """
        Cold-start item routing:
        Tìm user có preference vector gần cold item vector nhất.
        """

        if self.user_lsh is None:
            raise ValueError(
                "LSH index is not loaded. "
                "Build it first with scripts/build_user_lsh_index.py "
                "and pass lsh_index_path to AspectLightGCNInference."
            )

        item_vec = np.array(
            [[quality, value, design, usability, durability]],
            dtype=np.float32,
        )

        # cùng preprocessing với user profile
        item_vec = (item_vec - 3.0) / 2.0

        return self.user_lsh.query(
            query_vector=item_vec.reshape(-1),
            top_k=top_k_users,
        )

    def recommend_popular_for_new_user(self, top_k: int = 10) -> pd.DataFrame:
        train_path = str(self.train_path).replace("\\", "/")

        return self.con.execute(f"""
            SELECT
                item_id,
                item_idx,
                COUNT(*) AS popularity,
                AVG(rating) AS avg_rating,
                AVG(item_quality) AS quality,
                AVG(item_value) AS value,
                AVG(item_design) AS design,
                AVG(item_usability) AS usability,
                AVG(item_durability) AS durability

            FROM read_parquet('{train_path}')

            GROUP BY item_id, item_idx

            HAVING COUNT(*) >= 5

            ORDER BY
                popularity DESC,
                avg_rating DESC

            LIMIT {int(top_k)}
            """).df()

    def recommend_by_criteria_for_new_user(self, top_k: int = 10) -> dict:
        train_path = str(self.train_path).replace("\\", "/")

        criteria = {
            "quality": "item_quality",
            "value": "item_value",
            "design": "item_design",
            "usability": "item_usability",
            "durability": "item_durability",
        }

        results = {}

        for name, col in criteria.items():
            df = self.con.execute(f"""
                SELECT
                    item_id,
                    item_idx,
                    COUNT(*) AS popularity,
                    AVG(rating) AS avg_rating,
                    AVG(item_quality) AS quality,
                    AVG(item_value) AS value,
                    AVG(item_design) AS design,
                    AVG(item_usability) AS usability,
                    AVG(item_durability) AS durability,
                    AVG({col}) AS criteria_score

                FROM read_parquet('{train_path}')

                GROUP BY item_id, item_idx

                HAVING COUNT(*) >= 5

                ORDER BY
                    criteria_score DESC,
                    avg_rating DESC,
                    popularity DESC

                LIMIT {int(top_k)}
                """).df()

            results[name] = df

        return results

    @torch.no_grad()
    def recommend_for_user(
        self,
        user_id: str,
        top_k: int = 10,
        filter_seen: bool = True,
        candidate_batch_size: int = 50000,
    ) -> List[Recommendation]:
        if user_id not in self.user2idx:
            # raise ValueError(
            #     f"Unknown user_id={user_id}. "
            #     "This model supports known users in user2idx.parquet. "
            #     "For new users, use fallback popularity or collect onboarding preferences."
            # )
            if user_id not in self.user2idx:
                return {
                    "user_status": "cold_user_not_in_mapping",
                    "popular": self.recommend_popular_for_new_user(top_k=top_k),
                    "by_criteria": self.recommend_by_criteria_for_new_user(top_k=top_k),
                }

        user_idx = int(self.user2idx[user_id])
        seen_items = self._get_seen_items(user_idx)

        candidate_items = torch.arange(
            self.bundle.n_items,
            dtype=torch.long,
            device=self.device,
        )

        user_t = torch.tensor([user_idx], dtype=torch.long, device=self.device)

        scores = self.model.full_sort_scores(
            users=user_t,
            candidate_items=candidate_items,
            norm_adj=self.norm_adj,
            item_features=self.item_features_t,
            item_feature_mask=self.item_feature_mask_t,
            batch_items=candidate_batch_size,
        ).squeeze(0)

        if filter_seen and len(seen_items) > 0:
            seen_t = torch.tensor(
                list(seen_items),
                dtype=torch.long,
                device=self.device,
            )
            scores[seen_t] = -1e9

        top_scores, top_indices = torch.topk(scores, k=top_k)

        user_profile = self._get_user_aspect_profile(user_idx)

        recommendations = []

        for rank, (score, item_idx_t) in enumerate(
            zip(top_scores.detach().cpu(), top_indices.detach().cpu()),
            start=1,
        ):
            item_idx = int(item_idx_t)
            item_id = str(self.idx2item[item_idx])

            meta = self._get_item_metadata(item_id)
            aspects = self._get_item_aspects(item_idx)
            explanation = self._build_explanation(user_profile, aspects)

            recommendations.append(
                Recommendation(
                    rank=rank,
                    item_idx=item_idx,
                    item_id=item_id,
                    score=float(score),
                    title=meta["title"],
                    brand=meta["brand"],
                    aspects=aspects,
                    explanation=explanation,
                )
            )

        return recommendations

    def sample_known_users(self, n: int = 20) -> pd.DataFrame:
        train_path = str(self.train_path).replace("\\", "/")

        return self.con.execute(f"""
            SELECT
                user_id,
                user_idx,
                COUNT(*) AS train_interactions,
                AVG(rating) AS avg_rating
            FROM read_parquet('{train_path}')
            GROUP BY user_id, user_idx
            ORDER BY train_interactions DESC
            LIMIT {int(n)}
            """).df()

    # def sample_known_users(self, n: int = 20) -> pd.DataFrame:
    #     users = list(self.user2idx.keys())

    #     if len(users) == 0:
    #         return pd.DataFrame(columns=["user_id"])

    #     rng = np.random.default_rng(42)

    #     sampled = rng.choice(
    #         users,
    #         size=min(n, len(users)),
    #         replace=False,
    #     )

    #     return pd.DataFrame(
    #         {
    #             "user_id": sampled,
    #         }
    #     )
