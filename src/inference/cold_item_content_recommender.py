from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.inference.content_profile import build_item_content_text
from src.inference.content_lsh import RandomHyperplaneLSH


class ColdItemContentRecommender:
    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)

        self.meta = joblib.load(self.index_dir / "content_lsh_metadata.joblib")
        self.encoder = SentenceTransformer(self.meta["embedding_model"])
        self.index = RandomHyperplaneLSH.load(self.index_dir / "user_content_lsh.pkl")

    def encode_new_item(self, item_metadata: Dict[str, Any]) -> np.ndarray:
        text = build_item_content_text(item_metadata)

        emb = self.encoder.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        return emb[0]

    def recommend_users_for_new_item(
        self,
        item_metadata: Dict[str, Any],
        top_k_users: int = 50,
    ) -> pd.DataFrame:
        emb = self.encode_new_item(item_metadata)

        return self.index.query(
            query_vector=emb,
            top_k=top_k_users,
        )
