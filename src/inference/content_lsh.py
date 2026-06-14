from __future__ import annotations

from pathlib import Path
from typing import Optional

import pickle
import numpy as np
import pandas as pd


class RandomHyperplaneLSH:
    def __init__(self, n_planes: int = 24, n_tables: int = 12, seed: int = 42):
        self.n_planes = n_planes
        self.n_tables = n_tables
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.planes = []
        self.tables = []
        self.user_ids = None
        self.user_idxs = None
        self.vectors = None

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm[norm < 1e-8] = 1.0
        return x / norm

    def _hash(self, vectors: np.ndarray, planes: np.ndarray) -> np.ndarray:
        bits = (vectors @ planes.T) >= 0
        codes = np.zeros(vectors.shape[0], dtype=np.int64)

        for i in range(self.n_planes):
            codes |= bits[:, i].astype(np.int64) << i

        return codes

    def fit(self, user_ids: np.ndarray, user_idxs: np.ndarray, vectors: np.ndarray):
        self.user_ids = np.asarray(user_ids)
        self.user_idxs = np.asarray(user_idxs)
        self.vectors = self._normalize(vectors.astype(np.float32))

        dim = self.vectors.shape[1]

        self.planes = []
        self.tables = []

        for _ in range(self.n_tables):
            planes = self.rng.normal(size=(self.n_planes, dim)).astype(np.float32)
            codes = self._hash(self.vectors, planes)

            table = {}
            for idx, code in enumerate(codes):
                table.setdefault(int(code), []).append(idx)

            self.planes.append(planes)
            self.tables.append(table)

        return self

    def query(
        self,
        query_vector: np.ndarray,
        top_k: int = 50,
        max_candidates: int = 10000,
    ) -> pd.DataFrame:
        if self.vectors is None:
            raise RuntimeError("LSH index has not been fitted.")

        q = query_vector.astype(np.float32).reshape(1, -1)
        q = self._normalize(q)

        candidates = set()

        for planes, table in zip(self.planes, self.tables):
            code = int(self._hash(q, planes)[0])
            candidates.update(table.get(code, []))

        if len(candidates) == 0:
            candidates = set(
                np.random.choice(
                    len(self.user_ids),
                    size=min(max_candidates, len(self.user_ids)),
                    replace=False,
                )
            )

        candidates = list(candidates)

        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        cand_vecs = self.vectors[candidates]
        scores = cand_vecs @ q.reshape(-1)

        order = np.argsort(-scores)[:top_k]

        rows = []
        for rank, pos in enumerate(order, start=1):
            idx = candidates[pos]
            rows.append(
                {
                    "rank": rank,
                    "user_id": str(self.user_ids[idx]),
                    "user_idx": int(self.user_idxs[idx]),
                    "similarity": float(scores[pos]),
                }
            )

        return pd.DataFrame(rows)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

    # @staticmethod
    # def load(path: str | Path) -> "RandomHyperplaneLSH":
    #     with open(path, "rb") as f:
    #         return pickle.load(f)

    @staticmethod
    def load(path: str | Path) -> "RandomHyperplaneLSH":
        import __main__

        # alias cho pickle được tạo trên Kaggle notebook
        __main__.RandomHyperplaneLSH = RandomHyperplaneLSH
        with open(path, "rb") as f:
            return pickle.load(f)
