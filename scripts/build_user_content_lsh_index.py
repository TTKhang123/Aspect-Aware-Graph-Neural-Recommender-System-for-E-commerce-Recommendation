from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.inference.content_profile import build_item_content_text
from src.inference.content_lsh import RandomHyperplaneLSH


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-path",
        type=str,
        default="data/processed/model_ready/train_model.parquet",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default="data/processed/metadata_filtered.parquet",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/content_lsh",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--n-planes", type=int, default=24)
    parser.add_argument("--n-tables", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading train interactions...")
    train = pd.read_parquet(
        args.train_path,
        columns=["user_id", "user_idx", "item_id", "rating"],
    )

    train = train.dropna(subset=["user_id", "user_idx", "item_id", "rating"])
    train = train[train["rating"] >= args.min_rating].copy()

    print("Loading metadata...")
    meta = pd.read_parquet(args.metadata)

    if "asin" in meta.columns and "item_id" not in meta.columns:
        meta = meta.rename(columns={"asin": "item_id"})

    keep_cols = [
        "item_id",
        "title",
        "brand",
        "categories",
        "feature",
        "description",
        "price",
        "salesRank",
        "similar",
    ]

    keep_cols = [c for c in keep_cols if c in meta.columns]
    meta = meta[keep_cols].dropna(subset=["item_id"]).copy()

    print("Merging train with metadata...")
    df = train.merge(meta, on="item_id", how="inner")

    item_meta = df.drop_duplicates("item_id").copy()

    if args.max_items is not None and len(item_meta) > args.max_items:
        item_meta = item_meta.sample(args.max_items, random_state=args.seed)

    print("Unique metadata items:", len(item_meta))

    print("Building item profile texts...")
    item_meta["content_text"] = [
        build_item_content_text(row) for row in item_meta.to_dict("records")
    ]

    encoder = SentenceTransformer(args.embedding_model)

    print("Encoding item metadata...")
    item_embeddings = encoder.encode(
        item_meta["content_text"].tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    item_ids = item_meta["item_id"].astype(str).to_numpy()

    item_emb_df = pd.DataFrame(
        {
            "item_id": item_ids,
            "emb_idx": np.arange(len(item_ids)),
        }
    )

    print("Building user content profiles...")
    df = train.merge(item_emb_df, on="item_id", how="inner")

    user_rows = []

    for (user_id, user_idx), g in tqdm(
        df.groupby(["user_id", "user_idx"], sort=False),
        desc="user profiles",
    ):
        emb = item_embeddings[g["emb_idx"].to_numpy()]
        profile = emb.mean(axis=0)
        norm = np.linalg.norm(profile)

        if norm > 1e-8:
            profile = profile / norm

        user_rows.append(
            {
                "user_id": user_id,
                "user_idx": int(user_idx),
                "profile": profile.astype(np.float32),
                "n_positive_items": len(g),
            }
        )

    user_ids = np.array([r["user_id"] for r in user_rows])
    user_idxs = np.array([r["user_idx"] for r in user_rows], dtype=np.int64)
    user_vectors = np.vstack([r["profile"] for r in user_rows]).astype(np.float32)

    print("Fitting LSH index...")
    index = RandomHyperplaneLSH(
        n_planes=args.n_planes,
        n_tables=args.n_tables,
        seed=args.seed,
    )
    index.fit(user_ids=user_ids, user_idxs=user_idxs, vectors=user_vectors)

    index_path = output_dir / "user_content_lsh.pkl"
    index.save(index_path)

    np.save(output_dir / "item_content_embeddings.npy", item_embeddings)

    item_emb_df.to_parquet(output_dir / "item_embedding_map.parquet", index=False)

    user_profile_df = pd.DataFrame(
        {
            "user_id": user_ids,
            "user_idx": user_idxs,
            "n_positive_items": [r["n_positive_items"] for r in user_rows],
        }
    )
    user_profile_df.to_parquet(
        output_dir / "user_content_profiles.parquet", index=False
    )

    joblib.dump(
        {
            "embedding_model": args.embedding_model,
            "n_planes": args.n_planes,
            "n_tables": args.n_tables,
        },
        output_dir / "content_lsh_metadata.joblib",
    )

    print("Saved:")
    print(index_path)
    print(output_dir / "content_lsh_metadata.joblib")
    print(output_dir / "user_content_profiles.parquet")


if __name__ == "__main__":
    main()
