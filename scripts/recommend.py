from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.inference.aspect_lightgcn_recommender import AspectLightGCNInference


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/aspect_lightgcn/lightgcn_aspects_masks_support/best_model.pt",
    )

    parser.add_argument(
        "--user-id",
        type=str,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default="data/processed/item_clean_min5.parquet",  # metadata_filtered
    )

    parser.add_argument(
        "--active-ablation",
        type=str,
        default="lightgcn_aspects_masks_support",
    )

    parser.add_argument(
        "--show-sample-users",
        action="store_true",
    )

    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    metadata_path = metadata_path if metadata_path.exists() else None

    recommender = AspectLightGCNInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        active_ablation=args.active_ablation,
        metadata_path=metadata_path,
    )

    if args.show_sample_users:
        users = recommender.sample_known_users(n=20)
        print("\nSample known users:")
        print(users.to_string(index=False))
        return

    if args.user_id is None:
        raise ValueError("Please provide --user-id or use --show-sample-users first.")

    recs = recommender.recommend_for_user(
        user_id=args.user_id,
        top_k=args.top_k,
        filter_seen=True,
    )
    #####
    if isinstance(recs, dict) and recs.get("user_status") == "cold_user_not_in_mapping":
        print("\nCold-start user detected: user_id not found in mapping.")
        print("\nFallback 1: Top popular items")
        print(recs["popular"].to_string(index=False))

        print("\nFallback 2: Top items by each criteria")

        for criteria_name, df in recs["by_criteria"].items():
            print("\n" + "=" * 80)
            print(f"Top items by {criteria_name}")
            print("=" * 80)
            print(df.to_string(index=False))
        return
    ######

    rows = []

    for r in recs:
        rows.append(
            {
                "rank": r.rank,
                "item_id": r.item_id,
                "title": r.title,
                "brand": r.brand,
                "score": r.score,
                "quality": r.aspects["item_quality"],
                "value": r.aspects["item_value"],
                "design": r.aspects["item_design"],
                "usability": r.aspects["item_usability"],
                "durability": r.aspects["item_durability"],
                "explanation": r.explanation,
            }
        )

    df = pd.DataFrame(rows)

    print("\nTop recommendations:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
