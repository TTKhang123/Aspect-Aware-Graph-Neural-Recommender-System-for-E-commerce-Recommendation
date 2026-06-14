from __future__ import annotations

import argparse
import json

from src.inference.cold_item_content_recommender import ColdItemContentRecommender


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--item-json",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--index-dir",
        type=str,
        default="outputs/content_lsh",
    )

    parser.add_argument(
        "--top-k-users",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    with open(args.item_json, "r", encoding="utf-8") as f:
        item_metadata = json.load(f)

    recommender = ColdItemContentRecommender(
        index_dir=args.index_dir,
    )

    users = recommender.recommend_users_for_new_item(
        item_metadata=item_metadata,
        top_k_users=args.top_k_users,
    )

    print("\nNearest users for this cold-start item:")
    print(users.to_string(index=False))


if __name__ == "__main__":
    main()
