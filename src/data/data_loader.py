import gzip
import json
import gc
import logging
from pathlib import Path
from typing import Dict, Optional

import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    @staticmethod
    def clean_review_row(row: Dict) -> Optional[Dict]:
        reviewer_id = row.get("reviewerID")
        asin = row.get("asin")
        overall = row.get("overall")
        review_text = row.get("reviewText", "")
        summary = row.get("summary", "")
        unix_time = row.get("unixReviewTime")
        verified = row.get("verified", False)
        vote = row.get("vote", 0)

        if reviewer_id is None or asin is None or overall is None:
            return None

        try:
            overall = float(overall)
        except Exception:
            return None

        if overall < 1 or overall > 5:
            return None

        if not isinstance(review_text, str):
            review_text = ""

        if not isinstance(summary, str):
            summary = ""

        review_text = review_text.strip().lower()
        summary = summary.strip().lower()

        try:
            if isinstance(vote, str):
                vote = vote.replace(",", "")
            vote = int(vote)
        except Exception:
            vote = 0

        try:
            timestamp = pd.to_datetime(unix_time, unit="s")
        except Exception:
            return None

        return {
            "user_id": reviewer_id,
            "item_id": asin,
            "rating": overall,
            "review_text": review_text,
            "summary": summary,
            "verified": int(bool(verified)),
            "vote": vote,
            "helpful_score": np.log1p(vote),
            "timestamp": timestamp,
        }

    @staticmethod
    def clean_meta_row(row: Dict) -> Optional[Dict]:
        asin = row.get("asin")  # item id

        if asin is None:
            return None

        title = row.get("title", "")
        brand = row.get("brand", "")
        description = row.get("description", "")
        feature = row.get("feature", [])
        category = row.get("category", [])
        price = row.get("price")
        also_buy = row.get("also_buy", [])
        also_view = row.get("also_view", [])

        if not isinstance(title, str):
            title = ""

        if not isinstance(brand, str):
            brand = ""

        if isinstance(description, list):
            description = " ".join([str(x) for x in description])
        elif not isinstance(description, str):
            description = ""

        if isinstance(feature, list):
            feature_text = " ".join([str(x) for x in feature])
        else:
            feature_text = ""

        if isinstance(category, list):
            category_text = " > ".join([str(x) for x in category])
            main_category = category[0] if len(category) > 0 else ""
            sub_category = category[-1] if len(category) > 0 else ""
        else:
            category_text = ""
            main_category = ""
            sub_category = ""

        parsed_price = None

        if price is not None:
            try:
                price_str = str(price)
                price_str = price_str.replace("$", "").replace(",", "").strip()
                parsed_price = float(price_str)

                if parsed_price < 0:
                    parsed_price = None
            except Exception:
                parsed_price = None

        if not isinstance(also_buy, list):
            also_buy = []

        if not isinstance(also_view, list):
            also_view = []

        title = title.strip().lower()
        brand = brand.strip().lower()
        description = description.strip().lower()
        feature_text = feature_text.strip().lower()
        category_text = category_text.strip().lower()
        main_category = str(main_category).strip().lower()
        sub_category = str(sub_category).strip().lower()

        text_all = " ".join(
            [title, brand, category_text, feature_text, description]
        ).strip()

        return {
            "item_id": asin,
            "title": title,
            "brand": brand,
            "description": description,
            "feature_text": feature_text,
            "category_text": category_text,
            "main_category": main_category,
            "sub_category": sub_category,
            "price": parsed_price,
            "also_buy_count": len(also_buy),
            "also_view_count": len(also_view),
            "content_text": text_all,
        }


class AmazonDataLoader:
    def __init__(self, config: Dict):
        self.config = config

        self.reviews_path = Path(config["data"]["raw_data_path"])
        self.metadata_path = Path(config["data"]["metadata_path"])

        self.processed_dir = Path(config["data"].get("processed_dir", "data/processed"))
        self.temp_reviews_dir = Path(
            config["data"].get("temp_reviews_dir", "data/temp_reviews")
        )
        self.temp_metadata_dir = Path(
            config["data"].get("temp_metadata_dir", "data/temp_metadata")
        )

        self.min_user_interactions = config["data"].get("min_reviews_per_user", 5)
        self.min_item_interactions = config["data"].get("min_reviews_per_item", 5)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.temp_reviews_dir.mkdir(parents=True, exist_ok=True)
        self.temp_metadata_dir.mkdir(parents=True, exist_ok=True)

        self.reviews_dedup_path = self.processed_dir / "reviews_dedup.parquet"
        self.reviews_filtered_path = (
            self.processed_dir / "reviews_filtered_min5.parquet"
        )
        self.items_path = self.processed_dir / "items_clean_min5.parquet"
        self.merged_path = self.processed_dir / "reviews_merged_min5.parquet"

        self.train_path = self.processed_dir / "train.parquet"
        self.val_path = self.processed_dir / "val.parquet"
        self.test_path = self.processed_dir / "test.parquet"

        self.val_warm_path = self.processed_dir / "val_warm.parquet"
        self.test_warm_path = self.processed_dir / "test_warm.parquet"

    @staticmethod
    def _duck_path(path: Path) -> str:
        return str(path).replace("\\", "/").replace("'", "''")

    @staticmethod
    def read_json_gz(path: Path, limit: Optional[int] = None):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break

                if line.strip():
                    yield json.loads(line)

    def clear_temp_chunks(self):
        for p in self.temp_reviews_dir.glob("reviews_clean_chunk_*.parquet"):
            p.unlink()

        for p in self.temp_metadata_dir.glob("metadata_clean_chunk_*.parquet"):
            p.unlink()

    def clean_reviews_to_chunks(
        self,
        chunk_size: int = 300_000,
        limit: Optional[int] = None,
    ):
        buffer = []
        chunk_id = 0
        total_read = 0
        total_saved = 0

        for row in self.read_json_gz(self.reviews_path, limit=limit):
            total_read += 1

            cleaned = DataPreprocessor.clean_review_row(row)

            if cleaned is not None:
                buffer.append(cleaned)

            if len(buffer) >= chunk_size:
                df = pd.DataFrame(buffer)
                out_path = (
                    self.temp_reviews_dir
                    / f"reviews_clean_chunk_{chunk_id:04d}.parquet"
                )
                df.to_parquet(out_path, index=False)

                total_saved += len(df)

                logger.info(
                    "Saved review chunk %04d | rows: %s | total read: %s | total saved: %s",
                    chunk_id,
                    f"{len(df):,}",
                    f"{total_read:,}",
                    f"{total_saved:,}",
                )

                buffer.clear()
                del df
                gc.collect()

                chunk_id += 1

        if buffer:
            df = pd.DataFrame(buffer)
            out_path = (
                self.temp_reviews_dir / f"reviews_clean_chunk_{chunk_id:04d}.parquet"
            )
            df.to_parquet(out_path, index=False)

            total_saved += len(df)

            logger.info(
                "Saved final review chunk %04d | rows: %s | total read: %s | total saved: %s",
                chunk_id,
                f"{len(df):,}",
                f"{total_read:,}",
                f"{total_saved:,}",
            )

            buffer.clear()
            del df
            gc.collect()

        logger.info(
            "DONE reviews cleaning | total read=%s | total saved=%s",
            total_read,
            total_saved,
        )

    def deduplicate_reviews(self, con: duckdb.DuckDBPyConnection):
        reviews_chunks = self._duck_path(
            self.temp_reviews_dir / "reviews_clean_chunk_*.parquet"
        )
        dedup_path = self._duck_path(self.reviews_dedup_path)

        con.execute(f"""
            COPY (
                SELECT
                    user_id,
                    item_id,
                    rating,
                    review_text,
                    summary,
                    verified,
                    vote,
                    helpful_score,
                    timestamp
                FROM (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY user_id, item_id
                            ORDER BY timestamp DESC
                        ) AS rn
                    FROM read_parquet('{reviews_chunks}')
                )
                WHERE rn = 1
            ) TO '{dedup_path}' (FORMAT PARQUET)
            """)

        logger.info("Saved deduplicated reviews: %s", self.reviews_dedup_path)

    def filter_reviews_min5(self, con: duckdb.DuckDBPyConnection):
        dedup_path = self._duck_path(self.reviews_dedup_path)
        filtered_path = self._duck_path(self.reviews_filtered_path)

        con.execute(f"""
            COPY (
                WITH user_counts AS (
                    SELECT user_id, COUNT(*) AS user_n
                    FROM read_parquet('{dedup_path}')
                    GROUP BY user_id
                ),
                item_counts AS (
                    SELECT item_id, COUNT(*) AS item_n
                    FROM read_parquet('{dedup_path}')
                    GROUP BY item_id
                )
                SELECT r.*
                FROM read_parquet('{dedup_path}') r
                JOIN user_counts u
                    ON r.user_id = u.user_id
                JOIN item_counts i
                    ON r.item_id = i.item_id
                WHERE u.user_n >= {self.min_user_interactions}
                  AND i.item_n >= {self.min_item_interactions}
            ) TO '{filtered_path}' (FORMAT PARQUET)
            """)

        logger.info("Saved filtered reviews: %s", self.reviews_filtered_path)

    def clean_metadata_to_chunks(
        self,
        chunk_size: int = 100_000,
        limit: Optional[int] = None,
    ):
        buffer = []
        chunk_id = 0
        total_read = 0
        total_saved = 0

        for row in self.read_json_gz(self.metadata_path, limit=limit):
            total_read += 1

            cleaned = DataPreprocessor.clean_meta_row(row)

            if cleaned is not None:
                buffer.append(cleaned)

            if len(buffer) >= chunk_size:
                df = pd.DataFrame(buffer)
                out_path = (
                    self.temp_metadata_dir
                    / f"metadata_clean_chunk_{chunk_id:04d}.parquet"
                )
                df.to_parquet(out_path, index=False)

                total_saved += len(df)

                logger.info(
                    "Saved metadata chunk %04d | rows: %s | total read: %s | total saved: %s",
                    chunk_id,
                    f"{len(df):,}",
                    f"{total_read:,}",
                    f"{total_saved:,}",
                )

                buffer.clear()
                del df
                gc.collect()

                chunk_id += 1

        if buffer:
            df = pd.DataFrame(buffer)
            out_path = (
                self.temp_metadata_dir / f"metadata_clean_chunk_{chunk_id:04d}.parquet"
            )
            df.to_parquet(out_path, index=False)

            total_saved += len(df)

            logger.info(
                "Saved final metadata chunk %04d | rows: %s | total read: %s | total saved: %s",
                chunk_id,
                f"{len(df):,}",
                f"{total_read:,}",
                f"{total_saved:,}",
            )

            buffer.clear()
            del df
            gc.collect()

        logger.info(
            "DONE metadata cleaning | total read=%s | total saved=%s",
            total_read,
            total_saved,
        )

    def filter_metadata_by_reviews(self, con: duckdb.DuckDBPyConnection):
        meta_chunks = self._duck_path(
            self.temp_metadata_dir / "metadata_clean_chunk_*.parquet"
        )
        filtered_path = self._duck_path(self.reviews_filtered_path)
        items_path = self._duck_path(self.items_path)

        con.execute(f"""
            COPY (
                SELECT DISTINCT m.*
                FROM read_parquet('{meta_chunks}') m
                INNER JOIN (
                    SELECT DISTINCT item_id
                    FROM read_parquet('{filtered_path}')
                ) r
                ON m.item_id = r.item_id
            ) TO '{items_path}' (FORMAT PARQUET)
            """)

        logger.info("Saved clean metadata items: %s", self.items_path)

    def merge_reviews_metadata(self, con: duckdb.DuckDBPyConnection):
        filtered_path = self._duck_path(self.reviews_filtered_path)
        items_path = self._duck_path(self.items_path)
        merged_path = self._duck_path(self.merged_path)

        con.execute(f"""
            COPY (
                SELECT
                    r.user_id,
                    r.item_id,
                    r.rating,
                    r.review_text,
                    r.summary,
                    r.verified,
                    r.vote,
                    r.helpful_score,
                    r.timestamp,

                    m.title,
                    m.brand,
                    m.description,
                    m.feature_text,
                    m.category_text,
                    m.main_category,
                    m.sub_category,
                    m.price,
                    m.also_buy_count,
                    m.also_view_count,
                    m.content_text
                FROM read_parquet('{filtered_path}') r
                LEFT JOIN read_parquet('{items_path}') m
                ON r.item_id = m.item_id
            ) TO '{merged_path}' (FORMAT PARQUET)
            """)

        logger.info("Saved merged reviews: %s", self.merged_path)

    def create_temporal_splits(self, con: duckdb.DuckDBPyConnection):
        merged_path = self._duck_path(self.merged_path)

        train_path = self._duck_path(self.train_path)
        val_path = self._duck_path(self.val_path)
        test_path = self._duck_path(self.test_path)

        con.execute(f"""
            COPY (
                SELECT *
                FROM read_parquet('{merged_path}')
                WHERE timestamp < DATE '2017-01-01'
            ) TO '{train_path}' (FORMAT PARQUET)
            """)

        con.execute(f"""
            COPY (
                SELECT *
                FROM read_parquet('{merged_path}')
                WHERE timestamp >= DATE '2017-01-01'
                  AND timestamp < DATE '2018-01-01'
            ) TO '{val_path}' (FORMAT PARQUET)
            """)

        con.execute(f"""
            COPY (
                SELECT *
                FROM read_parquet('{merged_path}')
                WHERE timestamp >= DATE '2018-01-01'
            ) TO '{test_path}' (FORMAT PARQUET)
            """)

        logger.info("Saved temporal splits: train / val / test")

    def create_warm_splits(self, con: duckdb.DuckDBPyConnection):
        train_path = self._duck_path(self.train_path)
        val_path = self._duck_path(self.val_path)
        test_path = self._duck_path(self.test_path)

        val_warm_path = self._duck_path(self.val_warm_path)
        test_warm_path = self._duck_path(self.test_warm_path)

        con.execute(f"""
            COPY (
                WITH train_users AS (
                    SELECT DISTINCT user_id
                    FROM read_parquet('{train_path}')
                ),
                train_items AS (
                    SELECT DISTINCT item_id
                    FROM read_parquet('{train_path}')
                )
                SELECT v.*
                FROM read_parquet('{val_path}') v
                INNER JOIN train_users tu
                    ON v.user_id = tu.user_id
                INNER JOIN train_items ti
                    ON v.item_id = ti.item_id
            ) TO '{val_warm_path}' (FORMAT PARQUET)
            """)

        con.execute(f"""
            COPY (
                WITH train_users AS (
                    SELECT DISTINCT user_id
                    FROM read_parquet('{train_path}')
                ),
                train_items AS (
                    SELECT DISTINCT item_id
                    FROM read_parquet('{train_path}')
                )
                SELECT t.*
                FROM read_parquet('{test_path}') t
                INNER JOIN train_users tu
                    ON t.user_id = tu.user_id
                INNER JOIN train_items ti
                    ON t.item_id = ti.item_id
            ) TO '{test_warm_path}' (FORMAT PARQUET)
            """)

        logger.info("Saved warm splits: val_warm / test_warm")

    def save_summary_files(self, con: duckdb.DuckDBPyConnection):
        filtered_path = self._duck_path(self.reviews_filtered_path)
        items_path = self._duck_path(self.items_path)
        merged_path = self._duck_path(self.merged_path)

        train_path = self._duck_path(self.train_path)
        val_path = self._duck_path(self.val_path)
        test_path = self._duck_path(self.test_path)

        filtered_stats = con.execute(f"""
            SELECT
                COUNT(*) AS interactions,
                COUNT(DISTINCT user_id) AS users,
                COUNT(DISTINCT item_id) AS items,
                MIN(timestamp) AS min_time,
                MAX(timestamp) AS max_time
            FROM read_parquet('{filtered_path}')
            """).df()

        interactions = filtered_stats.loc[0, "interactions"]
        users = filtered_stats.loc[0, "users"]
        items = filtered_stats.loc[0, "items"]
        sparsity = 1 - interactions / (users * items)

        phase2_summary = filtered_stats.copy()
        phase2_summary["sparsity"] = sparsity
        phase2_summary["min_user_interactions"] = self.min_user_interactions
        phase2_summary["min_item_interactions"] = self.min_item_interactions
        phase2_summary.to_csv(
            self.processed_dir / "phase2_filtered_summary.csv", index=False
        )

        items_stats = con.execute(f"""
            SELECT
                COUNT(*) AS metadata_items,
                COUNT(DISTINCT item_id) AS unique_items,
                SUM(CASE WHEN title IS NULL OR title = '' THEN 1 ELSE 0 END) AS missing_title,
                SUM(CASE WHEN brand IS NULL OR brand = '' THEN 1 ELSE 0 END) AS missing_brand,
                SUM(CASE WHEN description IS NULL OR description = '' THEN 1 ELSE 0 END) AS missing_description,
                SUM(CASE WHEN price IS NULL THEN 1 ELSE 0 END) AS missing_price
            FROM read_parquet('{items_path}')
            """).df()
        items_stats.to_csv(self.processed_dir / "items_stats.csv", index=False)

        coverage = con.execute(f"""
            SELECT
                r.review_items,
                i.metadata_items,
                i.metadata_items * 100.0 / r.review_items AS metadata_coverage_percent
            FROM
                (
                    SELECT COUNT(DISTINCT item_id) AS review_items
                    FROM read_parquet('{filtered_path}')
                ) r,
                (
                    SELECT COUNT(DISTINCT item_id) AS metadata_items
                    FROM read_parquet('{items_path}')
                ) i
            """).df()
        coverage.to_csv(self.processed_dir / "metadata_coverage.csv", index=False)

        merged_stats = con.execute(f"""
            SELECT
                COUNT(*) AS interactions,
                COUNT(DISTINCT user_id) AS users,
                COUNT(DISTINCT item_id) AS items,
                SUM(CASE WHEN title IS NULL THEN 1 ELSE 0 END) AS rows_without_metadata
            FROM read_parquet('{merged_path}')
            """).df()
        merged_stats.to_csv(self.processed_dir / "merged_stats.csv", index=False)

        split_stats = con.execute(f"""
            SELECT 'train' AS split,
                   COUNT(*) AS interactions,
                   COUNT(DISTINCT user_id) AS users,
                   COUNT(DISTINCT item_id) AS items,
                   MIN(timestamp) AS min_time,
                   MAX(timestamp) AS max_time
            FROM read_parquet('{train_path}')

            UNION ALL

            SELECT 'val' AS split,
                   COUNT(*) AS interactions,
                   COUNT(DISTINCT user_id) AS users,
                   COUNT(DISTINCT item_id) AS items,
                   MIN(timestamp) AS min_time,
                   MAX(timestamp) AS max_time
            FROM read_parquet('{val_path}')

            UNION ALL

            SELECT 'test' AS split,
                   COUNT(*) AS interactions,
                   COUNT(DISTINCT user_id) AS users,
                   COUNT(DISTINCT item_id) AS items,
                   MIN(timestamp) AS min_time,
                   MAX(timestamp) AS max_time
            FROM read_parquet('{test_path}')
            """).df()
        split_stats.to_csv(self.processed_dir / "split_stats.csv", index=False)

        cold_start_stats = con.execute(f"""
            WITH train_users AS (
                SELECT DISTINCT user_id
                FROM read_parquet('{train_path}')
            ),
            train_items AS (
                SELECT DISTINCT item_id
                FROM read_parquet('{train_path}')
            ),
            val_data AS (
                SELECT *
                FROM read_parquet('{val_path}')
            ),
            test_data AS (
                SELECT *
                FROM read_parquet('{test_path}')
            )

            SELECT
                'val' AS split,
                COUNT(*) AS interactions,
                SUM(CASE WHEN tu.user_id IS NULL THEN 1 ELSE 0 END) AS cold_user_rows,
                SUM(CASE WHEN ti.item_id IS NULL THEN 1 ELSE 0 END) AS cold_item_rows
            FROM val_data v
            LEFT JOIN train_users tu ON v.user_id = tu.user_id
            LEFT JOIN train_items ti ON v.item_id = ti.item_id

            UNION ALL

            SELECT
                'test' AS split,
                COUNT(*) AS interactions,
                SUM(CASE WHEN tu.user_id IS NULL THEN 1 ELSE 0 END) AS cold_user_rows,
                SUM(CASE WHEN ti.item_id IS NULL THEN 1 ELSE 0 END) AS cold_item_rows
            FROM test_data t
            LEFT JOIN train_users tu ON t.user_id = tu.user_id
            LEFT JOIN train_items ti ON t.item_id = ti.item_id
            """).df()
        cold_start_stats.to_csv(
            self.processed_dir / "cold_start_stats.csv", index=False
        )

        phase2_final_summary = {
            "filtered_interactions": interactions,
            "filtered_users": users,
            "filtered_items": items,
            "filtered_sparsity": sparsity,
            "min_user_interactions": self.min_user_interactions,
            "min_item_interactions": self.min_item_interactions,
        }

        pd.DataFrame([phase2_final_summary]).to_csv(
            self.processed_dir / "phase2_final_summary.csv",
            index=False,
        )

        logger.info("Saved summary CSV files")

    def run_pipeline(
        self,
        review_chunk_size: int = 300_000,
        metadata_chunk_size: int = 100_000,
        review_limit: Optional[int] = None,
        metadata_limit: Optional[int] = None,
        clear_temp: bool = True,
    ):
        if clear_temp:
            self.clear_temp_chunks()

        con = duckdb.connect()

        try:
            logger.info("Step 1/9: Clean reviews to chunks")
            self.clean_reviews_to_chunks(
                chunk_size=review_chunk_size,
                limit=review_limit,
            )

            logger.info("Step 2/9: Deduplicate reviews")
            self.deduplicate_reviews(con)

            logger.info("Step 3/9: Filter reviews min user/item interactions")
            self.filter_reviews_min5(con)

            logger.info("Step 4/9: Clean metadata to chunks")
            self.clean_metadata_to_chunks(
                chunk_size=metadata_chunk_size,
                limit=metadata_limit,
            )

            logger.info("Step 5/9: Filter metadata by filtered review items")
            self.filter_metadata_by_reviews(con)

            logger.info("Step 6/9: Merge reviews and metadata")
            self.merge_reviews_metadata(con)

            logger.info("Step 7/9: Create temporal train/val/test splits")
            self.create_temporal_splits(con)

            logger.info("Step 8/9: Create warm validation/test splits")
            self.create_warm_splits(con)

            logger.info("Step 9/9: Save summary files")
            self.save_summary_files(con)

            logger.info("Pipeline completed successfully")

        finally:
            con.close()
