import logging
from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelReadyDataBuilder:
    """
    Build model-ready datasets by:
    1. Creating item-level criteria vectors from ABSA data
    2. Merging criteria vectors into train/val/test splits
    3. Validating data quality and coverage
    4. Building user/item ID mappings from train +val + test
    5. Creating encoded model-ready datasets for modeling
    """

    def __init__(self, config: Dict):
        self.config = config
        self.processed_dir = Path(config["data"].get("processed_dir", "data/processed"))
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Define paths
        self.absa_train_path = str(
            self.processed_dir / "absa_train_chunks" / "*.parquet"
        ).replace("\\", "/")

        self.item_criteria_path = (
            self.processed_dir / "item_criteria_vectors_train.parquet"
        )

        self.splits = {
            "train": self.processed_dir / "train.parquet",
            "val": self.processed_dir / "val.parquet",
            "test": self.processed_dir / "test.parquet",
            "val_warm": self.processed_dir / "val_warm.parquet",
            "test_warm": self.processed_dir / "test_warm.parquet",
        }

        # Merged split paths
        self.merged_splits = {
            split_name: self.processed_dir / f"{split_name}_with_item_criteria.parquet"
            for split_name in self.splits.keys()
        }

        # Final model-ready output dir
        self.model_ready_dir = self.processed_dir / "model_ready"
        self.model_ready_dir.mkdir(parents=True, exist_ok=True)

        # Mapping paths
        self.user_map_path = self.model_ready_dir / "user2idx.parquet"
        self.item_map_path = self.model_ready_dir / "item2idx.parquet"

        self.con = duckdb.connect()

    @staticmethod
    def _duck_path(path: Path) -> str:
        """Convert path to DuckDB-compatible format."""
        return str(path).replace("\\", "/").replace("'", "''")

    def build_item_criteria_vectors(self) -> pd.DataFrame:
        """
        STEP 4: Build item-level criteria vectors
        Aggregate ABSA criteria at item level from training reviews
        """
        logger.info("Building item-level criteria vectors from ABSA data...")

        output_path = self._duck_path(self.item_criteria_path)

        self.con.execute(f"""
        COPY (
            SELECT
                item_id,

                AVG(crit_quality) AS item_quality,
                AVG(crit_value) AS item_value,
                AVG(crit_design) AS item_design,
                AVG(crit_usability) AS item_usability,
                AVG(crit_durability) AS item_durability,

                COUNT(crit_quality) AS item_quality_count,
                COUNT(crit_value) AS item_value_count,
                COUNT(crit_design) AS item_design_count,
                COUNT(crit_usability) AS item_usability_count,
                COUNT(crit_durability) AS item_durability_count,

                COUNT(*) AS item_train_reviews

            FROM read_parquet('{self.absa_train_path}')
            GROUP BY item_id
        ) TO '{output_path}' (FORMAT PARQUET)
        """)

        logger.info(f"Item criteria vectors saved to: {self.item_criteria_path}")

        # Verify coverage
        coverage_result = self.con.execute(f"""
        SELECT
            COUNT(*) AS items,

            AVG(CASE WHEN item_quality IS NOT NULL THEN 1 ELSE 0 END) * 100 AS quality_pct,
            AVG(CASE WHEN item_value IS NOT NULL THEN 1 ELSE 0 END) * 100 AS value_pct,
            AVG(CASE WHEN item_design IS NOT NULL THEN 1 ELSE 0 END) * 100 AS design_pct,
            AVG(CASE WHEN item_usability IS NOT NULL THEN 1 ELSE 0 END) * 100 AS usability_pct,
            AVG(CASE WHEN item_durability IS NOT NULL THEN 1 ELSE 0 END) * 100 AS durability_pct

        FROM read_parquet('{output_path}')
        """).df()

        logger.info(f"\nItem criteria coverage:\n{coverage_result.to_string()}\n")

        return coverage_result

    def merge_criteria_to_splits(self) -> None:
        """
        STEP 5: Merge criteria vectors into train/val/test splits
        """
        logger.info("Merging criteria vectors into train/val/test splits...")

        item_criteria_path = self._duck_path(self.item_criteria_path)

        for split_name, input_path in self.splits.items():
            output_path = (
                self.processed_dir / f"{split_name}_with_item_criteria.parquet"
            )
            output_path_str = self._duck_path(output_path)
            input_path_str = self._duck_path(input_path)

            logger.info(f"Processing {split_name}...")

            self.con.execute(f"""
            COPY (
                SELECT
                    s.*,

                    c.item_quality,
                    c.item_value,
                    c.item_design,
                    c.item_usability,
                    c.item_durability,

                    c.item_quality_count,
                    c.item_value_count,
                    c.item_design_count,
                    c.item_usability_count,
                    c.item_durability_count,

                    c.item_train_reviews

                FROM read_parquet('{input_path_str}') s
                LEFT JOIN read_parquet('{item_criteria_path}') c
                    ON s.item_id = c.item_id

            ) TO '{output_path_str}' (FORMAT PARQUET)
            """)

            logger.info(f"Saved: {output_path}")

    def validate_merged_datasets(self) -> pd.DataFrame:
        """
        Validate merged datasets and check criteria coverage across all splits
        """
        logger.info("Validating merged datasets...")

        validation_results = []

        for split_name, _ in self.splits.items():
            merged_path = (
                self.processed_dir / f"{split_name}_with_item_criteria.parquet"
            )
            merged_path_str = self._duck_path(merged_path)

            result = self.con.execute(f"""
            SELECT
                '{split_name}' AS split,
                COUNT(*) AS rows,
                AVG(CASE WHEN item_quality IS NOT NULL THEN 1 ELSE 0 END) * 100 AS quality_pct,
                AVG(CASE WHEN item_value IS NOT NULL THEN 1 ELSE 0 END) * 100 AS value_pct,
                AVG(CASE WHEN item_design IS NOT NULL THEN 1 ELSE 0 END) * 100 AS design_pct,
                AVG(CASE WHEN item_usability IS NOT NULL THEN 1 ELSE 0 END) * 100 AS usability_pct,
                AVG(CASE WHEN item_durability IS NOT NULL THEN 1 ELSE 0 END) * 100 AS durability_pct
            FROM read_parquet('{merged_path_str}')
            """).df()

            validation_results.append(result)

        validation_df = pd.concat(validation_results, ignore_index=True)

        logger.info(f"\nValidation Results:\n{validation_df.to_string(index=False)}\n")

        return validation_df

    # def build_id_mappings(self) -> pd.DataFrame:
    #     """
    #     STEP 6: Build user_id/item_id -> integer index mappings from train split only.

    #     For warm-start MF/NCF:
    #     - user/item mappings must be learned from train only
    #     - val_warm/test_warm should only contain users/items seen in train
    #     """
    #     logger.info("Building user/item ID mappings from train split...")

    #     train_path = self._duck_path(
    #         self.processed_dir / "train_with_item_criteria.parquet"
    #     )

    #     user_map_path = self._duck_path(self.user_map_path)
    #     item_map_path = self._duck_path(self.item_map_path)

    #     self.con.execute(f"""
    #     COPY (
    #         SELECT
    #             user_id,
    #             ROW_NUMBER() OVER (ORDER BY user_id) - 1 AS user_idx
    #         FROM (
    #             SELECT DISTINCT user_id
    #             FROM read_parquet('{train_path}')
    #         )
    #     ) TO '{user_map_path}' (FORMAT PARQUET)
    #     """)

    #     self.con.execute(f"""
    #     COPY (
    #         SELECT
    #             item_id,
    #             ROW_NUMBER() OVER (ORDER BY item_id) - 1 AS item_idx
    #         FROM (
    #             SELECT DISTINCT item_id
    #             FROM read_parquet('{train_path}')
    #         )
    #     ) TO '{item_map_path}' (FORMAT PARQUET)
    #     """)

    #     stats = self.con.execute(f"""
    #     SELECT
    #         (SELECT COUNT(*) FROM read_parquet('{user_map_path}')) AS num_users,
    #         (SELECT COUNT(*) FROM read_parquet('{item_map_path}')) AS num_items
    #     """).df()

    #     logger.info(f"\nID mapping stats:\n{stats.to_string(index=False)}\n")

    #     return stats

    def build_id_mappings(self) -> pd.DataFrame:
        """
        STEP 6: Build global user_id/item_id -> integer index mappings
        from train + val + test.

        This allows val/test cold users/items to have valid integer ids
        instead of NULL user_idx/item_idx.
        """
        logger.info("Building GLOBAL user/item ID mappings from train + val + test...")

        split_paths = [
            self.processed_dir / "train_with_item_criteria.parquet",
            self.processed_dir / "val_with_item_criteria.parquet",
            self.processed_dir / "test_with_item_criteria.parquet",
        ]

        split_paths_str = [self._duck_path(p) for p in split_paths]

        user_map_path = self._duck_path(self.user_map_path)
        item_map_path = self._duck_path(self.item_map_path)

        union_sql = ", ".join([f"'{p}'" for p in split_paths_str])

        self.con.execute(f"""
        COPY (
            SELECT
                user_id,
                ROW_NUMBER() OVER (ORDER BY user_id) - 1 AS user_idx
            FROM (
                SELECT DISTINCT user_id
                FROM read_parquet([{union_sql}])
                WHERE user_id IS NOT NULL
            )
        ) TO '{user_map_path}' (FORMAT PARQUET)
        """)

        self.con.execute(f"""
        COPY (
            SELECT
                item_id,
                ROW_NUMBER() OVER (ORDER BY item_id) - 1 AS item_idx
            FROM (
                SELECT DISTINCT item_id
                FROM read_parquet([{union_sql}])
                WHERE item_id IS NOT NULL
            )
        ) TO '{item_map_path}' (FORMAT PARQUET)
        """)

        stats = self.con.execute(f"""
        SELECT
            (SELECT COUNT(*) FROM read_parquet('{user_map_path}')) AS num_users,
            (SELECT COUNT(*) FROM read_parquet('{item_map_path}')) AS num_items
        """).df()

        logger.info(f"\nGLOBAL ID mapping stats:\n{stats.to_string(index=False)}\n")

        return stats

    def build_single_model_ready_split(
        self,
        split_name: str,
        input_path: Path,
        output_path: Path,
    ) -> pd.DataFrame:
        """
        STEP 7 helper: Build one encoded model-ready split.

        Important:
        - train / val_warm / test_warm are encoded with INNER JOIN because they must be
          fully known users/items for warm-start MF/NCF style models.
        - val / test are encoded with LEFT JOIN so their original rows are preserved.
          Cold-start rows will have NULL user_idx/item_idx and are marked by:
              is_known_user, is_known_item, is_warm_row

        NaN criteria handling:
        - fill missing item criteria with neutral score 3.0
        - add mask columns to indicate whether criterion was observed
        - add support features using log(1 + count)
        """
        logger.info(f"Building model-ready split: {split_name}")

        input_path_str = self._duck_path(input_path)
        output_path_str = self._duck_path(output_path)
        user_map_path = self._duck_path(self.user_map_path)
        item_map_path = self._duck_path(self.item_map_path)

        train_path_str = self._duck_path(
            self.processed_dir / "train_with_item_criteria.parquet"
        )

        # Preserve original val/test rows. Do NOT silently turn them into warm splits.
        # For warm-only files, INNER JOIN is expected and should not drop rows if the
        # warm split was built correctly.
        preserve_cold_rows = split_name in {"val", "test"}
        join_type = "LEFT JOIN" if preserve_cold_rows else "INNER JOIN"

        # self.con.execute(f"""
        # COPY (
        #     SELECT
        #         u.user_idx,
        #         i.item_idx,

        #         CASE WHEN u.user_idx IS NULL THEN 0 ELSE 1 END AS is_known_user,
        #         CASE WHEN i.item_idx IS NULL THEN 0 ELSE 1 END AS is_known_item,
        #         CASE
        #             WHEN u.user_idx IS NOT NULL AND i.item_idx IS NOT NULL THEN 1
        #             ELSE 0
        #         END AS is_warm_row,

        #         s.user_id,
        #         s.item_id,

        #         CAST(s.rating AS FLOAT) AS rating,

        #         COALESCE(CAST(s.item_quality AS FLOAT), 3.0) AS item_quality,
        #         COALESCE(CAST(s.item_value AS FLOAT), 3.0) AS item_value,
        #         COALESCE(CAST(s.item_design AS FLOAT), 3.0) AS item_design,
        #         COALESCE(CAST(s.item_usability AS FLOAT), 3.0) AS item_usability,
        #         COALESCE(CAST(s.item_durability AS FLOAT), 3.0) AS item_durability,

        #         CASE WHEN s.item_quality IS NULL THEN 0 ELSE 1 END AS mask_quality,
        #         CASE WHEN s.item_value IS NULL THEN 0 ELSE 1 END AS mask_value,
        #         CASE WHEN s.item_design IS NULL THEN 0 ELSE 1 END AS mask_design,
        #         CASE WHEN s.item_usability IS NULL THEN 0 ELSE 1 END AS mask_usability,
        #         CASE WHEN s.item_durability IS NULL THEN 0 ELSE 1 END AS mask_durability,

        #         LOG(1 + COALESCE(s.item_quality_count, 0)) AS support_quality,
        #         LOG(1 + COALESCE(s.item_value_count, 0)) AS support_value,
        #         LOG(1 + COALESCE(s.item_design_count, 0)) AS support_design,
        #         LOG(1 + COALESCE(s.item_usability_count, 0)) AS support_usability,
        #         LOG(1 + COALESCE(s.item_durability_count, 0)) AS support_durability,

        #         LOG(1 + COALESCE(s.item_train_reviews, 0)) AS item_support,

        #         s.timestamp

        #     FROM read_parquet('{input_path_str}') s

        #     {join_type} read_parquet('{user_map_path}') u
        #         ON s.user_id = u.user_id

        #     {join_type} read_parquet('{item_map_path}') i
        #         ON s.item_id = i.item_id
        # ) TO '{output_path_str}' (FORMAT PARQUET)
        # """)
        self.con.execute(f"""
        COPY (
            WITH train_users AS (
                SELECT DISTINCT user_id
                FROM read_parquet('{train_path_str}')
            ),
            train_items AS (
                SELECT DISTINCT item_id
                FROM read_parquet('{train_path_str}')
            )

            SELECT
                u.user_idx,
                i.item_idx,

                CASE WHEN tu.user_id IS NULL THEN 0 ELSE 1 END AS is_train_user,
                CASE WHEN ti.item_id IS NULL THEN 0 ELSE 1 END AS is_train_item,
                CASE
                    WHEN tu.user_id IS NOT NULL AND ti.item_id IS NOT NULL THEN 1
                    ELSE 0
                END AS is_warm_row,

                s.user_id,
                s.item_id,

                CAST(s.rating AS FLOAT) AS rating,

                COALESCE(CAST(s.item_quality AS FLOAT), 3.0) AS item_quality,
                COALESCE(CAST(s.item_value AS FLOAT), 3.0) AS item_value,
                COALESCE(CAST(s.item_design AS FLOAT), 3.0) AS item_design,
                COALESCE(CAST(s.item_usability AS FLOAT), 3.0) AS item_usability,
                COALESCE(CAST(s.item_durability AS FLOAT), 3.0) AS item_durability,

                CASE WHEN s.item_quality IS NULL THEN 0 ELSE 1 END AS mask_quality,
                CASE WHEN s.item_value IS NULL THEN 0 ELSE 1 END AS mask_value,
                CASE WHEN s.item_design IS NULL THEN 0 ELSE 1 END AS mask_design,
                CASE WHEN s.item_usability IS NULL THEN 0 ELSE 1 END AS mask_usability,
                CASE WHEN s.item_durability IS NULL THEN 0 ELSE 1 END AS mask_durability,

                LOG(1 + COALESCE(s.item_quality_count, 0)) AS support_quality,
                LOG(1 + COALESCE(s.item_value_count, 0)) AS support_value,
                LOG(1 + COALESCE(s.item_design_count, 0)) AS support_design,
                LOG(1 + COALESCE(s.item_usability_count, 0)) AS support_usability,
                LOG(1 + COALESCE(s.item_durability_count, 0)) AS support_durability,

                LOG(1 + COALESCE(s.item_train_reviews, 0)) AS item_support,

                s.timestamp

            FROM read_parquet('{input_path_str}') s

            INNER JOIN read_parquet('{user_map_path}') u
                ON s.user_id = u.user_id

            INNER JOIN read_parquet('{item_map_path}') i
                ON s.item_id = i.item_id

            LEFT JOIN train_users tu
                ON s.user_id = tu.user_id

            LEFT JOIN train_items ti
                ON s.item_id = ti.item_id

        ) TO '{output_path_str}' (FORMAT PARQUET)
        """)

        stats = self.con.execute(f"""
        SELECT
            '{split_name}' AS split,
            COUNT(*) AS rows,
            COUNT(DISTINCT user_idx) AS users,
            COUNT(DISTINCT item_idx) AS items,
            MIN(rating) AS min_rating,
            MAX(rating) AS max_rating,

            AVG(mask_quality) * 100 AS quality_observed_pct,
            AVG(mask_value) * 100 AS value_observed_pct,
            AVG(mask_design) * 100 AS design_observed_pct,
            AVG(mask_usability) * 100 AS usability_observed_pct,
            AVG(mask_durability) * 100 AS durability_observed_pct,

            SUM(CASE WHEN is_train_user = 0 THEN 1 ELSE 0 END) AS cold_user_rows,
            SUM(CASE WHEN is_train_item = 0 THEN 1 ELSE 0 END) AS cold_item_rows,
            SUM(CASE WHEN is_warm_row = 0 THEN 1 ELSE 0 END) AS non_warm_rows,
            AVG(is_warm_row) * 100 AS warm_row_pct

        FROM read_parquet('{output_path_str}')
        """).df()

        logger.info(
            f"\nModel-ready stats for {split_name}:\n"
            f"{stats.to_string(index=False)}\n"
        )

        return stats

    def build_model_ready_splits(self) -> pd.DataFrame:
        """
        STEP 7: Build encoded model-ready train/val/test files.

        Outputs:
        - data/processed/model_ready/train_model.parquet
        - data/processed/model_ready/val_model.parquet
        - data/processed/model_ready/test_model.parquet
        - data/processed/model_ready/val_warm_model.parquet
        - data/processed/model_ready/test_warm_model.parquet
        """
        logger.info("Building final model-ready datasets...")

        all_stats = []

        for split_name, merged_path in self.merged_splits.items():
            output_path = self.model_ready_dir / f"{split_name}_model.parquet"

            stats = self.build_single_model_ready_split(
                split_name=split_name,
                input_path=merged_path,
                output_path=output_path,
            )

            all_stats.append(stats)

        model_ready_stats = pd.concat(all_stats, ignore_index=True)

        summary_path = self.model_ready_dir / "model_ready_summary.csv"
        model_ready_stats.to_csv(summary_path, index=False)

        logger.info(f"Model-ready summary saved to: {summary_path}")
        logger.info(
            f"\nModel-ready summary:\n" f"{model_ready_stats.to_string(index=False)}\n"
        )

        return model_ready_stats

    def validate_model_ready_splits(self) -> pd.DataFrame:
        """
        STEP 8: Validate final model-ready splits.

        Checks:
        - interaction overlap between train and warm val/test
        - expected result should be 0 for val_warm and test_warm
        """
        logger.info("Validating final model-ready splits...")

        train_model = self._duck_path(self.model_ready_dir / "train_model.parquet")
        val_warm_model = self._duck_path(
            self.model_ready_dir / "val_warm_model.parquet"
        )
        test_warm_model = self._duck_path(
            self.model_ready_dir / "test_warm_model.parquet"
        )

        overlap = self.con.execute(f"""
        WITH train_pairs AS (
            SELECT DISTINCT user_idx, item_idx
            FROM read_parquet('{train_model}')
        ),
        val_pairs AS (
            SELECT DISTINCT user_idx, item_idx
            FROM read_parquet('{val_warm_model}')
        ),
        test_pairs AS (
            SELECT DISTINCT user_idx, item_idx
            FROM read_parquet('{test_warm_model}')
        )

        SELECT
            'val_warm' AS split,
            COUNT(*) AS overlapping_pairs
        FROM train_pairs t
        INNER JOIN val_pairs v
            ON t.user_idx = v.user_idx
           AND t.item_idx = v.item_idx

        UNION ALL

        SELECT
            'test_warm' AS split,
            COUNT(*) AS overlapping_pairs
        FROM train_pairs t
        INNER JOIN test_pairs x
            ON t.user_idx = x.user_idx
           AND t.item_idx = x.item_idx
        """).df()

        logger.info(
            f"\nInteraction overlap check:\n" f"{overlap.to_string(index=False)}\n"
        )

        return overlap

    def build_pipeline(self, validate: bool = True) -> Dict:
        """
        Execute full model-ready data building pipeline
        """
        logger.info("=" * 80)
        logger.info("STARTING MODEL-READY DATA BUILDING PIPELINE")
        logger.info("=" * 80)

        results = {}

        # Step 4: Build item-level criteria vectors
        try:
            coverage = self.build_item_criteria_vectors()
            results["item_criteria_coverage"] = coverage
            logger.info("✓ Step 4 completed: Item-level criteria vectors built")
        except Exception as e:
            logger.error(f"✗ Step 4 failed: {str(e)}")
            raise

        # Step 5: Merge criteria vectors to splits
        try:
            self.merge_criteria_to_splits()
            logger.info("✓ Step 5 completed: Criteria merged into all splits")
        except Exception as e:
            logger.error(f"✗ Step 5 failed: {str(e)}")
            raise

        # Validation: Check merged datasets
        if validate:
            try:
                validation_results = self.validate_merged_datasets()
                results["validation_results"] = validation_results
                logger.info("✓ Validation completed: Merged splits validated")
            except Exception as e:
                logger.error(f"✗ Validation failed: {str(e)}")
                raise

        # Step 6: Build ID mappings
        try:
            mapping_stats = self.build_id_mappings()
            results["mapping_stats"] = mapping_stats
            logger.info("✓ Step 6 completed: ID mappings built")
        except Exception as e:
            logger.error(f"✗ Step 6 failed: {str(e)}")
            raise

        # Step 7: Build final model-ready splits
        try:
            model_ready_stats = self.build_model_ready_splits()
            results["model_ready_stats"] = model_ready_stats
            logger.info("✓ Step 7 completed: Model-ready splits built")
        except Exception as e:
            logger.error(f"✗ Step 7 failed: {str(e)}")
            raise

        # Final validation
        if validate:
            try:
                overlap_stats = self.validate_model_ready_splits()
                results["overlap_stats"] = overlap_stats
                logger.info("✓ Final validation completed: Model-ready splits checked")
            except Exception as e:
                logger.error(f"✗ Final validation failed: {str(e)}")
                raise

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return results


# def build_model_ready_data(config: Dict) -> Dict:
#     """
#     Convenience function to build model-ready datasets
#     """
#     builder = ModelReadyDataBuilder(config)
#     return builder.build_pipeline()


def build_model_ready_data(config: Dict, validate: bool = True) -> Dict:
    builder = ModelReadyDataBuilder(config)
    return builder.build_pipeline(validate=validate)


if __name__ == "__main__":
    import yaml

    # Load config
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Build pipeline
    results = build_model_ready_data(config)

    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE RESULTS SUMMARY")
    print("=" * 80)

    if "item_criteria_coverage" in results:
        print("\nItem Criteria Coverage:")
        print(results["item_criteria_coverage"])

    if "validation_results" in results:
        print("\nFinal Validation Results:")
        print(results["validation_results"])

    if "mapping_stats" in results:
        print("\nMapping Stats:")
        print(results["mapping_stats"])

    if "model_ready_stats" in results:
        print("\nModel-ready Stats:")
        print(results["model_ready_stats"])

    if "overlap_stats" in results:
        print("\nOverlap Stats:")
        print(results["overlap_stats"])
