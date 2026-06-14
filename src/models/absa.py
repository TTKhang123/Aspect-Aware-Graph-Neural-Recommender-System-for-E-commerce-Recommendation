# # src/absa.py

# from __future__ import annotations

# import re
# import gc
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForSequenceClassification


# class HybridABSA:
#     """
#     ABSA pseudo-label generator for multi-criteria recommendation.

#     Output:
#         crit_quality
#         crit_value
#         crit_design
#         crit_usability
#         crit_durability

#     Missing aspect => NaN, not overall rating.
#     """

#     DEFAULT_ASPECT_KEYWORDS = {
#         "quality": [
#             "quality",
#             "performance",
#             "perform",
#             "works",
#             "work",
#             "working",
#             "sound",
#             "screen",
#             "display",
#             "picture",
#             "audio",
#             "signal",
#             "reliable",
#             "solid",
#             "well made",
#             "well-made",
#             "defective",
#             "faulty",
#             "poor quality",
#             "cheap quality",
#         ],
#         "value": [
#             "price",
#             "worth",
#             "value",
#             "money",
#             "cost",
#             "deal",
#             "bargain",
#             "affordable",
#             "expensive",
#             "overpriced",
#             "cheap",
#             "waste of money",
#         ],
#         "design": [
#             "design",
#             "look",
#             "looks",
#             "appearance",
#             "style",
#             "sleek",
#             "compact",
#             "portable",
#             "size",
#             "weight",
#             "beautiful",
#             "ugly",
#             "heavy",
#             "lightweight",
#         ],
#         "usability": [
#             "easy",
#             "simple",
#             "difficult",
#             "hard",
#             "setup",
#             "install",
#             "installation",
#             "use",
#             "using",
#             "interface",
#             "user friendly",
#             "user-friendly",
#             "convenient",
#             "manual",
#             "instructions",
#         ],
#         "durability": [
#             "durable",
#             "durability",
#             "last",
#             "lasted",
#             "lasting",
#             "broke",
#             "broken",
#             "break",
#             "fragile",
#             "sturdy",
#             "dead",
#             "stopped working",
#             "failed",
#             "failure",
#         ],
#     }

#     CONTRAST_WORDS = [
#         "but",
#         "however",
#         "although",
#         "though",
#         "yet",
#         "whereas",
#         "while",
#         "except",
#         "nevertheless",
#     ]

#     def __init__(self, config: Dict):
#         self.config = config

#         absa_cfg = config.get("absa", {})

#         self.aspects = absa_cfg.get(
#             "aspects", ["quality", "value", "design", "usability", "durability"]
#         )

#         self.aspect_keywords = absa_cfg.get(
#             "aspect_keywords", self.DEFAULT_ASPECT_KEYWORDS
#         )

#         self.model_name = absa_cfg.get(
#             "model_name", "cardiffnlp/twitter-roberta-base-sentiment-latest"
#         )

#         self.max_length = absa_cfg.get("max_length", 128)
#         self.batch_size = absa_cfg.get("batch_size", 64)
#         self.min_confidence = absa_cfg.get("min_confidence", 0.45)

#         device_name = config.get("training", {}).get(
#             "device", "cuda" if torch.cuda.is_available() else "cpu"
#         )

#         self.device = torch.device(device_name)

#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
#         self.model.to(self.device)
#         self.model.eval()

#     # =========================================================
#     # TEXT PROCESSING
#     # =========================================================

#     def normalize_text(self, text: Optional[str]) -> str:
#         if text is None:
#             return ""

#         if not isinstance(text, str):
#             text = str(text)

#         text = text.replace("<br />", " ")
#         text = re.sub(r"\s+", " ", text)
#         return text.strip()

#     def split_sentences_and_clauses(self, text: str) -> List[str]:
#         text = self.normalize_text(text)

#         if not text:
#             return []

#         raw_sentences = re.split(r"[.!?]+", text)

#         clauses = []

#         contrast_pattern = r"\b(" + "|".join(self.CONTRAST_WORDS) + r")\b|;"

#         for sent in raw_sentences:
#             sent = sent.strip()

#             if not sent:
#                 continue

#             parts = re.split(contrast_pattern, sent, flags=re.IGNORECASE)

#             cleaned_parts = []

#             for p in parts:
#                 if p is None:
#                     continue

#                 p = p.strip()

#                 if not p:
#                     continue

#                 if p.lower() in self.CONTRAST_WORDS:
#                     continue

#                 cleaned_parts.append(p)

#             for p in cleaned_parts:
#                 if len(p.split()) >= 2:
#                     clauses.append(p)

#         return clauses

#     def detect_aspects_in_clause(self, clause: str) -> List[str]:
#         clause_lower = clause.lower()
#         detected = []

#         for aspect in self.aspects:
#             keywords = self.aspect_keywords.get(aspect, [])

#             for kw in keywords:
#                 kw_lower = kw.lower()

#                 if " " in kw_lower:
#                     if kw_lower in clause_lower:
#                         detected.append(aspect)
#                         break
#                 else:
#                     if re.search(rf"\b{re.escape(kw_lower)}\b", clause_lower):
#                         detected.append(aspect)
#                         break

#         return detected

#     def extract_aspect_contexts(self, review_text: str) -> Dict[str, List[str]]:
#         contexts = {aspect: [] for aspect in self.aspects}

#         clauses = self.split_sentences_and_clauses(review_text)

#         for clause in clauses:
#             detected_aspects = self.detect_aspects_in_clause(clause)

#             for aspect in detected_aspects:
#                 contexts[aspect].append(clause)

#         return contexts

#     # =========================================================
#     # SENTIMENT MODEL
#     # =========================================================

#     def _logits_to_score_and_confidence(
#         self, logits: torch.Tensor
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

#         num_labels = probs.shape[1]

#         if num_labels == 3:
#             # common mapping: negative, neutral, positive
#             # score range: 1 to 5
#             weights = np.array([1.0, 3.0, 5.0])
#             scores = probs @ weights
#             confidence = probs.max(axis=1)

#         elif num_labels == 5:
#             weights = np.array([1, 2, 3, 4, 5], dtype=float)
#             scores = probs @ weights
#             confidence = probs.max(axis=1)

#         else:
#             pred = probs.argmax(axis=1)
#             scores = 1.0 + 4.0 * pred / max(1, num_labels - 1)
#             confidence = probs.max(axis=1)

#         return scores.astype(float), confidence.astype(float)

#     def predict_sentiment_batch(self, texts: List[str]) -> List[Tuple[float, float]]:
#         if not texts:
#             return []

#         results = []

#         for start in range(0, len(texts), self.batch_size):
#             batch_texts = texts[start : start + self.batch_size]

#             inputs = self.tokenizer(
#                 batch_texts,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True,
#                 max_length=self.max_length,
#             )

#             inputs = {k: v.to(self.device) for k, v in inputs.items()}

#             with torch.no_grad():
#                 outputs = self.model(**inputs)

#             scores, confidences = self._logits_to_score_and_confidence(outputs.logits)

#             results.extend(list(zip(scores, confidences)))

#         return results

#     # =========================================================
#     # REVIEW-LEVEL ABSA
#     # =========================================================

#     def extract_review_aspect_scores(self, review_text: str) -> Dict[str, float]:
#         contexts = self.extract_aspect_contexts(review_text)

#         all_texts = []
#         index_map = []

#         for aspect, clauses in contexts.items():
#             for clause in clauses:
#                 all_texts.append(clause)
#                 index_map.append(aspect)

#         aspect_values = {
#             aspect: {"scores": [], "confidences": []} for aspect in self.aspects
#         }

#         predictions = self.predict_sentiment_batch(all_texts)

#         for aspect, (score, confidence) in zip(index_map, predictions):
#             if confidence >= self.min_confidence:
#                 aspect_values[aspect]["scores"].append(score)
#                 aspect_values[aspect]["confidences"].append(confidence)

#         output = {}

#         for aspect in self.aspects:
#             scores = np.array(aspect_values[aspect]["scores"], dtype=float)
#             confidences = np.array(aspect_values[aspect]["confidences"], dtype=float)

#             if len(scores) == 0:
#                 output[aspect] = np.nan
#             else:
#                 output[aspect] = float(
#                     np.sum(scores * confidences) / np.sum(confidences)
#                 )

#         return output

#     def process_dataframe(
#         self, df: pd.DataFrame, text_col: str = "review_text"
#     ) -> pd.DataFrame:
#         records = []

#         for _, row in tqdm(df.iterrows(), total=len(df), desc="ABSA"):
#             text = row.get(text_col, "")

#             aspect_scores = self.extract_review_aspect_scores(text)

#             record = {}

#             for aspect in self.aspects:
#                 record[f"crit_{aspect}"] = aspect_scores.get(aspect, np.nan)

#             records.append(record)

#         return pd.DataFrame(records, index=df.index)

#     # =========================================================
#     # ITEM-LEVEL AGGREGATION
#     # =========================================================

#     def aggregate_item_scores(
#         self, df: pd.DataFrame, item_col: str = "item_id"
#     ) -> pd.DataFrame:
#         crit_cols = [f"crit_{aspect}" for aspect in self.aspects]

#         agg_df = df.groupby(item_col)[crit_cols].mean().reset_index()

#         count_df = df.groupby(item_col)[crit_cols].count().reset_index()

#         count_df = count_df.rename(columns={col: f"{col}_count" for col in crit_cols})

#         return agg_df.merge(count_df, on=item_col, how="left")

#     # =========================================================
#     # PARQUET PIPELINE
#     # =========================================================

#     def process_parquet_file(
#         self,
#         input_path: str | Path,
#         output_path: str | Path,
#         text_col: str = "review_text",
#         chunk_size: int = 50_000,
#     ) -> None:
#         input_path = Path(input_path)
#         output_path = Path(output_path)

#         output_path.parent.mkdir(parents=True, exist_ok=True)

#         import pyarrow.parquet as pq

#         parquet_file = pq.ParquetFile(input_path)
#         first_write = True

#         for batch in tqdm(
#             parquet_file.iter_batches(batch_size=chunk_size),
#             total=parquet_file.num_row_groups,
#             desc=f"Processing {input_path.name}",
#         ):
#             df_chunk = batch.to_pandas()

#             absa_df = self.process_dataframe(df_chunk, text_col=text_col)

#             out_chunk = pd.concat(
#                 [df_chunk.reset_index(drop=True), absa_df.reset_index(drop=True)],
#                 axis=1,
#             )

#             if first_write:
#                 out_chunk.to_parquet(output_path, index=False)
#                 first_write = False
#             else:
#                 existing = pd.read_parquet(output_path)
#                 combined = pd.concat([existing, out_chunk], ignore_index=True)
#                 combined.to_parquet(output_path, index=False)
#                 del existing, combined

#             del df_chunk, absa_df, out_chunk
#             gc.collect()

#     def process_parquet_to_chunks(
#         self,
#         input_path: str | Path,
#         output_dir: str | Path,
#         text_col: str = "review_text",
#         chunk_size: int = 50_000,
#     ) -> None:
#         """
#         Safer for large files:
#         input.parquet -> many output chunks.
#         """

#         input_path = Path(input_path)
#         output_dir = Path(output_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)

#         import pyarrow.parquet as pq

#         parquet_file = pq.ParquetFile(input_path)

#         chunk_id = 0

#         for batch in tqdm(
#             parquet_file.iter_batches(batch_size=chunk_size),
#             desc=f"ABSA chunks from {input_path.name}",
#         ):
#             df_chunk = batch.to_pandas()

#             absa_df = self.process_dataframe(df_chunk, text_col=text_col)

#             out_chunk = pd.concat(
#                 [df_chunk.reset_index(drop=True), absa_df.reset_index(drop=True)],
#                 axis=1,
#             )

#             out_path = output_dir / f"absa_chunk_{chunk_id:04d}.parquet"
#             out_chunk.to_parquet(out_path, index=False)

#             print(f"Saved {out_path} | rows={len(out_chunk):,}")

#             chunk_id += 1

#             del df_chunk, absa_df, out_chunk
#             gc.collect()


# def build_default_absa_config() -> Dict:
#     return {
#         "training": {"device": "cuda" if torch.cuda.is_available() else "cpu"},
#         "absa": {
#             "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
#             "aspects": ["quality", "value", "design", "usability", "durability"],
#             "max_length": 128,
#             "batch_size": 64,
#             "min_confidence": 0.45,
#         },
#     }


# # if __name__ == "__main__":
# #     config = build_default_absa_config()

# #     absa = HybridABSA(config)

# #     sample = pd.DataFrame(
# #         {
# #             "item_id": ["A1", "A1", "A2"],
# #             "review_text": [
# #                 "The build quality is excellent but the price is too expensive.",
# #                 "Easy to install and very durable. Great value for money.",
# #                 "Looks beautiful but stopped working after two days.",
# #             ],
# #             "rating": [4.0, 5.0, 2.0],
# #         }
# #     )

# #     result = absa.process_dataframe(sample)
# #     print(pd.concat([sample, result], axis=1))


##############################################################################
#### ver 3

from __future__ import annotations

import re
import gc
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_num_threads(2)


class HybridABSA:
    DEFAULT_ASPECT_KEYWORDS = {
        "quality": [
            "quality",
            "performance",
            "perform",
            "works",
            "work",
            "working",
            "sound",
            "screen",
            "display",
            "picture",
            "audio",
            "signal",
            "reliable",
            "solid",
            "well made",
            "well-made",
            "defective",
            "faulty",
            "poor quality",
            "cheap quality",
        ],
        "value": [
            "price",
            "worth",
            "value",
            "money",
            "cost",
            "deal",
            "bargain",
            "affordable",
            "expensive",
            "overpriced",
            "cheap",
            "waste of money",
        ],
        "design": [
            "design",
            "look",
            "looks",
            "appearance",
            "style",
            "sleek",
            "compact",
            "portable",
            "size",
            "weight",
            "beautiful",
            "ugly",
            "heavy",
            "lightweight",
        ],
        "usability": [
            "easy",
            "simple",
            "difficult",
            "hard",
            "setup",
            "install",
            "installation",
            "use",
            "using",
            "interface",
            "user friendly",
            "user-friendly",
            "convenient",
            "manual",
            "instructions",
        ],
        "durability": [
            "durable",
            "durability",
            "last",
            "lasted",
            "lasting",
            "broke",
            "broken",
            "break",
            "fragile",
            "sturdy",
            "dead",
            "stopped working",
            "failed",
            "failure",
        ],
    }

    CONTRAST_WORDS = [
        "but",
        "however",
        "although",
        "though",
        "yet",
        "whereas",
        "while",
        "except",
        "nevertheless",
    ]

    def __init__(self, config: Dict):
        self.config = config
        absa_cfg = config.get("absa", {})

        self.aspects = absa_cfg.get(
            "aspects",
            ["quality", "value", "design", "usability", "durability"],
        )

        self.aspect_keywords = absa_cfg.get(
            "aspect_keywords",
            self.DEFAULT_ASPECT_KEYWORDS,
        )

        self.model_name = absa_cfg.get(
            "model_name",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
        )

        self.max_length = absa_cfg.get("max_length", 128)
        self.batch_size = absa_cfg.get("batch_size", 128)  #### 64 ver 3
        self.min_confidence = absa_cfg.get("min_confidence", 0.45)

        device_name = config.get("training", {}).get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )

        self.device = torch.device(device_name)

        print("=" * 60)
        print("Device:", self.device)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        print("Batch size:", self.batch_size)
        print("=" * 60)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def normalize_text(self, text: Optional[str]) -> str:
        if text is None:
            return ""

        if not isinstance(text, str):
            text = str(text)

        text = text.replace("<br />", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def split_sentences_and_clauses(self, text: str) -> List[str]:
        text = self.normalize_text(text)

        if not text:
            return []

        raw_sentences = re.split(r"[.!?]+", text)
        clauses = []

        contrast_pattern = r"\b(" + "|".join(self.CONTRAST_WORDS) + r")\b|;"

        for sent in raw_sentences:
            sent = sent.strip()

            if not sent:
                continue

            parts = re.split(contrast_pattern, sent, flags=re.IGNORECASE)
            cleaned_parts = []

            for p in parts:
                if p is None:
                    continue

                p = p.strip()

                if not p:
                    continue

                if p.lower() in self.CONTRAST_WORDS:
                    continue

                cleaned_parts.append(p)

            for p in cleaned_parts:
                if len(p.split()) >= 2:
                    clauses.append(p)

        return clauses

    def detect_aspects_in_clause(self, clause: str) -> List[str]:
        clause_lower = clause.lower()
        detected = []

        for aspect in self.aspects:
            keywords = self.aspect_keywords.get(aspect, [])

            for kw in keywords:
                kw_lower = kw.lower()

                if " " in kw_lower:
                    if kw_lower in clause_lower:
                        detected.append(aspect)
                        break
                else:
                    if re.search(rf"\b{re.escape(kw_lower)}\b", clause_lower):
                        detected.append(aspect)
                        break

        return detected

    def _logits_to_score_and_confidence(
        self,
        logits: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        num_labels = probs.shape[1]

        if num_labels == 3:
            weights = np.array([1.0, 3.0, 5.0])
            scores = probs @ weights
            confidence = probs.max(axis=1)

        elif num_labels == 5:
            weights = np.array([1, 2, 3, 4, 5], dtype=float)
            scores = probs @ weights
            confidence = probs.max(axis=1)

        else:
            pred = probs.argmax(axis=1)
            scores = 1.0 + 4.0 * pred / max(1, num_labels - 1)
            confidence = probs.max(axis=1)

        return scores.astype(float), confidence.astype(float)

    def predict_sentiment_batch(self, texts: List[str]) -> List[Tuple[float, float]]:
        if not texts:
            return []

        results = []

        for start in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Sentiment batches",
            leave=False,
            dynamic_ncols=True,
        ):
            batch_texts = texts[start : start + self.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                if self.device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

            scores, confidences = self._logits_to_score_and_confidence(outputs.logits)
            results.extend(list(zip(scores, confidences)))

        return results

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "review_text",
    ) -> pd.DataFrame:
        texts = df[text_col].fillna("").astype(str).tolist()
        n = len(texts)

        all_clauses = []
        meta = []

        for review_idx, text in tqdm(
            enumerate(texts),
            total=n,
            desc="Extract clauses",
            leave=False,
            dynamic_ncols=True,
        ):
            clauses = self.split_sentences_and_clauses(text)

            for clause in clauses:
                detected_aspects = self.detect_aspects_in_clause(clause)

                for aspect in detected_aspects:
                    all_clauses.append(clause)
                    meta.append((review_idx, aspect))

        aspect_values = [
            {aspect: {"scores": [], "confidences": []} for aspect in self.aspects}
            for _ in range(n)
        ]

        if all_clauses:
            predictions = self.predict_sentiment_batch(all_clauses)

            for (review_idx, aspect), (score, confidence) in zip(meta, predictions):
                if confidence >= self.min_confidence:
                    aspect_values[review_idx][aspect]["scores"].append(score)
                    aspect_values[review_idx][aspect]["confidences"].append(confidence)

        records = []

        for review_idx in range(n):
            record = {}

            for aspect in self.aspects:
                scores = np.array(
                    aspect_values[review_idx][aspect]["scores"],
                    dtype=float,
                )
                confidences = np.array(
                    aspect_values[review_idx][aspect]["confidences"],
                    dtype=float,
                )

                if len(scores) == 0:
                    record[f"crit_{aspect}"] = np.nan
                else:
                    record[f"crit_{aspect}"] = float(
                        np.sum(scores * confidences) / np.sum(confidences)
                    )

            records.append(record)

        return pd.DataFrame(records, index=df.index)

    def extract_review_aspect_scores(self, review_text: str) -> Dict[str, float]:
        temp_df = pd.DataFrame({"review_text": [review_text]})
        result = self.process_dataframe(temp_df, text_col="review_text")
        return {aspect: result.loc[0, f"crit_{aspect}"] for aspect in self.aspects}

    def aggregate_item_scores(
        self,
        df: pd.DataFrame,
        item_col: str = "item_id",
    ) -> pd.DataFrame:
        crit_cols = [f"crit_{aspect}" for aspect in self.aspects]

        agg_df = df.groupby(item_col)[crit_cols].mean().reset_index()
        count_df = df.groupby(item_col)[crit_cols].count().reset_index()

        count_df = count_df.rename(columns={col: f"{col}_count" for col in crit_cols})

        return agg_df.merge(count_df, on=item_col, how="left")

    def process_parquet_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        text_col: str = "review_text",
        chunk_size: int = 50_000,
    ) -> None:
        input_path = Path(input_path)
        output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(input_path)
        first_write = True

        for batch in tqdm(
            parquet_file.iter_batches(batch_size=chunk_size),
            desc=f"Processing {input_path.name}",
            dynamic_ncols=True,
        ):
            df_chunk = batch.to_pandas()

            absa_df = self.process_dataframe(df_chunk, text_col=text_col)

            out_chunk = pd.concat(
                [df_chunk.reset_index(drop=True), absa_df.reset_index(drop=True)],
                axis=1,
            )

            if first_write:
                out_chunk.to_parquet(output_path, index=False)
                first_write = False
            else:
                existing = pd.read_parquet(output_path)
                combined = pd.concat([existing, out_chunk], ignore_index=True)
                combined.to_parquet(output_path, index=False)
                del existing, combined
            del df_chunk, absa_df, out_chunk
            gc.collect()

    def process_parquet_to_chunks(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        text_col: str = "review_text",
        chunk_size: int = 50_000,
    ) -> None:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(input_path)
        chunk_id = 0

        for batch in tqdm(
            parquet_file.iter_batches(batch_size=chunk_size),
            desc="Chunks",
            dynamic_ncols=True,
        ):
            df_chunk = batch.to_pandas()

            absa_df = self.process_dataframe(df_chunk, text_col=text_col)

            out_chunk = pd.concat(
                [df_chunk.reset_index(drop=True), absa_df.reset_index(drop=True)],
                axis=1,
            )

            out_path = output_dir / f"absa_chunk_{chunk_id:04d}.parquet"
            out_chunk.to_parquet(out_path, index=False)

            tqdm.write(f"Saved {out_path} | rows={len(out_chunk):,}")

            chunk_id += 1

            del df_chunk, absa_df, out_chunk
            gc.collect()


def build_default_absa_config() -> Dict:
    return {
        "training": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "absa": {
            "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "aspects": ["quality", "value", "design", "usability", "durability"],
            "max_length": 128,
            "batch_size": 64,
            "min_confidence": 0.45,
        },
    }
