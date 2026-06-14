# Aspect-Aware Graph Neural Recommender System for Cold-Start E-commerce Recommendation

A practical, reproducible recommendation research pipeline that combines:

- LightGCN collaborative filtering
- Aspect-based sentiment analysis (ABSA) from review text
- Gated fusion of collaborative and semantic item embeddings
- Warm/cold split evaluation for cold-start robustness
- Ranking and beyond-accuracy metrics

The repository is designed for e-commerce review datasets, such as Amazon Electronics, and supports end-to-end execution from raw data to inference.

---

## What this project implements

- LightGCN collaborative filtering backbone
- Item-level aspect features extracted from ABSA review analysis
- Gated semantic residual fusion of CF and aspect embeddings
- Pairwise ranking training with BPR loss
- Warm and full split evaluation for cold-start research
- Ranking metrics: `Recall@K`, `Precision@K`, `NDCG@K`, `HitRate@K`, `MRR@K`
- Beyond-accuracy metrics: `Coverage@K`, `Novelty@K`, `Diversity@K`
- Data preprocessing for model-ready parquet datasets
- Explainable inference with aspect-based recommendations

---

## Project structure

```
.
├── app/                          # optional Streamlit demo pages
├── config/                       # canonical YAML configuration
├── data/                         # raw and processed datasets
├── doc/                          # documentation and thesis material
├── eda/                          # exploratory data analysis outputs
├── notebooks/                    # Jupyter notebooks
├── outputs/                      # saved checkpoints, metrics, and index files
├── scripts/                      # CLI entrypoints for preprocessing, training, and inference
│   ├── pipeline.py
│   ├── run_absa.py
│   ├── build_model_ready_data.py
│   ├── train_aspect_lightgcn.py
│   ├── train_ranking_baselines.py
│   ├── recommend.py
│   ├── build_user_content_lsh_index.py
│   ├── recommend_users_for_new_item.py
│   ├── evaluate_ranking_baselines.py
│   ├── inspect_parquet_columns.py
│   ├── run_kaggle.sh
│   └── train.py
├── src/                          # implementation code
│   ├── data/
│   │   ├── data_loader.py
│   │   └── model_ready_data.py
│   ├── inference/
│   │   ├── aspect_lightgcn_recommender.py
│   │   ├── cold_item_content_recommender.py
│   │   ├── content_lsh.py
│   │   └── content_profile.py
│   ├── models/
│   │   ├── absa.py
│   │   ├── mcf.py
│   │   └── aspect_gcn/
│   ├── training/
│   │   ├── evaluator.py
│   │   ├── ranking_evaluator.py
│   │   ├── ranking_trainer.py
│   │   └── trainer.py
│   └── utils/
│       └── __init__.py
├── test/                         # unit tests
├── README.md                     # project overview
├── requirements.txt              # Python dependencies
├── run_flow.txt                  # execution notes and workflow
└── IMPORTANT_LIMITATIONS.md      # limitations and caveats
```

---

## Setup

Create and activate the Python environment, then install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

---

## Recommended command flow

The repository should be executed using module-style entrypoints: `python -m scripts.<name>`. This ensures consistent package imports and avoids path issues.

> `run_flow.txt` includes outdated examples and a typo (`buil_model_ready_data.py`). Prefer the commands below.

### 1. Data preprocessing

```bash
python -m scripts.pipeline --config config/config.yaml
```

Options:

- `--config` — path to YAML config file
- `--review-limit` — limit review rows for fast testing
- `--metadata-limit` — limit metadata rows for fast testing
- `--review-chunk-size` — review parquet chunk size (default `300000`)
- `--metadata-chunk-size` — metadata parquet chunk size (default `100000`)
- `--no-clear-temp` — keep existing temporary chunk files

### 2. Run ABSA

```bash
python -m scripts.run_absa --config config/config.yaml --input data/processed/train.parquet --output-dir data/processed/absa_train_chunks
```

Options:

- `--config` — path to YAML config file
- `--input` — input parquet file (default `data/processed/train.parquet`)
- `--output-dir` — output folder for ABSA chunks
- `--text-col` — review text column name (default `review_text`)
- `--chunk-size` — chunk size for parquet processing (default `20000`)
- `--mode` — `chunks` or `single` output mode

### 3. Build model-ready data

```bash
python -m scripts.build_model_ready_data --config config/config.yaml
```

Options:

- `--config` — path to YAML config file
- `--validate` — enable dataset validation (default enabled)
- `--no-validate` — skip validation

### 4. Train Aspect LightGCN variants

```bash
python -m scripts.train_aspect_lightgcn --config config/config.yaml --ablation all
```

Options:

- `--config` — path to YAML config file
- `--ablation` — one of `lightgcn_only`, `lightgcn_aspects`, `lightgcn_aspects_masks`, `lightgcn_aspects_masks_support`, or `all`

### 5. Train ranking baselines

```bash
python -m scripts.train_ranking_baselines --config config/config.yaml --model all
```

Options:

- `--config` — path to YAML config file
- `--model` — one of `mf_bpr`, `ncf_bpr`, `deepfm_bpr`, or `all`

### 6. Recommendation inference

Generate a sample list of known warm users first:

```bash
python -m scripts.recommend --config config/config.yaml --checkpoint outputs/aspect_lightgcn/lightgcn_aspects_masks_support/best_model.pt --show-sample-users
```

This command prints a sample list of users that exist in the warm-user mapping, so you can choose a valid `--user-id` for recommendation.

Then run recommendation for a chosen user:

```bash
python -m scripts.recommend --config config/config.yaml --checkpoint outputs/aspect_lightgcn/lightgcn_aspects_masks_support/best_model.pt --user-id <USER_ID> --top-k 10
```

Behavior differs based on whether the user is known or cold-start:

- Known user (`user_id` exists in the training/user mapping):
  - returns top-`k` personalized recommendations
  - includes aspect-based scores and explanation per item
- Cold-start user (`user_id` is not found in the mapping):
  - falls back to popularity-based recommendations
  - additionally returns top items by each aspect criterion
  - this is a heuristic fallback, not a learned cold-user model

Options:

- `--config` — path to YAML config file
- `--checkpoint` — model checkpoint path
- `--user-id` — user ID for recommendation
- `--top-k` — number of recommended items (default `10`)
- `--device` — device name, default `cuda`
- `--metadata` — metadata parquet file path, default `data/processed/item_clean_min5.parquet`
- `--active-ablation` — active ablation variant name
- `--show-sample-users` — print sample warm users and exit

### 7. Cold-start item recommendation

Build an item-content LSH index first:

```bash
python -m scripts.build_user_content_lsh_index --train-path data/processed/model_ready/train_model.parquet --metadata data/processed/metadata_filtered.parquet --output-dir outputs/content_lsh
```

This step does the following:

- loads positive item interactions from the model-ready training data
- loads item metadata and builds content text profiles
- encodes item metadata with a SentenceTransformer model
- creates user content profiles by averaging embeddings of positively rated items per user
- fits a Random Hyperplane LSH index over user content profiles

Then recommend users for a new cold-start item:

```bash
python -m scripts.recommend_users_for_new_item --item-json examples/new_item.json --index-dir outputs/content_lsh --top-k-users 20
```

This command encodes the new item metadata into the same content embedding space, then returns the top users whose historical item preferences are most similar to the new item. It is a cold-start item strategy, not an item ranking from the model. The cold-item route is user retrieval with metadata similarity, not a unified cold-item item-ranking model.

Options for `build_user_content_lsh_index`:

- `--train-path` — model-ready train parquet path
- `--metadata` — item metadata parquet path
- `--output-dir` — output dir for LSH outputs
- `--embedding-model` — sentence-transformers model name
- `--min-rating` — minimum positive rating threshold
- `--max-items` — maximum number of items to encode
- `--batch-size` — embedding batch size
- `--n-planes`, `--n-tables` — LSH index parameters
- `--seed` — random seed

Options for `recommend_users_for_new_item`:

- `--item-json` — JSON file for new item metadata
- `--index-dir` — LSH index directory
- `--top-k-users` — number of similar users to retrieve

---

## Configuration reference

The main config is `config/config.yaml`.

Important sections:

- `aspect_lightgcn.paths` — model-ready and output directories
- `aspect_lightgcn.columns` — user/item/rating/aspect columns
- `aspect_lightgcn.ablation` — ablation variant definitions
- `aspect_lightgcn.data` — evaluation settings such as `max_eval_users` and `num_negatives_eval`
- `aspect_lightgcn.model` — LightGCN dimensions, fusion, and dropout

---

## Model comparison

| Model family      | Model / variant                  | Purpose                       | Description                                                                    |
| ----------------- | -------------------------------- | ----------------------------- | ------------------------------------------------------------------------------ |
| Aspect LightGCN   | `lightgcn_only`                  | CF baseline                   | LightGCN with no aspect features, tests collaborative-only performance         |
| Aspect LightGCN   | `lightgcn_aspects`               | Aspect-aware variant          | Uses item aspect scores but no masks/support, tests semantic features alone    |
| Aspect LightGCN   | `lightgcn_aspects_masks`         | Masked aspect variant         | Adds aspect masks to suppress weak aspect signals                              |
| Aspect LightGCN   | `lightgcn_aspects_masks_support` | Full proposal                 | Uses aspect scores, masks, and support features for the strongest fusion model |
| Ranking baselines | `mf_bpr`                         | Matrix factorization baseline | Standard BPR-trained MF on model-ready train data                              |
| Ranking baselines | `ncf_bpr`                        | Neural CF baseline            | Neural Collaborative Filtering with BPR ranking loss                           |
| Ranking baselines | `deepfm_bpr`                     | Hybrid deep-learning baseline | DeepFM variant trained with BPR for item ranking                               |

---

## Evaluation results

These tables use metrics extracted from the existing `outputs/` artifacts.

### Warm vs full split explanation

The repository evaluates both warm-only and full evaluation splits.

- `val_warm` / `test_warm` are warm-only splits where users and items are already present in the training mapping.
- `val_full` / `test_full` are full splits that include cold users and cold items for a more challenging generalization test.

Current dataset ratios show that the full splits still contain mostly in-training entities, but with a meaningful cold portion:

- `val_full`: `is_train_user` ≈ 0.8328, `is_train_item` ≈ 0.9660
- `test_full`: `is_train_user` ≈ 0.8034, `is_train_item` ≈ 0.9394

This means the full evaluation includes about 16–20% cold users and 4–6% cold items, so `val_full`/`test_full` better measures cold-start robustness.

### Ranking baseline performance — warm splits

| Baseline     | val_warm hit@10 | test_warm hit@10 | val_warm ndcg@10 | test_warm ndcg@10 |
| ------------ | --------------- | ---------------- | ---------------- | ----------------- |
| `mf_bpr`     | 0.7180          | 0.6740           | 0.4248           | 0.3934            |
| `ncf_bpr`    | 0.7330          | 0.6790           | 0.4239           | 0.3900            |
| `deepfm_bpr` | 0.7250          | 0.6650           | 0.4280           | 0.3887            |

### Ranking baseline performance — full splits

| Baseline     | val_full hit@10 | test_full hit@10 | val_full ndcg@10 | test_full ndcg@10 |
| ------------ | --------------- | ---------------- | ---------------- | ----------------- |
| `mf_bpr`     | 0.7280          | 0.6440           | 0.4292           | 0.3656            |
| `ncf_bpr`    | 0.7330          | 0.6460           | 0.4270           | 0.3623            |
| `deepfm_bpr` | 0.7285          | 0.6455           | 0.4246           | 0.3644            |

### Aspect LightGCN ablation comparison — warm splits

| Variant                          | val_warm hit@10 | test_warm hit@10 | val_warm ndcg@10 | test_warm ndcg@10 |
| -------------------------------- | --------------- | ---------------- | ---------------- | ----------------- |
| `lightgcn_only`                  | 0.7415          | 0.6495           | 0.4428           | 0.3891            |
| `lightgcn_aspects`               | 0.7090          | 0.6175           | 0.4097           | 0.3560            |
| `lightgcn_aspects_masks`         | 0.6800          | 0.5755           | 0.3946           | 0.3415            |
| `lightgcn_aspects_masks_support` | 0.7450          | 0.6495           | 0.4374           | 0.3768            |

### Aspect LightGCN ablation comparison — full splits

| Variant                          | val_full hit@10 | test_full hit@10 | val_full ndcg@10 | test_full ndcg@10 |
| -------------------------------- | --------------- | ---------------- | ---------------- | ----------------- |
| `lightgcn_only`                  | 0.7335          | 0.6125           | 0.4213           | 0.3364            |
| `lightgcn_aspects`               | 0.7165          | 0.6130           | 0.4088           | 0.3346            |
| `lightgcn_aspects_masks`         | 0.7065          | 0.6205           | 0.4034           | 0.3359            |
| `lightgcn_aspects_masks_support` | 0.7445          | 0.6475           | 0.4399           | 0.3624            |

`lightgcn_aspects_masks_support` remains the strongest available variant on the full-split test set, with the highest hit@10 and ndcg@10 among the recorded ablations.

---

## Notes

- `run_flow.txt` is a useful reference, but it contains older command examples and a typo (`buil_model_ready_data.py`).
- The correct execution pattern is `python -m scripts.<script>`.
- `config/config.yaml` is the repository's source of truth.
- ABSA output chunks are aggregated into item-level aspect scores by the model-ready builder.

---

## References

- LightGCN: https://arxiv.org/abs/2002.02126
- Neural Collaborative Filtering: https://arxiv.org/abs/1708.05031
- Amazon Review Data 2018: https://nijianmo.github.io/amazon/index.html

---

## Acknowledgements

Built as part of a cold-start e-commerce recommendation research pipeline with ABSA-enhanced LightGCN.
