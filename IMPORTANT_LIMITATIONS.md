# Important limitations and honest claims

This project contains a real hybrid recommender implementation, but the system does not support every claim in the proposal document.

## 1. What the code actually implements

The current repository implements:

- ABSA feature extraction from review text via `scripts/run_absa.py` and `src/models/absa.py`.
- Model-ready data construction from ABSA/criteria features in `scripts/build_model_ready_data.py` and `src/data/model_ready_data.py`.
- A LightGCN-based ranking model with aspect-aware semantic fusion in `src/models/aspect_gcn/model.py`.
- Training and evaluation of Aspect LightGCN variants in `scripts/train_aspect_lightgcn.py`.
- Ranking baseline evaluation for MF/NCF/DeepFM in `scripts/evaluate_ranking_baselines.py`.
- Known-user inference with aspect explanations in `scripts/recommend.py` and `src/inference/aspect_lightgcn_recommender.py`.
- A separate cold-item user-retrieval pipeline using metadata embeddings and LSH in `src/inference/cold_item_content_recommender.py` and `scripts/build_user_content_lsh_index.py`.

## 2. What the system can do

- Train and evaluate collaborative filtering baselines and Aspect LightGCN variants.
- Use review-derived aspect scores, masks, and support signals as static item features.
- Evaluate warm splits (`val_warm` / `test_warm`) and full splits (`val_full` / `test_full`).
- Provide known-user top-K recommendations with per-item aspect scores and a simple explanation string.
- Fallback to popularity and aspect-criteria based recommendations for unknown user IDs.
- Build a content-based LSH index for new-item user routing when metadata exists.

## 3. Cold-start support — what is real and what is not

### Cold users

- `scripts/recommend.py` does not perform model-based recommendation for a truly unseen `user_id`.
- If the input `user_id` is missing from `data/processed/model_ready/user2idx.parquet`, the code returns:
  - top popular items,
  - top items by each aspect criterion.
- That is a heuristic fallback, not a learned cold-user recommendation model.

### Cold items

- The main `AspectLightGCN` model is trained with item features aggregated from training data only.
- True new items with no training interactions receive no learned ABSA semantic signals and effectively have zero item feature mask in the LightGCN model.
- Cold-item generalization in the main model is therefore weak.
- The separate cold-item pipeline (`scripts/recommend_users_for_new_item.py`) is a content-based user retrieval approach, not a unified end-to-end cold-item ranking model.
- It requires item metadata and a prebuilt LSH index, so it is only valid if metadata is available before recommendation time.

### Cold-start evaluation nuance

- The full evaluation splits (`val_full` / `test_full`) can include cold users and/or cold items, but the code only evaluates them as part of the existing `LightGCN` embedding space.
- The model does not fully solve cold-start in the open-world sense of unknown user IDs + unknown item IDs simultaneously.
- Inference for unknown users is handled by heuristics, not by the trained graph model.

## 4. Evaluation limitations

- Evaluation is performed with sampled negative items (`num_negatives_eval`) and top-K ranking metrics on candidate subsets.
- This means reported `Recall@K` / `NDCG@K` are approximate, not full-candidate exhaustive ranking metrics.
- The model training and evaluation use warm-item negative sampling, so cold items are not adequately trained as negatives.
- The observed full-split metrics therefore measure robustness under a restricted protocol, not perfect cold-start ranking.

## 5. ABSA and semantic feature limitations

- The ABSA component relies on keyword-based aspect detection and a pretrained sentiment classifier.
- It extracts aspect clauses from reviews and averages sentiment scores per review/item.
- This is a heuristic semantic pipeline, so aspect labels and sentiment strengths may be noisy.
- The item semantics are static and derived from historical reviews; they are not updated online per new review.

## 6. Honest claims you can make

Valid claims include:

- The repository builds a hybrid recommendation pipeline combining LightGCN collaborative filtering with aspect-aware item semantic features.
- It evaluates warm and full splits, showing performance under more challenging cold-start conditions.
- It provides a known-user inference path with aspect-based explanations and a cold-user fallback strategy.
- It includes a separate content-based cold-item user retrieval pipeline using item metadata and LSH.

## 7. Do not claim

Do not claim:

- state-of-the-art performance,
- a full solution for cold-start users/items at inference time,
- that the model fully handles new users and new items simultaneously,
- that evaluation uses exhaustive ranking over all items,
- that ABSA-derived aspect signals are perfect or fully interpretable.

## 8. Practical guidance

- Treat the full-split evaluation as a robustness check, not proof of open-world cold-start capability.
- Treat the unknown-user path as a heuristic fallback.
- Treat the cold-item LSH route as separate content-based routing that requires metadata.
- Keep ranking metrics (`Recall@K`, `NDCG@K`) as the primary reported results, and avoid over-emphasizing RMSE/MAE.
