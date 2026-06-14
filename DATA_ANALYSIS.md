# Data Analysis Report

This report summarizes the raw dataset and existing EDA artifacts for the Amazon Electronics dataset used in this project.

## Data sources

- `data/raw/Electronics.json.gz` — raw review records
- `data/raw/meta_Electronics.json.gz` — raw item metadata
- `eda/phase1_dataset_summary.csv`
- `eda/rating_distribution.csv`
- `eda/reviews_by_year.csv`
- `eda/top_brands.csv`
- `eda/top_categories.csv`

## Summary of raw review data

The full review dataset contains 20,994,353 records.

### Key statistics

- Total review records: 20,994,353
- Unique users: 9,838,676
- Unique items: 756,489
- Rating distribution:
  - 1 star: 2,415,650 (11.5%)
  - 2 stars: 1,139,589 (5.4%)
  - 3 stars: 1,529,818 (7.3%)
  - 4 stars: 3,306,379 (15.8%)
  - 5 stars: 12,602,917 (60.1%)
- Verified review distribution:
  - `True`: 18,597,092
  - `False`: 2,397,261
- Missing text fields:
  - `reviewText`: 9,731 records
  - `summary`: 4,839 records
- Duplicate records in first 200,000 rows by `(reviewerID, asin, unixReviewTime)`: 284

### Distribution details

- User review counts:
  - min: 1 review per user
  - max: 633 reviews per user
  - mean: 2.13 reviews per user
  - quantiles: 25% = 1, 50% = 1, 75% = 2, 90% = 4, 95% = 6, 99% = 15
- Item review counts:
  - min: 1 review per item
  - max: 28,539 reviews per item
  - mean: 27.75 reviews per item
  - quantiles: 25% = 1, 50% = 3, 75% = 10, 90% = 40, 95% = 92, 99% = 449
- Most-reviewed items:
  - `B010OYASRG`: 28,539 reviews
  - `B00L0YLRUW`: 20,873 reviews
  - `B00DIF2BO2`: 17,045 reviews
  - `B006GWO5WK`: 16,130 reviews
  - `B003L1ZYYW`: 16,056 reviews

### Observations

- The dataset is extremely sparse: most users have only one or two reviews.
- User-item interactions are long-tailed, with a small number of highly-reviewed products and many single-review items.
- The rating distribution is heavily skewed toward positive feedback: 76% of ratings are 4 or 5 stars.
- Verified reviews make up the majority of records.
- Text data quality is high, with only a tiny fraction of missing review text or summary.
- Duplicate review entries exist at a small rate in the sampled portion, so deduplication should be validated during preprocessing.

## Summary of raw metadata

The metadata dataset contains 786,445 records.

### Key statistics

- Total metadata records: 786,445
- Unique `asin` values: 756,077
- Missing metadata fields:
  - `brand`: 5,427 records
  - `main_cat`: 4,137 records
  - `price`: 476,109 records
- Top brands:
  - Sony, Generic, Dell, HP, Samsung, Canon, Panasonic, Asus, Neewer, uxcell
- Top main categories:
  - Computers, All Electronics, Camera & Photo, Home Audio & Theater, Cell Phones & Accessories
- Top categories:
  - Electronics, Computers & Accessories, Camera & Photo, Accessories & Supplies, Audio & Video Accessories

### Observations

- Metadata coverage is generally good for brand and main category, but price is largely missing.
- `price` should not be relied on as a primary feature without additional imputation or filtering.
- Category fields are mixed between normal strings and HTML-escaped values, so cleaning is necessary.
- Metadata has a close match to review items: 756,061 items appear in both review and metadata datasets, with very few items in metadata lacking reviews.

## Existing EDA files

The repository includes the following EDA outputs:

- `eda/phase1_dataset_summary.csv`
- `eda/rating_distribution.csv`
- `eda/reviews_by_year.csv`
- `eda/top_brands.csv`
- `eda/top_categories.csv`

### EDA file highlights

- `phase1_dataset_summary.csv` confirms the full dataset size and sparsity:
  - total reviews: 20,994,353
  - total users: 9,838,676
  - total review items: 756,489
  - total metadata products: 786,445
  - items with both review and metadata: 756,061
  - sparsity: 0.999997
- `rating_distribution.csv` confirms the strong positive bias in ratings.
- `reviews_by_year.csv` shows the dataset spans 1997 to 2018.
- `top_brands.csv` and `top_categories.csv` reflect the dominant electronics brands and categories.

## Data quality and risks

### Sparse / long-tail behavior

- The review dataset is extremely sparse and long-tailed.
- Most users and items appear only once or a few times.
- This makes warm/cold split evaluation important and justifies focusing on cold-start robustness.

### Bias and distribution issues

- Ratings are heavily skewed to positive values.
- The model may learn popularity and positivity bias instead of negative preference signals.
- Evaluation should consider both warm and full splits to reduce over-optimistic results.

### Missing and noisy metadata

- Price is missing in roughly 60% of metadata records.
- Many category values are duplicated with HTML-escaped forms such as `&amp;`.
- Metadata also contains variable structures (strings vs lists), requiring normalization.

### Duplicates and structural issues

- Duplicate (user, item, timestamp) patterns are present at low frequency in a 200k-row sample.
- Metadata records with missing `asin` are rare but should be filtered or handled during feature building.

## Recommendations

1. Treat the dataset as sparse and long-tailed.
2. Use warm/full split evaluation to measure cold-start robustness.
3. Normalize metadata categories and clean HTML entities.
4. Handle missing prices explicitly, or avoid relying on `price` as a core feature.
5. Use review text and summary fields carefully; they are largely present but should be validated for edge cases.
6. Perform deduplication and metadata completeness checks as part of preprocessing.

## Conclusion

This dataset is valuable for recommendation research, but it carries realistic cold-start and sparsity challenges.

The main warnings are:

- strong positive rating bias,
- very sparse user-item interactions,
- incomplete item metadata (especially `price`),
- mixed category formatting.

These points should be reflected in project documentation and evaluation design.
