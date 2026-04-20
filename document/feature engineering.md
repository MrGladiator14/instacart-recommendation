# Instacart Recommendation System

A memory-efficient, object-oriented recommendation system built on LightGBM that predicts which products a user is likely to reorder, using temporal and behavioral features engineered from Instacart's transaction history.

---

## Architecture Overview

The system is composed of five core classes, each with a single responsibility:

```
InstacartRecommender          ← Orchestrator / entry point
├── DataLoader                ← CSV ingestion with chunked reads
├── FeatureEngineer           ← Product, user, and user×product features
├── TemporalFeatureEngineer   ← Time-aware features and training data builder
└── ModelTrainer              ← LightGBM training, inference, and submission
```

---

## Classes

### `DataLoader`

Handles loading and initial processing of Instacart CSV files.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_dir` | `str` | Path to directory containing CSV files |
| `batch_size` | `int` | Rows per chunk during CSV loading (default: `100000`) |
| `use_gpu` | `bool` | Whether to enable GPU acceleration |

**Key Methods**

| Method | Description |
|--------|-------------|
| `load_data_in_batches(filepath, dtype_dict)` | Reads large CSVs in memory-safe chunks and concatenates them |
| `load_all_data()` | Loads all four datasets — `priors`, `train`, `orders`, `products` — with optimised dtypes |

**Datasets Loaded**

| File | Key Columns |
|------|-------------|
| `order_products__prior.csv` | `order_id`, `product_id`, `add_to_cart_order`, `reordered` |
| `order_products__train.csv` | Same schema as prior |
| `orders.csv` | `order_id`, `user_id`, `eval_set`, `order_number`, `order_dow`, `order_hour_of_day`, `days_since_prior_order` |
| `products.csv` | `product_id`, `aisle_id`, `department_id` |

---

### `FeatureEngineer`

Builds static features from the prior order history.

**Key Methods**

| Method | Output Features |
|--------|-----------------|
| `create_product_features(priors, products)` | `orders`, `reorders`, `reorder_rate` per product |
| `create_user_features(priors, orders)` | `total_items`, `total_distinct_items`, `nb_orders`, `average_basket`, `average_days_between_orders` |
| `create_user_product_features(priors)` | `nb_orders`, `last_order_id`, `sum_pos_in_cart` per user×product pair |

The user×product computation uses a **key-value merge strategy**: priors are iterated in batches, partial dictionaries are accumulated, and then merged in a single pass — avoiding the memory overhead of a full groupby on the entire dataset.

---

### `TemporalFeatureEngineer`

Converts relative order timing into an absolute day timeline and derives time-series features per user×product pair.

**Key Methods**

| Method | Description |
|--------|-------------|
| `build_absolute_timeline(orders)` | Cumulative sum of `days_since_prior_order` per user to produce `abs_day` |
| `build_userproduct_temporal_features(priors, orders_sorted)` | Per user×product: avg/std/min/max purchase interval, last purchase day, purchase count |
| `build_user_timeline(orders_sorted)` | Per user: last active day, total orders, avg days between orders |
| `build_prediction_features(user_ids, target_day_offset, ...)` | Assembles feature matrix for a given set of users at a future day offset |
| `build_training_data(train_orders, train, ...)` | Builds labelled training rows by replaying historical orders |

**Derived Temporal Features**

| Feature | Formula |
|---------|---------|
| `UP_days_since_last` | `target_abs_day − UP_last_abs_day` |
| `UP_days_overdue` | `UP_days_since_last − UP_avg_interval` |
| `UP_overdue_zscore` | `UP_days_overdue / (UP_std_interval + ε)` |
| `UP_interval_progress` | `UP_days_since_last / (UP_avg_interval + ε)` |

---

### `ModelTrainer`

Trains a binary LightGBM classifier and handles inference and submission generation.

**Model Features (17 total)**

```
UP_days_since_last      UP_days_overdue         UP_overdue_zscore
UP_interval_progress    UP_avg_interval         UP_std_interval
UP_min_interval         UP_max_interval         UP_purchase_count
nb_orders               average_days_between_orders  average_basket
total_distinct_items    reorder_rate            aisle_id
department_id           target_day_offset
```

**LightGBM Configuration**

| Parameter | Value |
|-----------|-------|
| Objective | `binary` (log loss) |
| Num leaves | `64` |
| Max depth | `8` |
| Learning rate | `0.05` |
| Feature fraction | `0.8` |
| Bagging fraction | `0.9` |
| Boosting rounds | `200` |
| Device | `gpu` (falls back to `cpu`) |

**Key Methods**

| Method | Description |
|--------|-------------|
| `train_model(df_train)` | Trains the LightGBM model and stores it in `self.model` |
| `predict_for_user_on_day(user_id, target_day_offset, ...)` | Returns top-K reorder probabilities for a single user at a given future day |
| `generate_submission(df_test, test_orders, threshold)` | Applies threshold (default `0.22`) and writes `sub.csv` in Instacart competition format |

---

### `InstacartRecommender`

Top-level orchestrator that wires all components together.

```python
recommender = InstacartRecommender(
    data_dir='/path/to/data/',
    batch_size=100000,
    use_gpu=True
)
recommender.run_pipeline()
```

**`run_pipeline()` Behaviour**

The pipeline includes a **feature caching step**: if `train_features.csv` and `test_features.csv` already exist on disk, it skips feature engineering entirely and loads them directly, saving significant compute time on re-runs.

---

## Data Flow

```
Raw CSVs
    │
    ▼
DataLoader.load_all_data()
    │
    ├──▶ FeatureEngineer → product features, user features, user×product features
    │
    ├──▶ TemporalFeatureEngineer → absolute timeline → temporal UP features
    │                                                 → training data (labelled)
    │                                                 → test feature matrix
    ▼
ModelTrainer.train_model()   ←── labelled training data
    │
    ├──▶ .predict_for_user_on_day()   (interactive inference)
    └──▶ .generate_submission()       → sub.csv
```

---

## Memory Optimisations

- **Chunked CSV reads** — large files are loaded in configurable-size chunks and concatenated once.
- **Dtype downcasting** — integers use `np.int8 / int16 / int32 / uint8 / uint16` where possible; floats use `np.float32`.
- **Batch processing throughout** — feature engineering, training data construction, and submission generation all operate in batches with periodic `gc.collect()` calls.
- **Dictionary-based UP aggregation** — avoids a full in-memory groupby on the 32M-row prior dataset.

---

## Output

| File | Description |
|------|-------------|
| `train_features.csv` | Cached labelled training feature matrix |
| `test_features.csv` | Cached test feature matrix |
| `sub.csv` | Submission file with columns `order_id`, `products` |