# Instacart Recommendation System Documentation

This document explains three key components of the Instacart recommendation system: feature scaling, training logs, and prediction functionality.

## Table of Contents

- [Feature Scaling (`feature_scaling.py`)](#feature-scaling-featurescalingpy)
- [Training Logs (`logs/train.log.txt`)](#training-logs-logstrainlogtxt)
- [Prediction System (`predict.py`)](#prediction-system-predictpy)

---

## Feature Scaling (`feature_scaling.py`)

### Overview

The `FeatureScaler` class is responsible for preprocessing numerical features through various scaling techniques to improve model performance. It handles large datasets efficiently using batch processing.

### Key Features

#### **Supported Scalers**

- **StandardScaler**: Removes mean and scales to unit variance
- **MinMaxScaler**: Scales features to a range [0, 1]
- **RobustScaler**: Uses statistics robust to outliers

#### **Batch Processing**

- Processes data in configurable batches (default: 100,000 rows)
- Memory-efficient handling of large datasets
- Automatic garbage collection to prevent memory leaks

#### **Column Type Detection**

- Automatically identifies numeric vs categorical columns
- Excludes ID columns from scaling: `user_id`, `product_id`, `department_id`
- Supports multiple data types: int64, float64, int32, float32

### Core Methods

#### `__init__(features_dir, scaler_type, batch_size)`

Initializes the scaler with specified parameters and creates the appropriate scaling object.

#### `identify_column_types(sample_file, sample_size)`

Analyzes a sample of the data to identify which columns should be scaled:

```python
# Excludes from scaling: ['user_id', 'product_id', 'department_id']
# Numeric columns detected based on dtype
# Categorical columns remain unchanged
```

#### `fit_scaler(file_path)`

Fits the scaling parameters on the training data:

- Uses `partial_fit` for incremental learning when available
- Falls back to accumulating data for scalers without `partial_fit`
- Processes data in batches to handle memory constraints

#### `process_file(input_file)`

Transforms and saves scaled features:

- Reads input file in batches
- Applies scaling transformation to numeric columns
- Preserves categorical columns unchanged
- Saves output as `scaled_{filename}.csv`

#### `scale_all_features()`

Main pipeline method that:

1. Identifies all CSV files needing scaling
2. Determines column types
3. Fits scaler on training data
4. Processes all files and saves scaled versions

### Usage Example

```python
scaler = FeatureScaler(
    features_dir='features',
    scaler_type='standard',
    batch_size=50000
)
scaler.scale_all_features()
```

---

## Training Logs (`logs/train.log.txt`)

### Overview

The training log captures the model training process, including dataset statistics, model configurations, and performance metrics.

### Key Log Sections

#### **Dataset Information**

```
Checking for dataset...
Data located at: /home/bryson/.cache/kagglehub/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset/versions/1
```

#### **LightGBM Training Details**

- **Dataset Size**: 10,646,362 samples with 4 features
- **Class Distribution**: 663,059 positive vs 9,983,303 negative samples
- **Imbalance Ratio**: ~6.2% positive samples (pavg=0.062280)
- **Initial Score**: -2.711805 (log-odds for the imbalance)

#### **Training Warnings**

Multiple warnings indicate potential issues:

```
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
```

This suggests the model may have reached optimal splits or there are issues with feature quality.

#### **Optimization Warning**

```
[CUML] [warning] L-BFGS stopped, because the line search failed to advance
```

Indicates convergence issues in the optimization process.

#### **Model Performance Comparison**

```
Model Scores:
LGB: 0.7293
XGB: 0.7293
LR: 0.7053
Best model: XGB with score: 0.7293
```

### Performance Analysis

- **LightGBM and XGBoost**: Identical performance (0.7293 F1-score)
- **Logistic Regression**: Lower performance (0.7053)
- **Selected Model**: XGBoost (chosen as best performer)

---

## Prediction System (`predict.py`)

### Overview

The `InstacartPredictor` class handles generating reorder predictions using trained models, with support for both CPU and GPU acceleration.

### Key Features

#### **Model Loading**

- Supports both pickle and XGBoost native formats
- Automatic GPU acceleration when available
- Fallback to CPU if GPU fails

#### **Data Processing**

- Loads data from Kaggle dataset
- Supports both pandas (CPU) and cuDF (GPU) dataframes
- Batch processing for memory efficiency

#### **Feature Engineering**

- Handles categorical feature encoding
- Ensures feature consistency with training
- Supports 21 different features including:
  - User-product interaction features
  - Temporal features
  - Product popularity features
  - User behavior features

### Core Methods

#### `__init__(model_path, probability_threshold, use_gpu)`

Initializes predictor with configuration:

- **model_path**: Path to trained model
- **probability_threshold**: Minimum probability for reorder prediction (default: 0.01)
- **use_gpu**: Enable GPU acceleration

#### `load_model()`

Loads the trained model with GPU optimization:

```python
# Attempts GPU acceleration first
# Falls back to CPU if GPU unavailable
# Supports multiple model formats
```

#### `load_data()`

Loads necessary data from Kaggle:

- Product information (ID, name)
- Test orders for prediction
- Automatically detects and uses GPU when available

#### `prepare_features(features_df)`

Prepares features for prediction:

- Encodes categorical variables
- Ensures feature order consistency
- Filters to required features only

#### `predict(features_df, batch_size)`

Generates predictions in batches:

- Processes data in configurable batches (default: 10,000)
- Applies probability threshold filtering
- Returns only high-confidence predictions

#### `aggregate_predictions(predictions_df, features_df)`

Aggregates predictions by product:

- Counts predicted reorders per product
- Merges with product information
- Returns sorted results by predicted volume

#### `generate_stock_predictions(output_path)`

Complete prediction pipeline:

1. Loads model and data
2. Processes test features
3. Generates predictions
4. Aggregates results
5. Saves to output file

### Expected Features

The system expects 21 features including:

- `UP_avg_interval`, `UP_std_interval`, `UP_min_interval`, `UP_max_interval`
- `UP_last_abs_day`, `UP_purchase_count`, `user_last_abs_day`, `user_total_orders`
- `user_avg_days_between_orders`, `target_abs_day`, `UP_days_since_last`
- `UP_days_overdue`, `UP_overdue_zscore`, `UP_interval_progress`
- `aisle_id`, `orders`, `reorders`, `reorder_rate`, `total_items`
- `total_distinct_items`, `average_days_between_orders`, `average_basket`
- `nb_orders`, `target_day_offset`

### Usage Example

```python
predictor = InstacartPredictor(
    use_gpu=True,
    probability_threshold=0.01
)
predictor.generate_stock_predictions("stock_predictions.csv")
```

---

## System Integration

### Workflow

1. **Feature Scaling**: `feature_scaling.py` preprocesses raw features
2. **Model Training**: Models trained on scaled features (logged in `train.log.txt`)
3. **Prediction**: `predict.py` uses trained models for inference

### Performance Considerations

- **Memory Management**: Batch processing prevents memory overflow
- **GPU Acceleration**: Optional GPU support for faster processing
- **Scalability**: Designed for large datasets (millions of samples)

### Configuration

All components use the `Config` class for centralized configuration management, ensuring consistency across the pipeline.
