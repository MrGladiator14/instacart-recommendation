# 🛒 The Retail Oracle — Instacart Reorder Prediction

## Project Overview

This project builds a **binary classification model** to predict whether a customer will reorder a previously purchased product in their next Instacart order. It covers the full pipeline from raw data loading through exploratory data analysis (EDA) to feature engineering, producing a final feature matrix ready for model training.

**Target Variable:** `reordered` — `1` if the product was reordered in the user's latest order, `0` if it's a new purchase.

---

## Table of Contents

1. [Dataset Description](#1-dataset-description)
2. [Data Schema](#2-data-schema)
3. [Missing Value Analysis](#3-missing-value-analysis)
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
5. [Feature Engineering](#5-feature-engineering)
   - [Group A — User Features](#group-a--user-features)
   - [Group B — Product Features](#group-b--product-features)
   - [Group C — User-Product Interaction Features](#group-c--user-product-interaction-features)
6. [Final Feature Matrix](#6-final-feature-matrix)
7. [Feature Importance Hypotheses](#7-feature-importance-hypotheses)
8. [Correlation Analysis](#8-correlation-analysis)
9. [Output Files](#9-output-files)

---

## 1. Dataset Description

The dataset is from the [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) competition. It contains anonymized grocery orders from over 200,000 users.

| File | Rows | Description |
|------|------|-------------|
| `aisles.csv` | 134 | Aisle IDs and names |
| `departments.csv` | 21 | Department IDs and names |
| `products.csv` | 49,688 | Product catalogue |
| `orders.csv` | 3,421,083 | All orders across all users |
| `order_products__train.csv` | 1,384,617 | Product rows for train orders (with labels) |
| `order_products__prior.csv` | ~32M | Historical product rows for prior orders |

### `eval_set` Splits in `orders.csv`

| Split | Rows | Purpose |
|-------|------|---------|
| `prior` | 3,214,874 | All historical orders — used to build features |
| `train` | 131,209 | Last order for training users — used as ground truth |
| `test` | 75,000 | Last order for test users — for model inference |

---

## 2. Data Schema

### `orders.csv`
| Column | Description |
|--------|-------------|
| `order_id` | Unique order identifier |
| `user_id` | Customer ID |
| `eval_set` | `prior` / `train` / `test` |
| `order_number` | Sequence number per user (1 = first ever order) |
| `order_dow` | Day of week (0 = Sunday, 6 = Saturday) |
| `order_hour_of_day` | Hour of purchase (0–23) |
| `days_since_prior_order` | Days since last order (NULL for first order) |

### `order_products__train.csv` / `order_products__prior.csv`
| Column | Description |
|--------|-------------|
| `order_id` | Links to `orders.csv` |
| `product_id` | Links to `products.csv` |
| `add_to_cart_order` | Position item was added to cart (1st, 2nd…) |
| `reordered` | **Target** — `1` = product bought before, `0` = new |

### `products.csv`
| Column | Description |
|--------|-------------|
| `product_id` | Unique product ID |
| `product_name` | Product name |
| `aisle_id` | Links to `aisles.csv` |
| `department_id` | Links to `departments.csv` |

---

## 3. Missing Value Analysis

| Table | Column | Missing Count | % Missing | Explanation |
|-------|--------|--------------|-----------|-------------|
| `orders` | `days_since_prior_order` | 206,209 | ~6% | First-ever orders have no prior order |

**Strategy Applied:**
- Fill `days_since_prior_order` with `0` for first orders.
- Add a new derived column: `is_first_order = 1` when `order_number == 1`.

```python
# Verified: all NULLs are exactly order_number == 1
orders['days_since_prior_order'] = orders['days_since_prior_order'].fillna(0)
orders['is_first_order'] = (orders['order_number'] == 1).astype(int)
```

All other tables (`order_products_train`, `products`, `aisles`, `departments`) have **zero missing values**.

---

## 4. Exploratory Data Analysis (EDA)

### 4A. Target Variable Distribution

The target `reordered` has a **mild class imbalance**:

| Class | Count | Percentage |
|-------|-------|-----------|
| Reordered (1) | ~895,539 | ~64.7% |
| New (0) | ~489,078 | ~35.3% |

**Implication:** Accuracy alone is not a reliable metric. Use **F1-score**, **Precision-Recall curve**, and **ROC-AUC** for evaluation.

---

### 4B. Order Behavior Patterns

**Orders Per User (Prior Set)**
- Most users have a moderate number of prior orders.
- Over 10% of users have more than 10 prior orders.
- More orders = richer feature signal for model training.

**Day of Week Patterns**
- Sunday and Monday are the busiest shopping days.
- Weekend ordering is higher than weekday ordering overall.

**Hour of Day Patterns**
- Peak ordering hours fall between **10:00–16:00**.
- Very low activity during late night (00:00–06:00).

**Days Between Orders**
- Mean inter-order gap: ~11–12 days.
- ~20% of users shop on a **weekly cadence** (6–8 day gaps).
- A notable spike at **30 days** indicates monthly shoppers.

---

### 4C. Basket & Cart Position Analysis

**Basket Size Distribution**
- Median basket size: ~8–9 products per order.
- Mean basket size: ~10 products.
- Long tail extends to very large baskets (50+), but these are rare.

**Cart Position vs. Reorder Rate**
- Items added **first** to the cart have significantly higher reorder rates (~80%).
- Reorder rate **drops steadily** as cart position increases.
- This suggests early cart items are habitual purchases (staples, essentials), while later items are more exploratory.

---

### 4D. Department & Aisle Analysis

**Top 5 Departments by Order Volume:**
1. Produce
2. Dairy & Eggs
3. Snacks
4. Beverages
5. Frozen

**Top 5 Departments by Reorder Rate:**
Departments like **personal care**, **dairy & eggs**, and **beverages** tend to have the highest reorder rates — reflecting habitual consumption.

**Key Insight:** Produce has the highest order volume but not necessarily the highest reorder rate, because shoppers mix staples with fresh variety items.

---

## 5. Feature Engineering

Features are organized into **3 groups** based on entity level:

| Group | Entity Level | Question Answered |
|-------|-------------|-------------------|
| A — User Features | Per user | Who is this shopper? How do they behave? |
| B — Product Features | Per product | How popular and sticky is this product? |
| C — User-Product Interaction Features | Per (user, product) pair | How loyal is THIS user to THIS product? |

All features are derived **exclusively from the `prior` split** to prevent data leakage.

---

### Group A — User Features

Aggregated from `order_products_prior` joined with `prior_orders`, grouped by `user_id`.

| Feature | Type | Formula / Derivation | Description |
|---------|------|----------------------|-------------|
| `user_total_orders` | Count | `COUNT(order_id) GROUP BY user_id` | Total number of prior orders placed by the user |
| `user_avg_days_between` | Float | `MEAN(days_since_prior_order)` per user | Average gap between consecutive orders in days |
| `user_std_days_between` | Float | `STD(days_since_prior_order)` per user (0 if single order) | Variability in shopping cadence — high std = irregular shopper |
| `user_min_days_between` | Float | `MIN(days_since_prior_order)` per user | Shortest gap observed between any two orders |
| `user_max_days_between` | Float | `MAX(days_since_prior_order)` per user | Longest gap observed between any two orders |
| `user_avg_order_hour` | Float | `MEAN(order_hour_of_day)` per user | Average time of day when this user shops |
| `user_std_order_hour` | Float | `STD(order_hour_of_day)` per user (0 if single order) | Consistency of shopping time — low std = creature of habit |
| `user_total_products` | Count | `COUNT(product_id)` across all prior orders per user | Total product rows in prior history (includes duplicates) |
| `user_unique_products` | Count | `NUNIQUE(product_id)` per user | Breadth of user's product catalogue |
| `user_total_reorders` | Count | `SUM(reordered)` per user | Total times user has reordered any product |
| `user_reorder_rate` | Rate | `user_total_reorders / user_total_products` | Fraction of all purchased items that were reorders — measures user's habitual tendency |
| `user_avg_basket_size` | Float | `user_total_products / user_total_products` *(effectively total products / orders)* | Average number of products per order |

> **Note:** `user_std_days_between` and `user_std_order_hour` are filled with `0` for users with only one prior order (standard deviation is undefined for a single observation).

---

### Group B — Product Features

Aggregated from `order_products_prior` joined with product metadata, grouped by `product_id`.

| Feature | Type | Formula / Derivation | Description |
|---------|------|----------------------|-------------|
| `product_total_orders` | Count | `COUNT(order_id) GROUP BY product_id` | How many times this product appeared in prior orders across all users |
| `product_reorder_rate` | Rate | `MEAN(reordered) GROUP BY product_id` | Global fraction of times this product was a reorder (vs. first-time purchase) |
| `product_avg_cart_pos` | Float | `MEAN(add_to_cart_order) GROUP BY product_id` | Average cart position across all purchases — low value = staple item added early |
| `product_std_cart_pos` | Float | `STD(add_to_cart_order) GROUP BY product_id` (0 if only 1 order) | Variability in when this product is added to carts |
| `product_unique_users` | Count | `NUNIQUE(user_id) GROUP BY product_id` | Number of distinct customers who have ever purchased this product |
| `dept_avg_reorder_rate` | Rate | `MEAN(product_reorder_rate) GROUP BY department` | Department-level signal: average reorder rate across all products in the same department |
| `aisle_avg_reorder_rate` | Rate | `MEAN(product_reorder_rate) GROUP BY aisle` | Aisle-level signal: average reorder rate across all products in the same aisle |

> **`dept_avg_reorder_rate` and `aisle_avg_reorder_rate`** serve as **smoothed group-level signals**. For products with sparse purchase history, department/aisle averages provide a more stable estimate of reorder tendency.

---

### Group C — User-Product Interaction Features

Aggregated from `order_products_prior` joined with `prior_orders`, grouped by `(user_id, product_id)` pairs. These are the most predictive features as they capture individual loyalty.

| Feature | Type | Formula / Derivation | Description |
|---------|------|----------------------|-------------|
| `up_times_ordered` | Count | `COUNT(order_id) GROUP BY (user_id, product_id)` | Number of times this user has purchased this product in prior orders |
| `up_reorder_count` | Count | `SUM(reordered) GROUP BY (user_id, product_id)` | Number of those purchases that were classified as reorders |
| `up_avg_cart_position` | Float | `MEAN(add_to_cart_order) GROUP BY (user_id, product_id)` | User's personal average position for adding this product to cart |
| `up_first_order_number` | Int | `MIN(order_number) GROUP BY (user_id, product_id)` | The user's order number when they first purchased this product |
| `up_last_order_number` | Int | `MAX(order_number) GROUP BY (user_id, product_id)` | The user's order number of the most recent purchase |
| `up_reorder_rate` | Rate | `up_reorder_count / up_times_ordered` | Fraction of this user's purchases of this product that were reorders (user-level loyalty score) |
| `up_orders_span` | Int | `up_last_order_number - up_first_order_number` | How many orders span between first and last purchase — 0 means only ordered once |
| `up_buy_frequency` | Float | `up_times_ordered / (up_orders_span + 1)` | How regularly the user buys this product — 1.0 = bought every single order since first purchase; lower = occasional |
| `up_orders_since_last` | Int | `user_max_order - up_last_order_number` | How many orders ago was this product last purchased — 0 = bought in the most recent prior order |
| `up_recency_score` | Float | `1 / (up_orders_since_last + 1)` | Recency signal scaled to (0, 1] — score of 1.0 means bought in the most recent prior order; decays toward 0 for older purchases |

#### Formula Notes

**`up_buy_frequency`**
```
up_buy_frequency = up_times_ordered / (up_orders_span + 1)
```
The `+1` prevents division by zero when a product was only ordered once (`up_orders_span = 0`). A value of `1.0` means the user bought the product in every order from their first purchase onward.

**`up_recency_score`**
```
up_recency_score = 1 / (up_orders_since_last + 1)
```
The `+1` ensures the score is `1.0` when `up_orders_since_last = 0` (bought in last order). As recency increases, the score decays: 2 orders ago → 0.5; 4 orders ago → 0.2; 9 orders ago → 0.1.

---

## 6. Final Feature Matrix

All three feature groups are assembled into a single training matrix:

```
final_df = candidates
         ← MERGE user-product interaction features (up_feat)
         ← MERGE user features (user_feat)
         ← MERGE product features (prod_feat)
         ← MERGE train labels (order_products_train)
```

**Candidate Generation:** For each user in the train split, candidate products are all products that user ordered in the prior split. This ensures the model only predicts for products with known history.

**Label Assignment:** `reordered = 1` if the product appears in the user's train order; `0` otherwise (product was in their history but not repurchased).

**Final Matrix Stats:**
| Property | Value |
|----------|-------|
| Rows | ~13.3M (user-product pairs) |
| Columns | ~25 features + identifiers + target |
| Positive class (reordered=1) | ~59–65% |
| Negative class (reordered=0) | ~35–41% |

---

## 7. Feature Importance Hypotheses

Ranked by expected predictive power based on domain logic:

| Rank | Feature | Group | Expected Power | Reasoning |
|------|---------|-------|---------------|-----------|
| 1 | `up_times_ordered` | C | HIGH | More purchases = stronger reorder signal |
| 2 | `up_reorder_rate` | C | HIGH | Direct measure of user-product loyalty |
| 3 | `product_reorder_rate` | B | HIGH | Global popularity of the product being reordered |
| 4 | `up_orders_since_last` | C | HIGH | Recency — recently purchased items are more likely to be reordered |
| 5 | `up_buy_frequency` | C | MEDIUM | Regular purchases indicate a habitual pattern |
| 6 | `user_reorder_rate` | A | MEDIUM | Habitual buyers tend to reorder across all products |
| 7 | `product_avg_cart_pos` | B | MEDIUM | Items added early = habitual staple = higher reorder chance |
| 8 | `up_avg_cart_position` | C | MEDIUM | Same as above but user-specific |
| 9 | `user_total_orders` | A | LOW | Experienced users have clearer purchase preferences |
| 10 | `dept_avg_reorder_rate` | B | LOW | Department-level habitual categories (e.g. personal care) |

---

## 8. Correlation Analysis

Correlation heatmaps (Pearson) were computed for all 6 raw source tables and all 4 engineered feature tables:

| # | Table | Purpose |
|---|-------|---------|
| 1 | `orders` | Order metadata correlations (dow, hour, days_since) |
| 2 | `order_products_train` | Train label relationships (cart order vs. reordered) |
| 3 | `order_products_prior` | Prior history structure |
| 4 | `products` | Product ID correlations (mostly uninformative) |
| 5 | `aisles` | Aisle reference (2 columns) |
| 6 | `departments` | Department reference (2 columns) |
| 7 | `user_feat` | User-level feature correlations |
| 8 | `prod_feat` | Product-level feature correlations |
| 9 | `up_feat` | User-product interaction correlations |
| 10 | `final_df` | Full training matrix correlation heatmap |

**Key Correlation Findings:**
- `up_times_ordered` and `up_reorder_count` are highly correlated (by construction).
- `up_orders_since_last` and `up_recency_score` are strongly negatively correlated (inverse transformation).
- `product_total_orders` and `product_unique_users` are strongly positively correlated — popular products attract many users.
- `user_total_products` and `user_total_reorders` are positively correlated — more shopping = more reordering.

---

## 9. Output Files

All processed files are saved to `data/processed/`:

| File | Description | Key Columns |
|------|-------------|-------------|
| `user_features.csv` | Group A — one row per user | `user_id`, all user_* features |
| `product_features.csv` | Group B — one row per product | `product_id`, all product_* features |
| `user_product_features.csv` | Group C — one row per (user, product) pair | `user_id`, `product_id`, all up_* features |
| `final_features_train.csv` | Assembled training matrix | All features + `reordered` target |

---

## Next Steps

1. **Baseline Model** — Logistic Regression on the final feature matrix
2. **Model Comparison** — LightGBM, XGBoost, Random Forest
3. **Evaluation** — F1-score, ROC-AUC, Precision-Recall curve
4. **Threshold Tuning** — Optimize classification threshold per-user using F1
5. **Feature Selection** — Identify and remove low-importance or highly collinear features

---

## Dependencies

```python
pandas
numpy
matplotlib
seaborn
scikit-learn      # for modeling (next steps)
lightgbm          # for modeling (next steps)
```

---

## File Structure

```
project/
├── data/
│   ├── raw/
│   │   ├── aisles.csv
│   │   ├── departments.csv
│   │   ├── products.csv
│   │   ├── orders.csv
│   │   ├── order_products__train.csv
│   │   └── order_products__prior.csv
│   └── processed/
│       ├── user_features.csv
│       ├── product_features.csv
│       ├── user_product_features.csv
│       └── final_features_train.csv
├── instacart_eda_features_updated.ipynb
└── README.md
```
