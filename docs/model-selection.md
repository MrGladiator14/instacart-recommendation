## Model Training & Optimization Report

### 1. Phase I: Baseline & Class Imbalance Addressing

Initially, the models struggled significantly with a high class imbalance. While the **XGBoost** model showed a decent  **AUC (0.8004)** , the **F1-Score (0.2127)** revealed that the model was failing to capture the minority class effectively, resulting in a very low recall.

| **Metric**   | **Logistic Regression** | **LightGBM** | **XGBoost (Baseline)** |
| ------------------ | ----------------------------- | ------------------ | ---------------------------- |
| **AUC**      | 0.5428                        | 0.7995             | **0.8004**             |
| **Accuracy** | -                             | -                  | 0.9074                       |
| **F1-Score** | -                             | -                  | 0.2127                       |

---

### 2. Phase II: Hyperparameter Expansion & Scaling

To improve the model's depth and predictive power, we implemented two major changes:

* **Tree Parameter Expansion:** Increased the number of estimators (n_estimators) from **300 to 1,000** for both LGBM and XGBoost to capture more complex non-linear relationships.
* **Feature Scaling:** Applied Robust Scaling to the feature set to ensure that Logistic Regression and gradient-based optimizers weren't skewed by outliers.

> **Note:** Increasing parameters from 300 to 1,000 significantly reduced the bias in our tree-based models, allowing for a more granular fit of the data distribution.

---

### 3. Phase III: Final Validation Results

After the hyperparameter expansion and feature engineering, the model performance stabilized. By adjusting the classification threshold to account for the class imbalance, we achieved a significant boost in the  **Macro F1-Score** .

**Final Model Performance (Optimized XGBoost):**

| **Metric**               | **Value**  |
| ------------------------------ | ---------------- |
| **Validation AUC**       | **0.8412** |
| **Validation Accuracy**  | **0.8215** |
| **Validation Precision** | **0.6840** |
| **Validation Recall**    | **0.7180** |
| **Macro F1-Score**       | **70.04%** |

---

### Summary of Improvements

The transition from a basic model to the optimized version resulted in the following trajectory:

1. **Macro F1-Score (Baseline):** ~21.2% (Severe imbalance issues)
2. **Macro F1-Score (Mid-Optimization):** 66.7% (Improved via parameter tuning)
3. **Macro F1-Score (Final):** **70.0%** (Achieved via feature scaling and **$n=1000$** trees)

**Best model saved as:** `production_models/xgb_reorder_model_v2.pkl`
