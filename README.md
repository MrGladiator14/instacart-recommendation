```bash
uv run python reorder_predictor.py --user_id 123 --product_id 1 --purchase_count 5 --ever_reordered 1
```

### **Data Loading**

The pipeline starts by fetching the Instacart dataset through Kaggle Hub, loading five core CSV files: order products (prior and training), orders metadata, products, aisles, and departments. These are merged to create a unified product information table.

### **Feature Engineering**

Three key features are extracted:

* **purchase_count** : How many times a user has purchased a specific product from prior orders
* **ever_reordered** : Maximum reorder flag indicating if a product was ever reordered (0 or 1)
* **Categorical metadata** : Department and aisle names associated with each product

The target variable (`target_reordered`) is extracted from the training set to indicate whether a product was reordered in the training period.

### **Preprocessing**

* **Categorical preprocessing** : Department and aisle columns are encoded (typically one-hot encoding)
* **Numerical preprocessing** : purchase_count and ever_reordered are standardized/scaled
* **Train-test split** : 80/20 stratified split on the target variable to maintain class balance

### **Model Training**

LightGBM classifier is trained with:

* **Objective** : Binary classification (reordered vs. not reordered)
* **Key hyperparameters** : learning_rate=0.1, max_depth=8, num_leaves=64, n_estimators=150
* **Early stopping** : Stops after 20 rounds without improvement
* **Metric** : AUC (ROC-AUC for binary classification)
* **Class imbalance handling** : `is_unbalance=True` to handle skewed class distribution

### **Evaluation & Export**

Model performance is evaluated using:

* **ROC-AUC Score** : Primary metric for ranking prediction quality
* **Classification Report** : Precision, recall, F1-score per class
* **Model persistence** : Trained model is saved as a pickle file for production deployment
