import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import pickle
import warnings

warnings.filterwarnings('ignore')

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    print("Checking for dataset...")
    data_dir = kagglehub.dataset_download("yasserh/instacart-online-grocery-basket-analysis-dataset")
    print(f"Data located at: {data_dir}")
except ImportError:
    raise RuntimeError("Please install kagglehub: pip install kagglehub")

prior_path = os.path.join(data_dir, "order_products__prior.csv")
train_path = os.path.join(data_dir, "order_products__train.csv")
orders_path = os.path.join(data_dir, "orders.csv")
products_path = os.path.join(data_dir, "products.csv")
aisles_path = os.path.join(data_dir, "aisles.csv")
depts_path = os.path.join(data_dir, "departments.csv")

print("Loading core lookup tables...")
products_df = pd.read_csv(products_path)
aisles_df = pd.read_csv(aisles_path)
departments_df = pd.read_csv(depts_path)
orders_df = pd.read_csv(orders_path, usecols=['order_id', 'user_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order'])

product_info = products_df.merge(aisles_df, on='aisle_id').merge(departments_df, on='department_id')

del products_df, aisles_df, departments_df
gc.collect()

def process_orders_in_chunks(filepath, chunk_size=250_000):
    print(f"Processing {os.path.basename(filepath)} in chunks...")
    chunks = []
    
    for chunk in pd.read_csv(filepath, chunksize=chunk_size, usecols=['order_id', 'product_id', 'reordered']):
        chunk = chunk.merge(orders_df[['order_id', 'user_id']], on='order_id', how='inner')
        
        chunk['user_id'] = pd.to_numeric(chunk['user_id'], downcast='integer')
        chunk['product_id'] = pd.to_numeric(chunk['product_id'], downcast='integer')
        chunk['reordered'] = pd.to_numeric(chunk['reordered'], downcast='integer')
        
        agg_chunk = chunk.groupby(['user_id', 'product_id']).agg(
            purchase_count=('order_id', 'count'),
            ever_reordered=('reordered', 'max')
        ).reset_index()
        
        chunks.append(agg_chunk)
        del chunk
        gc.collect()
        
    print(f"Combining chunks for {os.path.basename(filepath)}...")
    combined = pd.concat(chunks, ignore_index=True)
    
    final_agg = combined.groupby(['user_id', 'product_id']).agg(
        purchase_count=('purchase_count', 'sum'),
        ever_reordered=('ever_reordered', 'max')
    ).reset_index()
    
    del combined, chunks
    gc.collect()
    return final_agg

prior_features = process_orders_in_chunks(prior_path)
train_features = process_orders_in_chunks(train_path)

print("Creating target variables...")
train_features = train_features[['user_id', 'product_id', 'ever_reordered']].rename(
    columns={'ever_reordered': 'target_reordered'}
)

features_df = prior_features.merge(train_features, on=['user_id', 'product_id'], how='left')
features_df['target_reordered'] = features_df['target_reordered'].fillna(0).astype('int8')

del prior_features, train_features
gc.collect()

print("Merging product metadata...")
features_df = features_df.merge(product_info[['product_id', 'department', 'aisle']], on='product_id', how='inner')

print("Encoding categories and splitting data...")
cat_cols = ['department', 'aisle']
num_cols = ['purchase_count', 'ever_reordered']

for col in cat_cols:
    features_df[col] = features_df[col].astype('category')

X = features_df[num_cols + cat_cols]
y = features_df['target_reordered']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

del features_df, X, y
gc.collect()

print("Training LightGBM Model...")
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'max_depth': 8,
    'num_leaves': 64,
    'is_unbalance': True,
    'random_state': 42,
    'n_jobs': -1
}

model = lgb.LGBMClassifier(**params, n_estimators=150)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(50)]
)

print("Evaluating Model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Saving model to disk...")
os.makedirs('production_models', exist_ok=True)
with open('production_models/lgbm_reorder_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Pipeline completed successfully!")