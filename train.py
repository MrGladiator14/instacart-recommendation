import cuml.accel
cuml.accel.install()

import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import pickle
import warnings

warnings.filterwarnings('ignore')

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    print("Checking for dataset...")
    d = kagglehub.dataset_download("yasserh/instacart-online-grocery-basket-analysis-dataset")
    print(f"Data located at: {d}")
except ImportError:
    raise RuntimeError("Please install kagglehub: pip install kagglehub")

p = os.path.join(d, "order_products__prior.csv")
t = os.path.join(d, "order_products__train.csv")
o = os.path.join(d, "orders.csv")
pr = os.path.join(d, "products.csv")
a = os.path.join(d, "aisles.csv")
dept = os.path.join(d, "departments.csv")

products_df = pd.read_csv(pr)
aisles_df = pd.read_csv(a)
departments_df = pd.read_csv(dept)
orders_df = pd.read_csv(o, usecols=['order_id', 'user_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order'])

product_info = products_df.merge(aisles_df, on='aisle_id').merge(departments_df, on='department_id')

del products_df, aisles_df, departments_df
gc.collect()

def process_orders_in_chunks(filepath, chunk_size=250_000):
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
        
    combined = pd.concat(chunks, ignore_index=True)
    
    final_agg = combined.groupby(['user_id', 'product_id']).agg(
        purchase_count=('purchase_count', 'sum'),
        ever_reordered=('ever_reordered', 'max')
    ).reset_index()
    
    del combined, chunks
    gc.collect()
    return final_agg

prior_features = process_orders_in_chunks(p)
train_features = process_orders_in_chunks(t)

train_features = train_features[['user_id', 'product_id', 'ever_reordered']].rename(
    columns={'ever_reordered': 'target_reordered'}
)

features_df = prior_features.merge(train_features, on=['user_id', 'product_id'], how='left')
features_df['target_reordered'] = features_df['target_reordered'].fillna(0).astype('int8')

del prior_features, train_features
gc.collect()

features_df = features_df.merge(product_info[['product_id', 'department', 'aisle']], on='product_id', how='inner')

cat_cols = ['department', 'aisle']
num_cols = ['purchase_count', 'ever_reordered']

for col in cat_cols:
    features_df[col] = features_df[col].astype('category')

X = features_df[num_cols + cat_cols]
y = features_df['target_reordered']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

del features_df, X, y
gc.collect()

def train_models(X_train, X_test, y_train, y_test, cat_cols):
    models = {}
    scores = {}
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.1,
        'max_depth': 8,
        'num_leaves': 64,
        'is_unbalance': True,
        'random_state': 42,
        'n_jobs': -1
    }
    
    lgb_model = lgb.LGBMClassifier(**lgb_params, n_estimators=150)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    lgb_score = roc_auc_score(y_test, lgb_pred)
    models['lgb'] = lgb_model
    scores['lgb'] = lgb_score
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        learning_rate=0.1,
        max_depth=8,
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        enable_categorical=True
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    xgb_score = roc_auc_score(y_test, xgb_pred)
    models['xgb'] = xgb_model
    scores['xgb'] = xgb_score
    
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    for col in cat_cols:
        le = LabelEncoder()
        X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype('str'))
        X_test_encoded[col] = le.transform(X_test_encoded[col].astype('str'))
    
    lr_model = SKLogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    lr_model.fit(X_train_encoded, y_train)
    lr_pred = lr_model.predict_proba(X_test_encoded)[:, 1]
    lr_score = roc_auc_score(y_test, lr_pred)
    models['lr'] = lr_model
    scores['lr'] = lr_score
    
    return models, scores

models, scores = train_models(X_train, X_test, y_train, y_test, cat_cols)

print("Model Scores:")
for name, score in scores.items():
    print(f"{name.upper()}: {score:.4f}")

best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]

os.makedirs('production_models', exist_ok=True)
with open(f'production_models/{best_model_name}_reorder_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"Best model: {best_model_name.upper()} with score: {scores[best_model_name]:.4f}")