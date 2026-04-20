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
from sklearn.metrics import f1_score, classification_report, roc_auc_score
import pickle
import warnings

warnings.filterwarnings('ignore')

print("Loading train features...")
train_features = pd.read_csv('features/scaled_train_features.csv')
print(f"Train features shape: {train_features.shape}")

print("Preparing features for training...")

feature_cols = [col for col in train_features.columns if col not in ['label', 'user_id', 'product_id', 'order_id', 'department_id']]
print(f"Feature columns: {feature_cols}")

cat_cols = ['aisle_id']
num_cols = [col for col in feature_cols if col not in cat_cols]

print(f"Categorical columns: {cat_cols}")
print(f"Numerical columns: {num_cols}")

X = train_features[feature_cols].copy()
y = train_features['label']

y = (y > 0).astype(int)

print(f"Target unique values after conversion: {sorted(y.unique())}")
print(f"Target value counts after conversion: {y.value_counts()}")

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

for col in cat_cols:
    X[col] = X[col].fillna(-1).astype('int32')

for col in cat_cols:
    X[col] = X[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Target distribution - Train: {y_train.mean():.4f}, Test: {y_test.mean():.4f}")

del X, y
gc.collect()

def train_models(X_train, X_test, y_train, y_test, cat_cols):
    models = {}
    scores = {}
    
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"Class weights: {weight_dict}")
    print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.01,
        'max_depth': 8,
        'num_leaves': 64,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'scale_pos_weight': weight_dict[1] / weight_dict[0],
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'device': 'gpu'
    }
    
    print("Training LightGBM model...")
    lgb_model = lgb.LGBMClassifier(**lgb_params, n_estimators=10000)
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='binary_logloss', 
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        lgb_pred = (lgb_proba >= threshold).astype(int)
        current_f1 = f1_score(y_test, lgb_pred, average='macro')
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
    
    lgb_pred_final = (lgb_proba >= best_threshold).astype(int)
    lgb_score = f1_score(y_test, lgb_pred_final, average='macro')
    models['lgb'] = lgb_model
    scores['lgb'] = lgb_score
    print(f"LightGBM Macro F1: {lgb_score:.4f} (threshold: {best_threshold:.3f})")
    
    print("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        learning_rate=0.01,
        max_depth=8,
        n_estimators=10000,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=1,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=weight_dict[1] / weight_dict[0],
        random_state=42,
        n_jobs=-1,
        enable_categorical=True,
        eval_metric='logloss',
        early_stopping_rounds=100,
        verbose=-1,
        device='gpu'
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    best_threshold_xgb = 0.5
    best_f1_xgb = 0
    
    for threshold in thresholds:
        xgb_pred = (xgb_proba >= threshold).astype(int)
        current_f1 = f1_score(y_test, xgb_pred, average='macro')
        if current_f1 > best_f1_xgb:
            best_f1_xgb = current_f1
            best_threshold_xgb = threshold
    
    xgb_pred_final = (xgb_proba >= best_threshold_xgb).astype(int)
    xgb_score = f1_score(y_test, xgb_pred_final, average='macro')
    models['xgb'] = xgb_model
    scores['xgb'] = xgb_score
    print(f"XGBoost Macro F1: {xgb_score:.4f} (threshold: {best_threshold_xgb:.3f})")
    
    print("Training Logistic Regression model...")
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    for col in cat_cols:
        le = LabelEncoder()
        X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype('str'))
        X_test_encoded[col] = le.transform(X_test_encoded[col].astype('str'))
    
    lr_model = SKLogisticRegression(
        random_state=42, 
        max_iter=10000, 
        n_jobs=-1, 
        C=0.1,
        class_weight='balanced'
    )
    lr_model.fit(X_train_encoded, y_train)
    
    lr_proba = lr_model.predict_proba(X_test_encoded)[:, 1]
    best_threshold_lr = 0.5
    best_f1_lr = 0
    
    for threshold in thresholds:
        lr_pred = (lr_proba >= threshold).astype(int)
        current_f1 = f1_score(y_test, lr_pred, average='macro')
        if current_f1 > best_f1_lr:
            best_f1_lr = current_f1
            best_threshold_lr = threshold
    
    lr_pred_final = (lr_proba >= best_threshold_lr).astype(int)
    lr_score = f1_score(y_test, lr_pred_final, average='macro')
    models['lr'] = lr_model
    scores['lr'] = lr_score
    print(f"Logistic Regression Macro F1: {lr_score:.4f} (threshold: {best_threshold_lr:.3f})")
    
    return models, scores

models, scores = train_models(X_train, X_test, y_train, y_test, cat_cols)

print("\nFinal Model Scores:")
for name, score in scores.items():
    print(f"{name.upper()}: {score:.4f}")

best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]

print(f"\nBest model: {best_model_name.upper()} with score: {scores[best_model_name]:.4f}")

os.makedirs('production_models', exist_ok=True)
with open(f'production_models/{best_model_name}_reorder_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"Best model saved as: production_models/{best_model_name}_reorder_model.pkl")

print(f"\n=== Final Evaluation Results ===")
print(f"Best Model: {best_model_name.upper()}")
print(f"Validation Macro F1: {scores[best_model_name]:.4f}")

if best_model_name == 'lgb':
    val_pred = best_model.predict_proba(X_test)[:, 1]
elif best_model_name == 'xgb':
    val_pred = best_model.predict_proba(X_test)[:, 1]
else:  # lr
    X_test_encoded = X_test.copy()
    for col in cat_cols:
        le = LabelEncoder()
        X_test_encoded[col] = le.fit_transform(X_test_encoded[col].astype('str'))
    val_pred = best_model.predict_proba(X_test_encoded)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.05)
best_threshold_final = 0.5
best_f1_final = 0

for threshold in thresholds:
    val_pred_binary = (val_pred >= threshold).astype(int)
    current_f1 = f1_score(y_test, val_pred_binary, average='macro')
    if current_f1 > best_f1_final:
        best_f1_final = current_f1
        best_threshold_final = threshold

val_pred_binary_final = (val_pred >= best_threshold_final).astype(int)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

print(f"\nFinal Validation Metrics (threshold: {best_threshold_final:.3f}):")
print(f"Validation Macro F1: {f1_score(y_test, val_pred_binary_final, average='macro'):.4f}")
print(f"Validation Accuracy: {accuracy_score(y_test, val_pred_binary_final):.4f}")
print(f"Validation Precision: {precision_score(y_test, val_pred_binary_final, average='binary'):.4f}")
print(f"Validation Recall: {recall_score(y_test, val_pred_binary_final, average='binary'):.4f}")
print(f"Validation Binary F1: {f1_score(y_test, val_pred_binary_final, average='binary'):.4f}")
print(f"Prediction Distribution - Mean: {val_pred.mean():.4f}, Std: {val_pred.std():.4f}")

print(f"\nDetailed Classification Report:")
print(classification_report(y_test, val_pred_binary_final))

print("\nTraining completed successfully!")