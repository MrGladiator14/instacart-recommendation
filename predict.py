#!/usr/bin/env python3
"""
Prediction script for Instacart reorder recommendation system.

This script generates stock predictions based on a trained model.
"""

import os
import pickle
import warnings
from typing import Dict, Optional

import pandas as pd
import cudf
import kagglehub
import xgboost as xgb

from config import Config
from utils import validate_file_path, create_output_directory

warnings.filterwarnings('ignore')


class InstacartPredictor:
    
    def __init__(self, model_path: str = None, probability_threshold: float = None, use_gpu: bool = True):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
            probability_threshold: Threshold for reorder prediction
            use_gpu: Whether to use GPU acceleration
        """
        config = Config.get_model_config()
        self.model_path = model_path or config['model_path']
        self.probability_threshold = probability_threshold or config['probability_threshold']
        self.use_gpu = use_gpu
        self.model = None
        self.product_info = None
        self.orders_df = None
        
    def load_model(self) -> None:
        validate_file_path(self.model_path)
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            if self.use_gpu and hasattr(self.model, 'get_booster'):
                booster = self.model.get_booster()
                booster.set_param({'predictor': 'gpu_predictor'})
                self.model = booster
                print(f"Model loaded from {self.model_path} (GPU mode)")
            else:
                print(f"Model loaded from {self.model_path} (CPU mode)")
                
        except Exception as e:
            if self.use_gpu:
                try:
                    self.model = xgb.Booster()
                    self.model.load_model(self.model_path)
                    self.model.set_param({'predictor': 'gpu_predictor'})
                    print(f"Model loaded from {self.model_path} (GPU mode)")
                    return
                except:
                    pass
            
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {self.model_path} (CPU mode - GPU fallback failed)")
    
    def load_data(self) -> None:
        data_config = Config.get_data_config()
        d = kagglehub.dataset_download(data_config['dataset_name'])
        
        if self.use_gpu:
            products_df = cudf.read_csv(os.path.join(d, "products.csv"))
            self.product_info = products_df[['product_id', 'product_name']]
            
            orders_df = cudf.read_csv(os.path.join(d, "orders.csv"))
            self.test_orders = orders_df[orders_df['eval_set'] == 'test'][['order_id', 'user_id']]
            
            print("Data loaded successfully (GPU mode)")
        else:
            products_df = pd.read_csv(os.path.join(d, "products.csv"))
            self.product_info = products_df[['product_id', 'product_name']]
            
            orders_df = pd.read_csv(os.path.join(d, "orders.csv"))
            self.test_orders = orders_df[orders_df['eval_set'] == 'test'][['order_id', 'user_id']]
            
            print("Data loaded successfully (CPU mode)")
    
    def load_test_features(self) -> pd.DataFrame:
        test_features_path = "features/scaled_test_features.csv"
        validate_file_path(test_features_path)
        
        if self.use_gpu:
            test_df = cudf.read_csv(test_features_path)
            test_df = test_df.to_pandas()
            print(f"Loaded test features (GPU mode): {len(test_df)} samples")
        else:
            test_df = pd.read_csv(test_features_path)
            print(f"Loaded test features (CPU mode): {len(test_df)} samples")
        
        return test_df
    
    
    def prepare_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        feature_config = Config.get_feature_config()
        
        df = features_df.copy()
        for col in feature_config['categorical_features']:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.Categorical(df[col]).codes
        
        return df[feature_config['numerical_features'] + feature_config['categorical_features']]
    
    def predict(self, features_df: pd.DataFrame, batch_size: int = 10000) -> pd.DataFrame:
        data_config = Config.get_data_config()
        batch_size = batch_size or data_config.get('prediction_batch_size', 10000)
        
        expected_features = ['UP_avg_interval', 'UP_std_interval', 'UP_min_interval', 'UP_max_interval', 
                           'UP_last_abs_day', 'UP_purchase_count', 'user_last_abs_day', 'user_total_orders', 
                           'user_avg_days_between_orders', 'target_abs_day', 'UP_days_since_last', 
                           'UP_days_overdue', 'UP_overdue_zscore', 'UP_interval_progress', 'aisle_id', 
                           'orders', 'reorders', 'reorder_rate', 'total_items', 'total_distinct_items', 
                           'average_days_between_orders', 'average_basket', 'nb_orders', 'target_day_offset']
        
        feature_columns = [col for col in expected_features if col in features_df.columns]
        
        X = features_df[feature_columns]
        predictions = []
        
        for i in range(0, len(X), batch_size):
            batch_end = min(i + batch_size, len(X))
            X_batch = X.iloc[i:batch_end]
            
            if self.use_gpu and hasattr(self.model, 'predict'):
                dmatrix = xgb.DMatrix(X_batch)
                probabilities = self.model.predict(dmatrix)
            else:
                probabilities = self.model.predict_proba(X_batch)[:, 1]
            
            batch_results = pd.DataFrame({
                'user_id': features_df['user_id'].iloc[i:batch_end],
                'product_id': features_df['product_id'].iloc[i:batch_end],
                'reorder_probability': probabilities
            })
            
            predictions.append(batch_results)
            
            if i % (batch_size * 5) == 0:
                mode = "GPU" if self.use_gpu else "CPU"
                print(f"Processed {i:,}/{len(X):,} samples ({mode} mode)")
        
        all_predictions = pd.concat(predictions, ignore_index=True)
        
        filtered = all_predictions[
            all_predictions['reorder_probability'] > self.probability_threshold
        ]
        
        mode = "GPU" if self.use_gpu else "CPU"
        print(f"Generated {len(filtered)} predictions above threshold {self.probability_threshold} ({mode} mode)")
        return filtered
    
    def aggregate_predictions(self, predictions_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        if hasattr(self.product_info, 'to_pandas'):
            product_info_pd = self.product_info.to_pandas()
        else:
            product_info_pd = self.product_info
            
        if hasattr(self.test_orders, 'to_pandas'):
            test_orders_pd = self.test_orders.to_pandas()
        else:
            test_orders_pd = self.test_orders
        
        test_user_products = features_df.groupby('product_id').agg(
            count=('user_id', 'nunique')
        ).reset_index()
        
        result = test_user_products.merge(
            product_info_pd[['product_id', 'product_name']], 
            on='product_id'
        )
        
        return result[['product_id', 'product_name', 'count']].sort_values('count', ascending=False)
    
    def generate_stock_predictions(self, output_path: str = None) -> None:
        output_path = output_path or Config.DEFAULT_OUTPUT_PATH
        create_output_directory(output_path)
        
        print("Starting stock prediction generation...")
        self.load_model()
        self.load_data()
        
        features_df = self.load_test_features()
        predictions_df = self.predict(features_df)
        
        stock_df = self.aggregate_predictions(predictions_df, features_df)
        stock_df.to_csv(output_path, index=False)
        
        print(f"Stock predictions saved to {output_path}")
        print(f"Generated predictions for {len(stock_df)} products")
        print("\nTop 10 products by predicted reorder volume:")
        print(stock_df.head(10).to_string(index=False))


def main():
    predictor = InstacartPredictor(use_gpu=True, probability_threshold=0.01)
    predictor.generate_stock_predictions()


if __name__ == "__main__":
    main()
