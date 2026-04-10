#!/usr/bin/env python3
"""
Command-line tool for predicting reorder probability
Usage: python reorder_predictor.py --user_id 123 --product_id 456 --purchase_count 3 --ever_reordered 1
"""

import argparse
import sys
import os
from test_reorder_model import ReorderModelTester

def main():
    parser = argparse.ArgumentParser(description='Predict reorder probability for user-product pairs')
    parser.add_argument('--user_id', type=int, required=True, help='User ID')
    parser.add_argument('--product_id', type=int, required=True, help='Product ID')
    parser.add_argument('--purchase_count', type=int, required=True, help='Number of times user purchased this product')
    parser.add_argument('--ever_reordered', type=int, required=True, choices=[0, 1], help='Whether user ever reordered this product (0 or 1)')
    parser.add_argument('--model_path', type=str, default='production_models/lgbm_reorder_model.pkl', help='Path to trained model')
    
    args = parser.parse_args()
    
    tester = ReorderModelTester(args.model_path)
    
    if not tester.load_model():
        print(f"Error: Could not load model from {args.model_path}")
        sys.exit(1)
    
    if not tester.load_sample_data():
        print("Error: Could not load sample data")
        sys.exit(1)
    
    prob = tester.predict_reorder_probability(
        args.user_id,
        args.product_id,
        args.purchase_count,
        args.ever_reordered
    )
    
    if prob is not None:
        print(f"Reorder Probability: {prob:.4f} ({prob*100:.2f}%)")
        
        if prob > 0.7:
            print("Interpretation: High likelihood of reorder")
        elif prob > 0.4:
            print("Interpretation: Moderate likelihood of reorder")
        else:
            print("Interpretation: Low likelihood of reorder")
    else:
        print("Error: Prediction failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
