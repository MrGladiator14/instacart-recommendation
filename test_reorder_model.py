#!/usr/bin/env python3
"""
Test script for Instacart reorder probability model
Tests user-product pair reorder predictions
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class ReorderModelTester:
    """Test class for reorder probability model"""
    
    def __init__(self, path: str = "production_models/xgb_reorder_model.pkl"):
        self.path = path
        self.model = None
        self.products = None
        self.cat_cols = ['department', 'aisle']
        self.num_cols = ['purchase_count', 'ever_reordered']
        
    def load_model(self) -> bool:
        try:
            with open(self.path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"[SUCCESS] Model loaded from {self.path}")
            return True
        except FileNotFoundError:
            print(f"[ERROR] Model file not found: {self.path}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False
    
    def load_sample_data(self) -> bool:
        try:
            import kagglehub
            from kagglehub import KaggleDatasetAdapter
            
            data_dir = kagglehub.dataset_download("yasserh/instacart-online-grocery-basket-analysis-dataset")
            products_path = os.path.join(data_dir, "products.csv")
            aisles_path = os.path.join(data_dir, "aisles.csv")
            depts_path = os.path.join(data_dir, "departments.csv")
            
            products_df = pd.read_csv(products_path)
            aisles_df = pd.read_csv(aisles_path)
            departments_df = pd.read_csv(depts_path)
            
            self.products = products_df.merge(aisles_df, on='aisle_id').merge(departments_df, on='department_id')
            print(f"[SUCCESS] Loaded {len(self.products)} products")
            return True
            
        except Exception as e:
            print(f"[WARNING] Could not load sample data: {e}")
            self.products = pd.DataFrame({
                'product_id': [1, 2, 3, 4, 5],
                'product_name': ['Banana', 'Milk', 'Bread', 'Eggs', 'Apple'],
                'department': ['produce', 'dairy eggs', 'bakery', 'dairy eggs', 'produce'],
                'aisle': ['fresh fruits', 'milk', 'bread', 'eggs', 'fresh fruits']
            })
            print("[INFO] Using minimal sample data for testing")
            return True
    
    def prepare_features(self, uid: int, pid: int, pc: int, er: int) -> Optional[pd.DataFrame]:
        if self.products is None:
            print("[ERROR] Product info not loaded")
            return None
        
        product_data = self.products[self.products['product_id'] == pid]
        if product_data.empty:
            print(f"[ERROR] Product ID {pid} not found")
            return None
        
        product_row = product_data.iloc[0]
        
        features = pd.DataFrame({
            'purchase_count': [pc],
            'ever_reordered': [er],
            'department': [product_row['department']],
            'aisle': [product_row['aisle']]
        })
        
        for col in self.cat_cols:
            features[col] = features[col].astype('category')
        
        return features
    
    def predict_reorder_probability(self, uid: int, pid: int, pc: int, er: int) -> Optional[float]:
        if self.model is None:
            print("[ERROR] Model not loaded")
            return None
        
        features = self.prepare_features(uid, pid, pc, er)
        if features is None:
            return None
        
        try:
            probability = self.model.predict_proba(features)[:, 1][0]
            return probability
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return None
    
    def generate_test_cases(self) -> List[Dict]:
        if self.products is None:
            return []
        
        cases = []
        
        sample_products = []
        if len(self.products) > 0:
            for dept in self.products['department'].unique()[:5]:
                dept_products = self.products[self.products['department'] == dept]
                if not dept_products.empty:
                    sample_products.append(dept_products.iloc[0])
        
        scenarios = [
            {"name": "First-time buyer", "pc": 1, "er": 0},
            {"name": "Regular customer", "pc": 5, "er": 1},
            {"name": "Very loyal customer", "pc": 15, "er": 1},
            {"name": "One-time purchaser", "pc": 1, "er": 0},
            {"name": "Occasional buyer", "pc": 3, "er": 0},
        ]
        
        for i, product in enumerate(sample_products[:5]):
            for scenario in scenarios:
                cases.append({
                    "user_id": 1000 + i,
                    "product_id": int(product['product_id']),
                    "product_name": product['product_name'],
                    "department": product['department'],
                    "aisle": product['aisle'],
                    "purchase_count": scenario["pc"],
                    "ever_reordered": scenario["er"],
                    "scenario": scenario["name"]
                })
        
        return cases
    
    def run_single_test(self, case: Dict) -> Dict:
        result = case.copy()
        
        prob = self.predict_reorder_probability(
            case['user_id'],
            case['product_id'],
            case['purchase_count'],
            case['ever_reordered']
        )
        
        result['predicted_probability'] = prob
        result['status'] = 'success' if prob is not None else 'failed'
        
        return result
    
    def run_all_tests(self) -> List[Dict]:
        print("Running Reorder Probability Model Tests")
        print("=" * 60)
        
        if not self.load_model():
            return []
        
        if not self.load_sample_data():
            return []
        
        cases = self.generate_test_cases()
        if not cases:
            print("[ERROR] No test cases generated")
            return []
        
        results = []
        
        for i, case in enumerate(cases, 1):
            print(f"\nTest {i}: {case['scenario']} - {case['product_name']}")
            print("-" * 50)
            print(f"User ID: {case['user_id']}")
            print(f"Product: {case['product_name']} ({case['department']})")
            print(f"Purchase Count: {case['purchase_count']}")
            print(f"Ever Reordered: {case['ever_reordered']}")
            
            result = self.run_single_test(case)
            results.append(result)
            
            if result['status'] == 'success':
                prob = result['predicted_probability']
                print(f"Reorder Probability: {prob:.4f} ({prob*100:.2f}%)")
                
                if prob > 0.7:
                    interp = "High likelihood of reorder"
                elif prob > 0.4:
                    interp = "Moderate likelihood of reorder"
                else:
                    interp = "Low likelihood of reorder"
                print(f"Interpretation: {interp}")
            else:
                print("Status: FAILED")
        
        return results
    
    def interactive_test(self):
        print("\nInteractive Reorder Probability Testing")
        print("=" * 40)
        
        if not self.load_model() or not self.load_sample_data():
            return
        
        while True:
            print("\nEnter test details (or 'quit' to exit):")
            
            try:
                user_id = input("User ID: ").strip()
                if user_id.lower() == 'quit':
                    break
                
                product_id = input("Product ID: ").strip()
                if product_id.lower() == 'quit':
                    break
                
                purchase_count = input("Purchase Count: ").strip()
                if purchase_count.lower() == 'quit':
                    break
                
                ever_reordered = input("Ever Reordered (0/1): ").strip()
                if ever_reordered.lower() == 'quit':
                    break
                
                uid = int(user_id)
                pid = int(product_id)
                pc = int(purchase_count)
                er = int(ever_reordered)
                
                prob = self.predict_reorder_probability(uid, pid, pc, er)
                
                if prob is not None:
                    print(f"\nReorder Probability: {prob:.4f} ({prob*100:.2f}%)")
                else:
                    print("Prediction failed!")
                    
            except ValueError:
                print("Invalid input. Please enter numeric values.")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    
    def print_summary(self, results: List[Dict]):
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        
        if not results:
            print("No tests completed")
            return
        
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')
        
        print(f"Total tests: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/len(results)*100:.1f}%")
        
        if successful > 0:
            probs = [r['predicted_probability'] for r in results if r['status'] == 'success']
            print(f"\nProbability Statistics:")
            print(f"Mean: {np.mean(probs):.4f}")
            print(f"Min: {np.min(probs):.4f}")
            print(f"Max: {np.max(probs):.4f}")
            print(f"Std: {np.std(probs):.4f}")

def main():
    tester = ReorderModelTester()
    
    results = tester.run_all_tests()
    
    tester.print_summary(results)
    
    if results:
        choice = input("\nRun interactive tests? (y/n): ").strip().lower()
        if choice == 'y':
            tester.interactive_test()

if __name__ == "__main__":
    main()
