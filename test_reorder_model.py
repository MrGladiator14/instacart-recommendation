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
    
    def __init__(self, model_path: str = "production_models/lgbm_reorder_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.product_info = None
        self.categorical_cols = ['department', 'aisle']
        self.numerical_cols = ['purchase_count', 'ever_reordered']
        
    def load_model(self) -> bool:
        """Load the trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"[SUCCESS] Model loaded from {self.model_path}")
            return True
        except FileNotFoundError:
            print(f"[ERROR] Model file not found: {self.model_path}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False
    
    def load_sample_data(self) -> bool:
        """Load sample product data for testing"""
        try:
            # Try to load from the original data source
            import kagglehub
            from kagglehub import KaggleDatasetAdapter
            
            data_dir = kagglehub.dataset_download("yasserh/instacart-online-grocery-basket-analysis-dataset")
            products_path = os.path.join(data_dir, "products.csv")
            aisles_path = os.path.join(data_dir, "aisles.csv")
            depts_path = os.path.join(data_dir, "departments.csv")
            
            products_df = pd.read_csv(products_path)
            aisles_df = pd.read_csv(aisles_path)
            departments_df = pd.read_csv(depts_path)
            
            self.product_info = products_df.merge(aisles_df, on='aisle_id').merge(departments_df, on='department_id')
            print(f"[SUCCESS] Loaded {len(self.product_info)} products")
            return True
            
        except Exception as e:
            print(f"[WARNING] Could not load sample data: {e}")
            # Create minimal sample data for testing
            self.product_info = pd.DataFrame({
                'product_id': [1, 2, 3, 4, 5],
                'product_name': ['Banana', 'Milk', 'Bread', 'Eggs', 'Apple'],
                'department': ['produce', 'dairy eggs', 'bakery', 'dairy eggs', 'produce'],
                'aisle': ['fresh fruits', 'milk', 'bread', 'eggs', 'fresh fruits']
            })
            print("[INFO] Using minimal sample data for testing")
            return True
    
    def prepare_features(self, user_id: int, product_id: int, purchase_count: int, ever_reordered: int) -> Optional[pd.DataFrame]:
        """Prepare features for a user-product pair"""
        if self.product_info is None:
            print("[ERROR] Product info not loaded")
            return None
        
        # Get product metadata
        product_data = self.product_info[self.product_info['product_id'] == product_id]
        if product_data.empty:
            print(f"[ERROR] Product ID {product_id} not found")
            return None
        
        product_row = product_data.iloc[0]
        
        # Create feature DataFrame
        features = pd.DataFrame({
            'purchase_count': [purchase_count],
            'ever_reordered': [ever_reordered],
            'department': [product_row['department']],
            'aisle': [product_row['aisle']]
        })
        
        # Convert categorical columns to category type
        for col in self.categorical_cols:
            features[col] = features[col].astype('category')
        
        return features
    
    def predict_reorder_probability(self, user_id: int, product_id: int, purchase_count: int, ever_reordered: int) -> Optional[float]:
        """Predict reorder probability for a user-product pair"""
        if self.model is None:
            print("[ERROR] Model not loaded")
            return None
        
        features = self.prepare_features(user_id, product_id, purchase_count, ever_reordered)
        if features is None:
            return None
        
        try:
            probability = self.model.predict_proba(features)[:, 1][0]
            return probability
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return None
    
    def generate_test_cases(self) -> List[Dict]:
        """Generate diverse test cases"""
        if self.product_info is None:
            return []
        
        test_cases = []
        
        # Get sample products from different departments
        sample_products = []
        if len(self.product_info) > 0:
            for dept in self.product_info['department'].unique()[:5]:
                dept_products = self.product_info[self.product_info['department'] == dept]
                if not dept_products.empty:
                    sample_products.append(dept_products.iloc[0])
        
        # Create test scenarios
        scenarios = [
            {"name": "First-time buyer", "purchase_count": 1, "ever_reordered": 0},
            {"name": "Regular customer", "purchase_count": 5, "ever_reordered": 1},
            {"name": "Very loyal customer", "purchase_count": 15, "ever_reordered": 1},
            {"name": "One-time purchaser", "purchase_count": 1, "ever_reordered": 0},
            {"name": "Occasional buyer", "purchase_count": 3, "ever_reordered": 0},
        ]
        
        for i, product in enumerate(sample_products[:5]):
            for scenario in scenarios:
                test_cases.append({
                    "user_id": 1000 + i,
                    "product_id": int(product['product_id']),
                    "product_name": product['product_name'],
                    "department": product['department'],
                    "aisle": product['aisle'],
                    "purchase_count": scenario["purchase_count"],
                    "ever_reordered": scenario["ever_reordered"],
                    "scenario": scenario["name"]
                })
        
        return test_cases
    
    def run_single_test(self, test_case: Dict) -> Dict:
        """Run a single test case"""
        result = test_case.copy()
        
        probability = self.predict_reorder_probability(
            test_case['user_id'],
            test_case['product_id'],
            test_case['purchase_count'],
            test_case['ever_reordered']
        )
        
        result['predicted_probability'] = probability
        result['status'] = 'success' if probability is not None else 'failed'
        
        return result
    
    def run_all_tests(self) -> List[Dict]:
        """Run all test cases"""
        print("Running Reorder Probability Model Tests")
        print("=" * 60)
        
        if not self.load_model():
            return []
        
        if not self.load_sample_data():
            return []
        
        test_cases = self.generate_test_cases()
        if not test_cases:
            print("[ERROR] No test cases generated")
            return []
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['scenario']} - {test_case['product_name']}")
            print("-" * 50)
            print(f"User ID: {test_case['user_id']}")
            print(f"Product: {test_case['product_name']} ({test_case['department']})")
            print(f"Purchase Count: {test_case['purchase_count']}")
            print(f"Ever Reordered: {test_case['ever_reordered']}")
            
            result = self.run_single_test(test_case)
            results.append(result)
            
            if result['status'] == 'success':
                prob = result['predicted_probability']
                print(f"Reorder Probability: {prob:.4f} ({prob*100:.2f}%)")
                
                # Interpret probability
                if prob > 0.7:
                    interpretation = "High likelihood of reorder"
                elif prob > 0.4:
                    interpretation = "Moderate likelihood of reorder"
                else:
                    interpretation = "Low likelihood of reorder"
                print(f"Interpretation: {interpretation}")
            else:
                print("Status: FAILED")
        
        return results
    
    def interactive_test(self):
        """Interactive testing mode"""
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
                
                # Convert to appropriate types
                user_id = int(user_id)
                product_id = int(product_id)
                purchase_count = int(purchase_count)
                ever_reordered = int(ever_reordered)
                
                # Make prediction
                probability = self.predict_reorder_probability(user_id, product_id, purchase_count, ever_reordered)
                
                if probability is not None:
                    print(f"\nReorder Probability: {probability:.4f} ({probability*100:.2f}%)")
                else:
                    print("Prediction failed!")
                    
            except ValueError:
                print("Invalid input. Please enter numeric values.")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    
    def print_summary(self, results: List[Dict]):
        """Print test summary"""
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
            probabilities = [r['predicted_probability'] for r in results if r['status'] == 'success']
            print(f"\nProbability Statistics:")
            print(f"Mean: {np.mean(probabilities):.4f}")
            print(f"Min: {np.min(probabilities):.4f}")
            print(f"Max: {np.max(probabilities):.4f}")
            print(f"Std: {np.std(probabilities):.4f}")

def main():
    """Main function"""
    tester = ReorderModelTester()
    
    # Run automated tests
    results = tester.run_all_tests()
    
    # Print summary
    tester.print_summary(results)
    
    # Offer interactive mode
    if results:
        choice = input("\nRun interactive tests? (y/n): ").strip().lower()
        if choice == 'y':
            tester.interactive_test()

if __name__ == "__main__":
    main()
