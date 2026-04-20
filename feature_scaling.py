import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import os
import gc
import warnings
warnings.filterwarnings('ignore')

class FeatureScaler:
    def __init__(self, features_dir='features', scaler_type='standard', batch_size=100000):
        """
        Initialize feature scaler with batching support
        
        Args:
            features_dir: Directory containing original CSV files (also where scaled files will be saved)
            scaler_type: 'standard', 'minmax', or 'robust'
            batch_size: Number of rows to process at once
        """
        self.features_dir = features_dir
        self.batch_size = batch_size
        self.scaler_type = scaler_type
        self.scaler = None
        self.numeric_columns = None
        self.categorical_columns = None
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard', 'minmax', or 'robust'")
    
    def identify_column_types(self, sample_file, sample_size=10000):
        """Identify numeric and categorical columns using a sample"""
        print("Identifying column types...")
        
        exclude_from_scaling = ['user_id', 'product_id', 'department_id']
        
        sample_files = []
        train_file = os.path.join(self.features_dir, 'train_features.csv')
        test_file = os.path.join(self.features_dir, 'test_features.csv')
        
        if os.path.exists(train_file):
            sample_files.append(train_file)
        if os.path.exists(test_file):
            sample_files.append(test_file)
        
        all_columns = set()
        numeric_columns = set()
        categorical_columns = set()
        
        for file_path in sample_files:
            sample_df = pd.read_csv(file_path, nrows=sample_size)
            
            for col in sample_df.columns:
                all_columns.add(col)
                if col not in exclude_from_scaling and sample_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    numeric_columns.add(col)
                else:
                    categorical_columns.add(col)
            
            del sample_df
            gc.collect()
        
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)
        
        print(f"Excluded from scaling: {exclude_from_scaling}")
        print(f"Found {len(self.numeric_columns)} numeric columns")
        print(f"Found {len(self.categorical_columns)} categorical columns")
        print(f"Numeric columns: {self.numeric_columns}")
    
    def fit_scaler(self, file_path):
        """Fit the scaler on the entire dataset in batches"""
        print(f"Fitting {self.scaler_type} scaler on {file_path}...")
        
        total_rows = sum(1 for line in open(file_path)) - 1
        print(f"Total rows to process: {total_rows:,}")
        
        batch_count = 0
        for chunk in pd.read_csv(file_path, chunksize=self.batch_size):
            batch_count += 1
            print(f"Processing batch {batch_count} for fitting...")
            
            available_numeric_cols = [col for col in self.numeric_columns if col in chunk.columns]
            numeric_data = chunk[available_numeric_cols].values
            
            if hasattr(self.scaler, 'partial_fit'):
                self.scaler.partial_fit(numeric_data)
            else:
                if not hasattr(self, '_accumulated_data'):
                    self._accumulated_data = []
                self._accumulated_data.append(numeric_data)
                del chunk, numeric_data
                gc.collect()
        
        if not hasattr(self.scaler, 'partial_fit') and hasattr(self, '_accumulated_data'):
            print("Fitting scaler on accumulated data...")
            all_data = np.vstack(self._accumulated_data)
            self.scaler.fit(all_data)
            del self._accumulated_data, all_data
            gc.collect()
        
        print("Scaler fitting completed!")
    
    def process_file(self, input_file):
        """Process a single CSV file in batches and save scaled features"""
        filename = os.path.basename(input_file)
        output_file = os.path.join(self.features_dir, f"scaled_{filename}")
        
        print(f"Processing {input_file} -> {output_file}")
        
        total_rows = sum(1 for line in open(input_file)) - 1
        processed_rows = 0
        
        first_chunk = True
        for chunk in pd.read_csv(input_file, chunksize=self.batch_size):
            processed_rows += len(chunk)
            
            available_numeric_cols = [col for col in self.numeric_columns if col in chunk.columns]
            numeric_data = chunk[available_numeric_cols].values
            scaled_numeric = self.scaler.transform(numeric_data)
            
            chunk_scaled = chunk.copy()
            chunk_scaled[available_numeric_cols] = scaled_numeric
            
            if first_chunk:
                chunk_scaled.to_csv(output_file, index=False)
                first_chunk = False
            else:
                chunk_scaled.to_csv(output_file, mode='a', header=False, index=False)
            
            progress = (processed_rows / total_rows) * 100
            print(f"Progress: {progress:.1f}% ({processed_rows:,}/{total_rows:,} rows)")
            
            del chunk, numeric_data, scaled_numeric, chunk_scaled
            gc.collect()
        
        print(f"Completed processing {output_file}")
        return output_file
    
    def scale_all_features(self):
        """Scale all CSV files in the features directory"""
        print("Starting feature scaling process...")
        
        csv_files = []
        for f in os.listdir(self.features_dir):
            if f.endswith('.csv') and not f.startswith('scaled_'):
                scaled_version = f"scaled_{f}"
                if scaled_version not in os.listdir(self.features_dir):
                    csv_files.append(f)
                else:
                    print(f"Skipping {f} - scaled version already exists")
        
        if not csv_files:
            print("No CSV files found in features directory")
            return
        
        print(f"Found CSV files to scale: {csv_files}")
        
        first_file = os.path.join(self.features_dir, csv_files[0])
        self.identify_column_types(first_file)
        
        train_file = os.path.join(self.features_dir, 'train_features.csv')
        if os.path.exists(train_file):
            self.fit_scaler(train_file)
        else:
            self.fit_scaler(first_file)
        
        scaled_files = []
        for csv_file in csv_files:
            input_path = os.path.join(self.features_dir, csv_file)
            output_path = self.process_file(input_path)
            scaled_files.append(output_path)
        
        print("Feature scaling completed!")
        print(f"Scaled files saved to features directory:")
        for file in scaled_files:
            print(f"  - {file}")
        
        if hasattr(self.scaler, 'mean_'):
            print(f"\nScaler statistics:")
            print(f"Mean: {self.scaler.mean_}")
            if hasattr(self.scaler, 'scale_'):
                print(f"Scale: {self.scaler.scale_}")

def main():
    """Main function to run feature scaling"""
    FEATURES_DIR = 'features'
    SCALER_TYPE = 'standard'  # Options: 'standard', 'minmax', 'robust'
    BATCH_SIZE = 50000  # Reduced batch size to handle memory constraints
    
    print("Feature Scaling Pipeline")
    print("=" * 50)
    print(f"Features directory: {FEATURES_DIR}")
    print(f"Scaler type: {SCALER_TYPE}")
    print(f"Batch size: {BATCH_SIZE:,}")
    print("=" * 50)
    
    scaler = FeatureScaler(
        features_dir=FEATURES_DIR,
        scaler_type=SCALER_TYPE,
        batch_size=BATCH_SIZE
    )
    
    try:
        scaler.scale_all_features()
        print("\n✅ Feature scaling completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during feature scaling: {str(e)}")
        raise

if __name__ == "__main__":
    main()
