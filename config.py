from typing import Dict, Any


class Config:
    """Configuration class for the Instacart recommendation system."""
    
    MODEL_PATH = 'production_models/xgb_reorder_model.pkl'
    PROBABILITY_THRESHOLD = 0.2
    
    DATASET_NAME = "yasserh/instacart-online-grocery-basket-analysis-dataset"
    
    # Output configuration
    DEFAULT_OUTPUT_PATH = 'stock.csv'
    
    NUMERICAL_FEATURES = ['purchase_count', 'ever_reordered']
    CATEGORICAL_FEATURES = ['department', 'aisle']
    
    PREDICTION_BATCH_SIZE = 100_000
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'model_path': cls.MODEL_PATH,
            'probability_threshold': cls.PROBABILITY_THRESHOLD
        }
    
    @classmethod
    def get_data_config(cls) -> Dict[str, Any]:
        """Get data processing configuration."""
        return {
            'dataset_name': cls.DATASET_NAME,
            'prediction_batch_size': cls.PREDICTION_BATCH_SIZE
        }
    
    @classmethod
    def get_feature_config(cls) -> Dict[str, list]:
        """Get feature configuration."""
        return {
            'numerical_features': cls.NUMERICAL_FEATURES,
            'categorical_features': cls.CATEGORICAL_FEATURES
        }
