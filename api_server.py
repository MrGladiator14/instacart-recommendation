#!/usr/bin/env python3
"""
FastAPI server with Ray Serve for Instacart reorder probability model
"""

import os
import logging
import pickle
import pandas as pd
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn
import ray
from ray import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

handle: Optional[serve.DeploymentHandle] = None
gpu_available = os.getenv("GPU_AVAILABLE", "false").lower() == "true"

@serve.deployment(
    name="reorder-model",
    num_replicas=1,
    ray_actor_options={"num_gpus": 1} if gpu_available else {},
)
class ReorderModelDeployment:
    """Ray Serve deployment for the reorder model"""
    
    def __init__(self, path: str = "production_models/xgb_reorder_model.pkl"):
        self.path = path
        self.categorical_cols = ['department', 'aisle']
        self.model = None
        self.products = None
        self._load_model()
        self._load_sample_data()
    
    def _load_model(self):
        try:
            with open(self.path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {self.path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_sample_data(self):
        self.products = pd.DataFrame({
            'product_id': [1, 2, 3, 4, 5],
            'product_name': ['Banana', 'Milk', 'Bread', 'Eggs', 'Apple'],
            'department': ['produce', 'dairy eggs', 'bakery', 'dairy eggs', 'produce'],
            'aisle': ['fresh fruits', 'milk', 'bread', 'eggs', 'fresh fruits']
        })
    
    def _prepare_features(self, u: int, pid: int, pc: int, er: int) -> pd.DataFrame:
        product_data = self.products[self.products['product_id'] == pid]
        if product_data.empty:
            raise ValueError(f"Product ID {pid} not found")
        
        product_row = product_data.iloc[0]
        
        features = pd.DataFrame({
            'purchase_count': [pc],
            'ever_reordered': [er],
            'department': [product_row['department']],
            'aisle': [product_row['aisle']]
        })
        
        for col in self.categorical_cols:
            features[col] = features[col].astype('category')
        
        return features
    
    async def predict_reorder_probability(self, u: int, pid: int, pc: int, er: int) -> float:
        features = self._prepare_features(u, pid, pc, er)
        return self.model.predict_proba(features)[:, 1][0]
    
    async def get_products(self, limit: int = 20) -> Dict[str, Any]:
        sample_products = self.products.head(limit)
        
        return {
            "products": sample_products.to_dict('records'),
            "total_available": len(self.products)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        return {
            "model_loaded": self.model is not None,
            "data_loaded": self.products is not None,
            "gpu_available": gpu_available
        }

class PredictionRequest(BaseModel):
    user_id: int = Field(..., description="User ID", example=123)
    product_id: int = Field(..., description="Product ID", example=1)
    purchases: int = Field(..., description="Number of times user purchased this product", example=3)
    reordered: int = Field(..., description="Whether user ever reordered this product (0 or 1)", example=1)

class PredictionResponse(BaseModel):
    user_id: int
    product_id: int
    reorder_probability: float
    interpretation: str
    status: str = "success"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global handle
    
    try:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        serve.start(http_options={"host": "127.0.0.1", "port": 8001})
        
        path = os.getenv("MODEL_PATH", "production_models/xgb_reorder_model.pkl")
        model_deployment = ReorderModelDeployment.bind(path)
        serve.run(model_deployment, name="reorder-model-app")
        
        handle = serve.get_app_handle("reorder-model-app")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize Ray Serve: {e}")
        raise
    
    finally:
        try:
            serve.shutdown()
            if ray.is_initialized():
                ray.shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

app = FastAPI(
    title="Instacart Reorder Prediction API",
    description="API for predicting product reorder probability",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", response_model=Dict[str, Any])
async def root():
    return {
        "message": "Instacart Reorder Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    if handle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model deployment not available"
        )
    
    return await handle.health_check.remote()

@app.post("/predict", response_model=PredictionResponse)
async def predict_reorder(request: PredictionRequest):
    global handle
    
    if handle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model deployment not available"
        )
    
    if request.reordered not in [0, 1]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="reordered must be 0 or 1"
        )
    
    try:
        probability = await handle.predict_reorder_probability.remote(
            request.user_id,
            request.product_id,
            request.purchases,
            request.reordered
        )
        
        
        interpretation = (
            "High likelihood of reorder" if probability > 0.7 else
            "Moderate likelihood of reorder" if probability > 0.4 else
            "Low likelihood of reorder"
        )
        
        return PredictionResponse(
            user_id=request.user_id,
            product_id=request.product_id,
            reorder_probability=round(probability, 4),
            interpretation=interpretation
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )

@app.get("/products")
async def get_products():
    if handle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model deployment not available"
        )
    
    return await handle.get_products.remote()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
