#!/usr/bin/env python3
"""
FastAPI server with Ray Serve for Instacart reorder probability model
Provides REST API endpoints for predictions with GPU acceleration
"""

import os
import logging
import pickle
import pandas as pd
import numpy as np
import argparse
import sys
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn
import ray
from ray import serve
from ray.serve import Deployment
from ray.serve.handle import DeploymentHandle
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

handle: Optional[DeploymentHandle] = None

# Configure GPU based on environment
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
    
    def _load_model(self) -> bool:
        try:
            with open(self.path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"[SUCCESS] Model loaded from {self.path}")
            return True
        except FileNotFoundError:
            print(f"[ERROR] Model file not found: {self.path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _load_sample_data(self) -> bool:
        self.products = pd.DataFrame({
            'product_id': [1, 2, 3, 4, 5],
            'product_name': ['Banana', 'Milk', 'Bread', 'Eggs', 'Apple'],
            'department': ['produce', 'dairy eggs', 'bakery', 'dairy eggs', 'produce'],
            'aisle': ['fresh fruits', 'milk', 'bread', 'eggs', 'fresh fruits']
        })
        return True
    
    def _prepare_features(self, u: int, pid: int, pc: int, er: int) -> Optional[pd.DataFrame]:
        if self.products is None:
            print("[ERROR] Product info not loaded")
            return None
        
        product_data = self.products[self.products['product_id'] == pid]
        if product_data.empty:
            logger.error(f"Product ID {pid} not found")
            return None
        
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
    
    async def predict_reorder_probability(self, u: int, pid: int, pc: int, er: int) -> Optional[float]:
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        features = self._prepare_features(u, pid, pc, er)
        if features is None:
            return None
        
        try:
            probability = self.model.predict_proba(features)[:, 1][0]
            return probability
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    async def get_products(self, limit: int = 20) -> Dict[str, Any]:
        if self.products is None:
            return {"products": [], "total_available": 0}
        
        sample_products = self.products.head(limit)
        
        products = []
        for _, row in sample_products.iterrows():
            products.append({
                "product_id": int(row['product_id']),
                "product_name": str(row['product_name']),
                "department": str(row['department']),
                "aisle": str(row['aisle'])
            })
        
        return {
            "products": products,
            "total_available": len(self.products)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        gpu_resources = ray.get_runtime_context().get_resource_ids().get("GPU", [])
        return {
            "model_loaded": self.model is not None,
            "data_loaded": self.products is not None,
            "gpu_available": len(gpu_resources) > 0
        }

class PredictionRequest(BaseModel):
    user_id: int = Field(..., description="User ID", example=12345)
    product_id: int = Field(..., description="Product ID", example=24852)
    purchases: int = Field(..., description="Number of times user purchased this product", example=3)
    reordered: int = Field(..., description="Whether user ever reordered this product (0 or 1)", example=1)

class PredictionResponse(BaseModel):
    user_id: int
    product_id: int
    reorder_probability: float
    interpretation: str
    status: str = "success"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_loaded: bool

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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    global handle
    
    if handle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model deployment not available"
        )
    
    try:
        deployment_health = await handle.health_check.remote()
        
        return HealthResponse(
            status="healthy",
            model_loaded=deployment_health["model_loaded"],
            data_loaded=deployment_health["data_loaded"]
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

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
        
        if probability is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction failed"
            )
        
        if probability > 0.7:
            interpretation = "High likelihood of reorder"
        elif probability > 0.4:
            interpretation = "Moderate likelihood of reorder"
        else:
            interpretation = "Low likelihood of reorder"
        
        return PredictionResponse(
            user_id=request.user_id,
            product_id=request.product_id,
            reorder_probability=round(probability, 4),
            interpretation=interpretation
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/products", response_model=Dict[str, Any])
async def get_products():
    global handle
    
    if handle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model deployment not available"
        )
    
    try:
        products_data = await handle.get_products.remote()
        return products_data
        
    except Exception as e:
        logger.error(f"Failed to get products: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to get products: {str(e)}"
        )


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
