# Instacart Product Reorder Prediction

ML pipeline to predict product reorder probability using LightGBM.

## Project Structure

```
instacart-recommendation/
├── api_server.py          # FastAPI server for model inference
├── config.py              # Configuration settings
├── predict.py             # Prediction script with InstacartPredictor class
├── train.py               # Model training script
├── utils.py               # Utility functions
├── streamlit_app.py       # Streamlit web interface
├── test_reorder_model.py  # Model testing script
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration and dependencies
├── .env.example           # Environment variables template
├── stock.csv              # Sample data for predictions
├── production_models/     # Trained model files
│   ├── lgbm_reorder_model.pkl
│   ├── lgbm_reorder_model_sklearn16.pkl
│   └── xgb_reorder_model.pkl
├── logs/                  # Training and GPU logs
│   ├── logs.gpu.txt
│   └── train.log.txt
└── .github/workflows/     # CI/CD deployment workflows
    └── deploy-streamlit.yml
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run API server (default: http://localhost:8000)
uv run api_server.py

# Or with uvicorn options
uv run uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

# Test prediction
uv run python api_server.py --user_id 123 --product_id 1 --purchase_count 5 --ever_reordered 1
```

## Running Inference and Predictions

### Using the Prediction Script

```bash
# Basic prediction using default model
uv run python predict.py

# Specify custom model and data
uv run python predict.py --model_path production_models/lgbm_reorder_model.pkl --data_path stock.csv

# Adjust prediction threshold
uv run python predict.py --probability_threshold 0.3

# Disable GPU (use CPU)
uv run python predict.py --use_gpu False
```

### Using the API Server

```bash
# Start the API server
uv run api_server.py

# Make prediction requests via curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "product_id": 1,
    "purchase_count": 5,
    "ever_reordered": 1
  }'

# Batch predictions
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {"user_id": 123, "product_id": 1, "purchase_count": 5, "ever_reordered": 1},
      {"user_id": 456, "product_id": 2, "purchase_count": 3, "ever_reordered": 0}
    ]
  }'
```

### Using Streamlit Interface

```bash
# Launch the web interface
uv run streamlit run streamlit_app.py

# Access at http://localhost:8501
```

### Model Training

```bash
# Train a new model
uv run python train.py

# Train with specific parameters
uv run python train.py --model_type lgbm --use_gpu True --test_size 0.2
```

## Deployment

```bash
# Install cloudflared
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# Expose local server
cloudflared tunnel --url http://localhost:8000
```

## Pipeline Overview

**Data**: Instacart dataset (orders, products, aisles, departments)

**Features**:

- `purchase_count`: User's purchase frequency for product
- `ever_reordered`: Historical reorder flag
- `department`, `aisle`: Categorical metadata

**Model**: LightGBM classifier (binary: reordered/not reordered)

**Metrics**: ROC-AUC, precision, recall, F1-score

**Output**: Pickled model for production inference
