# Instacart Product Reorder Prediction

ML pipeline to predict product reorder probability using LightGBM.

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
