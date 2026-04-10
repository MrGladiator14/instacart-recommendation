#!/usr/bin/env python3
"""
Streamlit application for Instacart reorder prediction
Professional UI with direct model integration
"""

import streamlit as st
import pandas as pd
import requests
from typing import Dict, Any
import time

st.set_page_config(
    page_title="Instacart Reorder Predictor",
    page_icon=":shopping_cart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ModelPredictor:
    def __init__(self, url: str = None):
        if url is None:
            raise ValueError("API URL must be provided by user")
        if not url.startswith(('http://', 'https://')):
            raise ValueError("API URL must start with http:// or https://")
        
        self.base_url = url.rstrip('/')
        if self.base_url.endswith('/predict'):
            self.base_url = self.base_url[:-8]  # Remove '/predict'
        
        self.api_url = self.base_url
        self.products = None
        self.load_sample_data()
    
    def test_api_connection(self) -> tuple[bool, str]:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return True, "Connection successful"
        except:
            pass
        
        try:
            payload = {
                "user_id": 1,
                "product_id": 1,
                "purchases": 1,
                "reordered": 0
            }
            response = requests.post(f"{self.base_url}/predict", 
                                   json=payload, timeout=10)
            if response.status_code == 200:
                return True, "Connection successful"
            else:
                return False, f"HTTP {response.status_code}: {response.text[:100]}"
        except requests.exceptions.Timeout:
            return False, "Connection timeout (10s)"
        except requests.exceptions.ConnectionError:
            return False, "Connection error - URL may be unreachable"
        except requests.exceptions.RequestException as e:
            return False, f"Request error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def load_sample_data(self) -> bool:
        self.products = pd.DataFrame({
            'product_id': [1, 2, 3, 4, 5],
            'product_name': ['Banana', 'Milk', 'Bread', 'Eggs', 'Apple'],
            'department': ['produce', 'dairy eggs', 'bakery', 'dairy eggs', 'produce'],
            'aisle': ['fresh fruits', 'milk', 'bread', 'eggs', 'fresh fruits']
        })
        return True
    
    def predict_reorder(self, uid: int, pid: int, pc: int, er: int) -> Dict[str, Any]:
        try:
            payload = {
                "user_id": uid,
                "product_id": pid,
                "purchases": pc,
                "reordered": er
            }
            
            response = requests.post(f"{self.base_url}/predict", 
                                   json=payload, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API request failed with status {response.status_code}: {response.text[:100]}"}
        except Exception as e:
            return {"error": f"API prediction failed: {e}"}
    
    def get_products(self, limit: int = 20) -> Dict[str, Any]:
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
    
    def health_check(self) -> Dict[str, Any]:
        api_connected, error_msg = self.test_api_connection()
        return {
            "api_connected": api_connected,
            "data_loaded": self.products is not None,
            "error_message": error_msg if not api_connected else None
        }

def initialize_session_state():
    if 'products' not in st.session_state:
        st.session_state.products = []
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'api_url' not in st.session_state:
        st.session_state.api_url = ""

def render_sidebar():
    st.sidebar.title("Configuration")
    
    url = st.sidebar.text_input(
        "API Base URL *",
        value=st.session_state.get('api_url', ''),
        help="Base API URL (http/https) - e.g., https://nancy-remained-cruise-barrier.trycloudflare.com"
    )
    
    st.session_state.api_url = url
    
    if st.sidebar.button("Connect to API", type="primary", disabled=not url):
        with st.sidebar.spinner("Connecting to API..."):
            try:
                st.session_state.model_predictor = ModelPredictor(url)
                health = st.session_state.model_predictor.health_check()
                if health["api_connected"]:
                    st.sidebar.success("✅ API connected successfully!")
                    st.rerun()
                else:
                    st.sidebar.error(f"❌ Connection failed: {health.get('error_message', 'Unknown error')}")
            except ValueError as e:
                st.sidebar.error(f"❌ Invalid URL: {e}")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to connect: {e}")
    
    if 'model_predictor' in st.session_state:
        with st.sidebar:
            st.subheader("API Status")
            health = st.session_state.model_predictor.health_check()
            
            if health["api_connected"] and health["data_loaded"]:
                st.success("API Connected")
            else:
                st.error("API Not Ready")
                if not health["api_connected"]:
                    st.sidebar.markdown(f"**Issue:** {health.get('error_message', 'API not connected')}")
                if not health["data_loaded"]:
                    st.sidebar.markdown("**Issue:** Data not loaded")
        
        with st.sidebar:
            st.subheader("API Settings")
            st.info(f"**Base URL:** {st.session_state.model_predictor.base_url}")
            st.info(f"**Predict Endpoint:** {st.session_state.model_predictor.base_url}/predict")
            if st.session_state.model_predictor.products is not None:
                st.info(f"**Products Available:** {len(st.session_state.model_predictor.products)}")
    else:
        st.sidebar.warning("Please provide an API URL and click 'Connect to API' to begin")

def render_prediction_form():
    st.header("Reorder Prediction")
    
    if 'model_predictor' not in st.session_state:
        st.warning("Please connect to an API from the sidebar before making predictions.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        uid = st.number_input(
            "User ID",
            min_value=1,
            value=12345,
            help="Unique identifier for the user"
        )
        
        pid = st.number_input(
            "Product ID",
            min_value=1,
            value=24852,
            help="Unique identifier for the product"
        )
    
    with col2:
        pc = st.number_input(
            "Purchase Count",
            min_value=0,
            value=3,
            help="Number of times user purchased this product"
        )
        
        er = st.selectbox(
            "Ever Reordered",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Whether user has ever reordered this product"
        )
    
    if st.button("Predict Reorder Probability", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            result = st.session_state.model_predictor.predict_reorder(
                uid, pid, pc, er
            )
            
            if "error" in result:
                st.error(f"Prediction failed: {result['error']}")
            else:
                render_prediction_result(result)
                st.session_state.history.append({
                    "timestamp": time.time(),
                    "user_id": uid,
                    "product_id": pid,
                    "purchase_count": pc,
                    "ever_reordered": er,
                    "result": result
                })

def render_prediction_result(res: Dict[str, Any]):
    st.success("Prediction completed!")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.metric(
            "User ID",
            res["user_id"]
        )
    
    with col2:
        prob = res["reorder_probability"]
        st.metric(
            "Reorder Probability",
            f"{prob:.4f} ({prob*100:.2f}%)",
            delta=None
        )
    
    with col3:
        st.metric(
            "Product ID",
            res["product_id"]
        )
    
    interp = res["interpretation"]
    if "High" in interp:
        st.info(f"**Interpretation:** {interp} :arrow_up:")
    elif "Moderate" in interp:
        st.warning(f"**Interpretation:** {interp} :arrow_right:")
    else:
        st.error(f"**Interpretation:** {interp} :arrow_down:")
    
    with st.expander("View Raw Response"):
        st.json(res)

def render_products_section():
    st.header("Available Products")
    
    if 'model_predictor' not in st.session_state:
        st.warning("Please connect to an API from the sidebar before viewing products.")
        return
    
    if st.button("Load Products", use_container_width=False):
        with st.spinner("Loading products..."):
            data = st.session_state.model_predictor.get_products()
            
            if "error" in data:
                st.error(f"Failed to load products: {data['error']}")
            else:
                st.session_state.products = data.get("products", [])
                st.success(f"Loaded {len(st.session_state.products)} products")
    
    if st.session_state.products:
        df = pd.DataFrame(st.session_state.products)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        selected_product = st.selectbox(
            "Select a product for quick prediction",
            options=st.session_state.products,
            format_func=lambda x: f"{x['product_name']} (ID: {x['product_id']})"
        )

def render_history_section():
    if not st.session_state.history:
        st.info("No prediction history yet. Make a prediction to see it here!")
        return
    
    st.header("Prediction History")
    
    data = []
    for entry in st.session_state.history:
        data.append({
            "Timestamp": pd.to_datetime(entry["timestamp"], unit='s').strftime("%Y-%m-%d %H:%M:%S"),
            "User ID": entry["user_id"],
            "Product ID": entry["product_id"],
            "Purchase Count": entry["purchase_count"],
            "Ever Reordered": "Yes" if entry["ever_reordered"] == 1 else "No",
            "Probability": f"{entry['result']['reorder_probability']:.4f}",
            "Interpretation": entry["result"]["interpretation"]
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    if st.button("Clear History", type="secondary"):
        st.session_state.history = []
        st.rerun()

def render_about_section():
    st.markdown("---")
    st.markdown("""
    ### About This Application
    
    This professional Streamlit application provides an intuitive interface for the Instacart 
    Reorder Prediction API. It uses machine learning to predict the likelihood of a customer 
    reordering a product based on their purchase history.
    
    **Features:**
    - Real-time predictions with probability scores
    - Product catalog browsing
    - Prediction history tracking
    - Professional UI with responsive design
    """)

def main():
    initialize_session_state()
    
    st.title(":shopping_cart: Instacart Reorder Predictor")
    
    st.markdown("""
    Predict the likelihood of customers reordering products with our machine learning model.
    Enter user and product information below to get instant predictions.
    """)
    
    render_sidebar()
    
    t1, t2, t3, t4 = st.tabs(["Prediction", "Products", "History", "About"])
    
    with t1:
        render_prediction_form()
    
    with t2:
        render_products_section()
    
    with t3:
        render_history_section()
    
    with t4:
        render_about_section()

if __name__ == "__main__":
    main()
