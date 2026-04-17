#!/bin/bash
# Deployment setup script for Streamlit Cloud / Hugging Face

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p .streamlit

# Run the app
streamlit run app.py --server.port=8501 --server.address=0.0.0.0