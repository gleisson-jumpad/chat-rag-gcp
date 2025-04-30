#!/bin/bash

# 🔧 Garante que os módulos da pasta /app/app/ sejam visíveis nos imports
export PYTHONPATH=$PYTHONPATH:/app/app

# Set Streamlit's server port to match the PORT environment variable
export PORT="${PORT:-8501}"

# Print environment info
echo "Starting application with INSTANCE_CONNECTION_NAME: $INSTANCE_CONNECTION_NAME"
echo "Database: $PG_DB"
echo "Using port: $PORT"

# Run Streamlit
streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0
