#!/usr/bin/env bash
# Backend startup script for Render.com

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main_api:app --host 0.0.0.0 --port $PORT