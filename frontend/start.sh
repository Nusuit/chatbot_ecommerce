# Frontend start script
#!/usr/bin/env bash

# Install dependencies
pip install -r requirements.txt

# Start Streamlit app
streamlit run app.py --server.port $PORT --server.address 0.0.0.0