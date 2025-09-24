# Tiki Book Chatbot - Separated Architecture

This project has been refactored to separate frontend and backend components for independent deployment.

## Architecture

```
chatbot_ecommerce/
├── frontend/          # Streamlit UI Application
│   ├── app.py        # Main Streamlit app
│   ├── streamlit_app.py  # Demo version
│   ├── utils_local.py    # Frontend utilities
│   ├── requirements.txt  # Frontend dependencies
│   └── README.md
├── backend/           # FastAPI Server
│   ├── main_api.py   # FastAPI application
│   ├── utils.py      # Backend utilities & RAG pipeline
│   ├── requirements.txt  # Backend dependencies
│   └── README.md
├── shared/            # Shared utilities & scripts
│   ├── benchmark_*.py    # Performance benchmarking
│   ├── build_index.py    # FAISS index building
│   └── other utilities
└── data/              # Data files (CSV, FAISS index)
    ├── tiki_books_processed_for_rag.csv
    └── tiki_books_faiss.index
```

## Quick Start

### Frontend (Streamlit)

```bash
cd frontend
pip install -r requirements.txt
cp .env.example .env  # Add your GEMINI_API_KEY
streamlit run app.py
```

### Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # Add your GEMINI_API_KEY
uvicorn main_api:app --reload
```

## Deployment

### Frontend Deployment Options:

- **Streamlit Cloud**: Direct deployment from GitHub
- **Vercel**: Static/serverless deployment
- **Heroku**: Container deployment

### Backend Deployment Options:

- **Render.com**: Recommended for FastAPI
- **Railway**: Container deployment
- **Google Cloud Run**: Serverless containers

## Environment Variables

Both frontend and backend require:

- `GEMINI_API_KEY`: Google Gemini AI API key

## Data Requirements

Ensure the `data/` folder contains:

- `tiki_books_processed_for_rag.csv` - Processed book data
- `tiki_books_faiss.index` - FAISS vector index

## Changes Made

1. **Separated codebase**: Frontend (Streamlit) and Backend (FastAPI) are now independent
2. **Updated import paths**: Fixed relative imports and data paths
3. **Separate dependencies**: Each component has its own requirements.txt
4. **Deployment configs**: Added startup scripts and Docker configs
5. **Documentation**: Individual READMEs for each component

This separation allows for:

- Independent scaling and deployment
- Different hosting platforms (Vercel for frontend, Render for backend)
- Better maintainability and development workflow
