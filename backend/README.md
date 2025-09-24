# Tiki Book Chatbot - Backend API

FastAPI backend service for the Tiki Book Chatbot with RAG (Retrieval-Augmented Generation) capabilities.

## Features

- FastAPI REST API endpoints
- RAG pipeline with FAISS vector search
- Google Gemini AI integration
- Book recommendation system

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.env` file:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

3. Ensure the `data/` folder exists in the parent directory with:
   - `tiki_books_processed_for_rag.csv`
   - `tiki_books_faiss.index`

## Local Development

```bash
uvicorn main_api:app --reload --host 0.0.0.0 --port 8000
```

## Deployment on Render.com

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn main_api:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `GEMINI_API_KEY`
6. Deploy

## API Endpoints

- `POST /chat` - Send chat message and get AI response
- `GET /health` - Health check endpoint

## Environment Variables

- `GEMINI_API_KEY` - Google Gemini AI API key (required)
