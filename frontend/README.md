# Tiki Book Chatbot - Frontend

Streamlit web application for the Tiki Book Chatbot interface.

## Features

- Interactive chat interface
- Book recommendations display
- Real-time RAG pipeline integration
- Responsive web design

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
streamlit run app.py
```

or for the demo version:

```bash
streamlit run streamlit_app.py
```

## Deployment on Vercel

1. Install Vercel CLI: `npm i -g vercel`
2. Login: `vercel login`
3. Deploy: `vercel --prod`
4. Set environment variables in Vercel dashboard:
   - `GEMINI_API_KEY`

## Deployment on Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and select `frontend/app.py`
4. Add secrets in Advanced settings:
   ```
   GEMINI_API_KEY = "your_api_key_here"
   ```

## Files

- `app.py` - Main Streamlit application
- `streamlit_app.py` - Demo version
- `utils_local.py` - Local utilities and RAG pipeline
- `visualize_*.py` - Visualization tools

## Environment Variables

- `GEMINI_API_KEY` - Google Gemini AI API key (required)
