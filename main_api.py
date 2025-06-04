from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from utils import rag_pipeline, can_run_rag, load_models_and_data
import os

# Initialize FastAPI app
app = FastAPI(
    title="Tiki Book Chatbot API",
    description="A RAG-powered chatbot for recommending books from Tiki dataset.",
    version="1.0.0",
)

# Pydantic model for request body
class ChatRequest(BaseModel):
    query: str
    top_k: int = 5 # Number of relevant documents to retrieve

# Pydantic model for response body (optional but good practice)
class ChatResponse(BaseModel):
    answer: str
    retrieved_products: list # List of dictionaries

@app.on_event("startup")
async def startup_event():
    """
    Load models and data when the FastAPI application starts up.
    """
    print("FastAPI startup event: Loading RAG components...")
    load_models_and_data()
    if not can_run_rag:
        print("Warning: RAG components failed to load. API might not function correctly.")

@app.get("/")
async def read_root():
    """
    Root endpoint for basic API health check.
    """
    return {"message": "Welcome to Tiki Book Chatbot API! Use /chat for interactions."}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint for chatbot interaction.
    Processes user query using the RAG pipeline and returns a generated response.
    """
    if not can_run_rag:
        raise HTTPException(status_code=503, detail="RAG system is not ready. Please check server logs.")

    user_query = request.query
    top_k_retrieval = request.top_k

    try:
        answer, retrieved_products_info = rag_pipeline(user_query, top_k_retrieval)
        return ChatResponse(answer=answer, retrieved_products=retrieved_products_info)
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

# To run the API locally:
if __name__ == "__main__":
# Ensure your data folder exists and contains the necessary files
    if not os.path.exists('./data'):
        os.makedirs('./data')
        print("Created ./data directory. Please place your processed CSV and FAISS index here.")
        uvicorn.run(app, host="0.0.0.0", port=8000)