import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import time

# --- Configuration ---
DRIVE_PATH = './data' # In a real deployment, this might be a cloud storage path
PROCESSED_FILE_PATH = os.path.join(DRIVE_PATH, 'tiki_books_processed_for_rag.csv')
FAISS_INDEX_FILE_PATH = os.path.join(DRIVE_PATH, 'tiki_books_faiss.index')
EMBEDDING_MODEL_NAME = 'VoVanPhuc/sup-SimCSE-VietNamese-phobert-base'
GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'

# Global variables to store loaded models and data
df_rag_input = None
faiss_index = None
embedding_model = None
gemini_llm_model = None
can_run_rag = False

def load_models_and_data():
    """
    Loads the processed DataFrame, FAISS index, and embedding model.
    This function should be called once when the API starts up.
    """
    global df_rag_input, faiss_index, embedding_model, gemini_llm_model, can_run_rag

    print("--- Loading data and models for API ---")

    # 1. Load DataFrame
    try:
        df_rag_input = pd.read_csv(PROCESSED_FILE_PATH)
        df_rag_input['rag_description'] = df_rag_input['rag_description'].astype(str)
        print(f"✅ Loaded processed data: {len(df_rag_input)} products.")
    except FileNotFoundError:
        print(f"❌ ERROR: Processed data file not found at {PROCESSED_FILE_PATH}")
        can_run_rag = False
        return
    except Exception as e:
        print(f"❌ ERROR loading processed data: {e}")
        can_run_rag = False
        return

    # 2. Load FAISS index
    try:
        faiss_index = faiss.read_index(FAISS_INDEX_FILE_PATH)
        print(f"✅ Loaded FAISS index with {faiss_index.ntotal} vectors.")
        if faiss_index.ntotal != len(df_rag_input):
            print("⚠️ WARNING: FAISS index vector count does not match DataFrame row count.")
    except FileNotFoundError:
        print(f"❌ ERROR: FAISS index file not found at {FAISS_INDEX_FILE_PATH}")
        can_run_rag = False
        return
    except Exception as e:
        print(f"❌ ERROR loading FAISS index: {e}")
        can_run_rag = False
        return

    # 3. Load Embedding Model
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"✅ Loaded embedding model: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        print(f"❌ ERROR loading embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        can_run_rag = False
        return

    # 4. Configure Gemini API (Assuming API key is set as environment variable or via .env)
    try:
        # It's better to load API key from environment variables for deployment
        # For local testing, you can use python-dotenv
        from dotenv import load_dotenv
        load_dotenv()
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=gemini_api_key)
        gemini_llm_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"✅ Configured Gemini API and loaded model: {GEMINI_MODEL_NAME}")
    except Exception as e:
        print(f"❌ ERROR configuring Gemini API or loading model: {e}")
        can_run_rag = False
        return

    can_run_rag = True
    print("--- All RAG components loaded successfully! ---")

def get_query_embedding(query_text: str):
    """Generates an embedding for the given query text."""
    if embedding_model is None:
        print("Error: embedding_model not loaded.")
        return None
    try:
        return embedding_model.encode([query_text])[0].astype('float32')
    except Exception as e:
        print(f"Error generating embedding for query: {e}")
        return None

def search_faiss_index(query_embedding: np.ndarray, top_k: int = 5):
    """Searches the FAISS index for top_k similar vectors."""
    if faiss_index is None:
        print("Error: faiss_index not loaded.")
        return np.array([]), np.array([])
    if query_embedding is None:
        return np.array([]), np.array([])
    try:
        query_vector = np.array([query_embedding])
        distances, indices = faiss_index.search(query_vector, top_k)
        return distances[0], indices[0]
    except Exception as e:
        print(f"Error searching FAISS index: {e}")
        return np.array([]), np.array([])

def format_retrieved_context(retrieved_indices: np.ndarray, retrieved_distances: np.ndarray):
    """Formats the retrieved product information into a string for the LLM."""
    if df_rag_input is None or df_rag_input.empty:
        return "No product information available to create context."

    context = "Below is information about some potentially relevant books from the Tiki database:\n\n"
    if not retrieved_indices.size:
        return "Sorry, I couldn't find any relevant products in the database to create context."

    for i, doc_index in enumerate(retrieved_indices):
        try:
            doc_index = int(doc_index)
            if 0 <= doc_index < len(df_rag_input):
                product_info = df_rag_input.iloc[doc_index]
                name = product_info.get('product_name', 'N/A')
                author = product_info.get('author_brand_name', 'Không rõ')
                price = product_info.get('price', 0)
                quantity_sold = product_info.get('quantity_sold', 0)
                product_link = product_info.get('product_url_path', '#')
                description_for_llm = product_info.get('rag_description', 'No description.')
                distance = retrieved_distances[i]

                context += f"--- Product {i+1} (ID: {product_info['product_id']}, Retrieval Relevance (distance): {distance:.4f}) ---\n"
                context += f"   Book Name: {name}\n"
                context += f"   Author/Brand: {author}\n"
                context += f"   Current Price: {price:,} VND\n"
                context += f"   Estimated Quantity Sold: {quantity_sold}\n"
                context += f"   Product Link: {product_link}\n"
                context += f"   Detailed Description: {description_for_llm}\n\n"
            else:
                print(f"Warning: Invalid index {doc_index} for DataFrame of size {len(df_rag_input)}.")
        except Exception as e:
            print(f"Error formatting product at index {doc_index}: {e}")
            continue
    return context

def generate_gemini_response(prompt_text: str):
    """Generates a response using the Gemini LLM."""
    if gemini_llm_model is None:
        return "Sorry, the AI model is not ready to respond."
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            # temperature=0.7 # Can experiment with temperature
        )
        response = gemini_llm_model.generate_content(
            prompt_text,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"Sorry, an error occurred while connecting to the AI service. Details: {str(e)}"

def rag_pipeline(user_query: str, top_k_retrieval: int = 5):
    """Orchestrates the RAG process."""
    if not can_run_rag:
        return "The RAG system is not ready. Please check logs for loading errors.", []

    print(f"\nProcessing query: '{user_query}'")

    query_embedding = get_query_embedding(user_query)
    if query_embedding is None:
        return "Could not create embedding for your query. Please ensure the embedding model is loaded.", []

    distances, indices = search_faiss_index(query_embedding, top_k=top_k_retrieval)

    if not indices.size:
        no_context_prompt = f"""User asks: "{user_query}".
        As an AI assistant for Tiki book store, please inform that you couldn't find any specific product information
        in the database relevant to this request.
        Suggest the user to try a different query or provide more details.
        Answer in Vietnamese.
        Assistant:"""
        print("No products found in the database. Generating a general response...")
        response_text = generate_gemini_response(no_context_prompt)
        return response_text, []

    retrieved_context = format_retrieved_context(indices, distances)

    # Updated Prompt to introduce up to 3 books
    final_prompt = f"""You are a professional and friendly AI assistant for the Tiki book website.
Your task is to answer user questions accurately and helpfully, ONLY BASED ON the product information provided in the [PRODUCT CONTEXT] section.
Absolutely do not fabricate or use knowledge outside of this context.

ANSWERING GUIDELINES:
1. Read the user's question carefully: "{user_query}"
2. Carefully review the information in [PRODUCT CONTEXT].
3. The goal is to recommend a MAXIMUM of 3 most relevant books to the user's question from the provided context. Prioritize products with lower "Retrieval Relevance (distance)" (closer to 0 is better).
4. For EACH book you choose to recommend, present the following information clearly and coherently:
    * Full book title.
    * Author or brand.
    * A concise and appealing description of the book (based on the "Detailed Description" in that product's information).
    * Estimated quantity sold (if available and greater than 0).
    * Current selling price.
    * Product link.
5. Present the information for each book separately, using bullet points or numbering.
6. If, after careful consideration of the context, you only find 1 or 2 truly relevant books, only recommend that number.
7. If the user asks for a specific detail that is not in the product description, answer that the information is not available.
8. If no products in the context are truly relevant to the question, politely inform them and perhaps suggest they try searching with different keywords.
9. Answer in Vietnamese, with a natural, friendly tone, and end politely.

[PRODUCT CONTEXT]
{retrieved_context}
[END PRODUCT CONTEXT]

Assistant's answer (in Vietnamese, recommending up to 3 books if relevant, each with required info):
"""
    response_text = generate_gemini_response(final_prompt)

    retrieved_products_info_for_output = []
    if indices.size > 0 and df_rag_input is not None:
        for idx in indices:
            try:
                idx = int(idx)
                if 0 <= idx < len(df_rag_input):
                    # Only return essential info for display in Streamlit if needed
                    product_data = df_rag_input.iloc[idx][['product_id', 'product_name', 'author_brand_name', 'price', 'quantity_sold', 'product_url_path']].to_dict()
                    product_data['rag_description_snippet'] = df_rag_input.iloc[idx]['rag_description'][:150] + '...' if len(df_rag_input.iloc[idx]['rag_description']) > 150 else df_rag_input.iloc[idx]['rag_description']
                    retrieved_products_info_for_output.append(product_data)
                else:
                    print(f"Warning: Retrieved index {idx} out of bounds.")
            except Exception as e_info:
                print(f"Error retrieving product info for output at index {idx}: {e_info}")

    return response_text, retrieved_products_info_for_output

# Call this once when the module is imported (e.g., when main.py starts)
load_models_and_data()