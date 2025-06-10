# utils.py
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
import json

# Configuration
LOCAL_DATA_PATH = './data'
PROCESSED_FILE_PATH = os.path.join(LOCAL_DATA_PATH, 'tiki_books_processed_for_rag.csv')
FAISS_INDEX_FILE_PATH = os.path.join(LOCAL_DATA_PATH, 'tiki_books_faiss.index')
EMBEDDING_MODEL_NAME = 'VoVanPhuc/sup-SimCSE-VietNamese-phobert-base'
GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'

# Global variables to store loaded components
df_rag_input = None
embedding_model = None
FAISS_INDEX = None
gemini_model = None
can_run_rag = False
_is_loading = False

def load_models_and_data():
    """Load all necessary models and data."""
    global df_rag_input, embedding_model, FAISS_INDEX, gemini_model, can_run_rag, _is_loading
    
    # Prevent multiple loads
    if _is_loading:
        print("Another process is loading models and data...")
        return
    
    # If already loaded successfully, don't load again
    if can_run_rag and df_rag_input is not None and embedding_model is not None and FAISS_INDEX is not None and gemini_model is not None:
        print("Models and data already loaded.")
        return
    
    _is_loading = True
    
    try:
        print("--- Loading data and models for API ---")
        
        # Try multiple ways to get API key
        GEMINI_API_KEY = None
        
        # 1. Try from .env in current directory
        if os.path.exists('.env'):
            load_dotenv('.env')
            GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
            print("Found .env in current directory")
        
        # 2. Try from .env in script directory
        if not GEMINI_API_KEY:
            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
            if os.path.exists(env_path):
                load_dotenv(env_path)
                GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
                print(f"Found .env in script directory: {env_path}")
        
        # 3. Try from environment variable directly
        if not GEMINI_API_KEY:
            GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
            if GEMINI_API_KEY:
                print("Found GEMINI_API_KEY in environment variables")
        
        # 4. Try hardcoded key (temporary for testing)
        if not GEMINI_API_KEY:
            GEMINI_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
            print("Using hardcoded API key")
        
        # Load processed data
        df_rag_input = pd.read_csv(PROCESSED_FILE_PATH)
        df_rag_input['rag_description'] = df_rag_input['rag_description'].astype(str)
        print(f"✅ Loaded processed data: {len(df_rag_input)} products.")
        
        # Load FAISS index
        FAISS_INDEX = faiss.read_index(FAISS_INDEX_FILE_PATH)
        print(f"✅ Loaded FAISS index with {FAISS_INDEX.ntotal} vectors.")
        print(f"   Index type: {FAISS_INDEX.__class__.__name__}")
        
        # Load embedding model
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"✅ Loaded embedding model: {EMBEDDING_MODEL_NAME}")
        
        # Configure Gemini
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in any location. Please set it in .env file or environment variables")
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"✅ Configured Gemini API and loaded model: {GEMINI_MODEL_NAME}")
        
        # Set can_run_rag to True only if all components are loaded successfully
        can_run_rag = True
        print("--- All RAG components loaded successfully! ---")
        
    except Exception as e:
        print(f"Error loading RAG components: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
        can_run_rag = False
    finally:
        _is_loading = False

def get_query_embedding(query_text):
    """Get embedding vector for a query text."""
    return embedding_model.encode([query_text], convert_to_tensor=False)

def search_faiss_index(query_embedding: np.ndarray, top_k: int = 5):
    """Searches the FAISS index for top_k similar vectors."""
    if FAISS_INDEX is None:
        print("Error: FAISS_INDEX not loaded.")
        return np.array([]), np.array([])
    if query_embedding is None:
        return np.array([]), np.array([])
    try:
        # Print index information
        print(f"\nFAISS Index Information:")
        print(f"Type: {FAISS_INDEX.__class__.__name__}")
        print(f"Total vectors: {FAISS_INDEX.ntotal}")
        print(f"Dimension: {FAISS_INDEX.d}")
        
        # Ensure query_embedding is 2D array with shape (1, dimension)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        elif len(query_embedding.shape) == 2 and query_embedding.shape[0] > 1:
            query_embedding = query_embedding[0].reshape(1, -1)
            
        # Ensure we're using float32
        query_embedding = query_embedding.astype(np.float32)
        
        distances, indices = FAISS_INDEX.search(query_embedding, top_k)
        return distances[0], indices[0]
    except Exception as e:
        print(f"Error searching FAISS index: {e}")
        print(f"Query embedding shape: {query_embedding.shape}")
        print(f"Query embedding dtype: {query_embedding.dtype}")
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
                product_info = df_rag_input.iloc[doc_index].to_dict()
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

def generate_gemini_response(prompt_text: str, max_retries=3, retry_delay=1):
    """Generates a response using the Gemini LLM with retry logic."""
    if gemini_model is None:
        return "Sorry, the AI model is not ready to respond."
    
    for attempt in range(max_retries):
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
            response = gemini_model.generate_content(
                prompt_text,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            return response.text
        except Exception as e:
            error_msg = str(e).lower()
            if "quota exceeded" in error_msg:
                # If quota exceeded, don't retry
                raise Exception("Hệ thống đang tạm thời quá tải. Vui lòng thử lại sau ít phút.")
            elif attempt < max_retries - 1:
                print(f"Error calling Gemini API (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
                continue
            else:
                raise Exception(f"Không thể kết nối với dịch vụ AI sau {max_retries} lần thử. Chi tiết lỗi: {str(e)}")

def generate_expanded_queries(user_query, num_queries=3):
    """Generates expanded queries using Gemini."""
    if gemini_model is None:
        print("Error: Gemini model not ready for query expansion.")
        return [user_query]  # Fallback to original query

    prompt = f"""Bạn là một chuyên gia về tìm kiếm sách. Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và tạo ra các truy vấn tìm kiếm hiệu quả.

Câu hỏi của người dùng: "{user_query}"

Yêu cầu:
1. Tạo ra TỐI ĐA {num_queries} truy vấn tìm kiếm khác nhau.
2. Mỗi truy vấn nên:
   - Ngắn gọn (2-4 từ)
   - Chỉ giữ lại các từ khóa quan trọng nhất
   - Tập trung vào chủ đề chính và phụ
   - Loại bỏ hoàn toàn các từ không cần thiết
3. Các truy vấn nên:
   - Truy vấn 1: Giữ nguyên các từ khóa chính
   - Truy vấn 2: Tập trung vào chủ đề chính
   - Truy vấn 3: Tập trung vào chủ đề phụ hoặc mục đích sử dụng

Chỉ trả lời bằng danh sách các truy vấn, mỗi truy vấn trên một dòng.
Không thêm bất kỳ giải thích hay định dạng nào khác.

Ví dụ nếu người dùng hỏi "Tôi muốn tìm sách dạy nấu ăn cho người mới bắt đầu":
sách dạy nấu ăn
sách nấu ăn
nấu ăn cơ bản
---
Trả lời cho câu hỏi: "{user_query}"
"""
    try:
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            temperature=0.2  # Giảm temperature để có kết quả ổn định hơn
        )
        response_text = generate_gemini_response(prompt)
        expanded_queries = [q.strip() for q in response_text.split('\n') if q.strip()]
        if user_query not in expanded_queries:
            expanded_queries.insert(0, user_query)
        return expanded_queries[:num_queries]
    except Exception as e:
        print(f"Error generating expanded queries: {e}")
        return [user_query]

def rerank_results(retrieved_products, query_embedding):
    """Reranks retrieved products based on multiple factors."""
    for product in retrieved_products:
        # 1. Relevance score (0-1, higher is better)
        relevance_score = 1 - min(product['distance'], 1)
        
        # 2. Sales score (0-1, higher is better)
        sales = product.get('quantity_sold', 0)
        max_sales = max(p.get('quantity_sold', 0) for p in retrieved_products)
        sales_score = sales / max_sales if max_sales > 0 else 0
        
        # 3. Rating score (0-1, higher is better)
        rating = product.get('rating', 0)
        review_count = product.get('review_count', 0)
        # Weight rating by number of reviews (more reviews = more reliable rating)
        max_reviews = max(p.get('review_count', 0) for p in retrieved_products)
        review_weight = review_count / max_reviews if max_reviews > 0 else 0
        rating_score = (rating / 5.0) * (0.3 + 0.7 * review_weight)  # Base 0.3 weight even with no reviews
        
        # Combined score with weights
        # - Relevance: 40% (still important but not dominant)
        # - Sales: 30% (strong indicator of popularity)
        # - Rating: 30% (quality indicator weighted by review count)
        product['final_score'] = (
            0.40 * relevance_score +
            0.30 * sales_score +
            0.30 * rating_score
        )
    
    # Sort by final score
    return sorted(retrieved_products, key=lambda x: x['final_score'], reverse=True)

def classify_question_type(user_query: str) -> str:
    """Classifies the type of question being asked.
    Returns: 'search' for book search queries, 'content' for book content queries
    """
    if gemini_model is None:
        print("Error: Gemini model not ready for query classification")
        return 'search'  # Default to search if model not available

    prompt = f"""Phân loại câu hỏi sau của người dùng về sách: "{user_query}"

Hãy phân loại câu hỏi thành một trong hai loại:
1. 'search': Câu hỏi tìm kiếm/gợi ý sách (VD: "Có sách nào về X không?", "Tìm sách về Y", "Gợi ý sách về Z")
2. 'content': Câu hỏi về nội dung cụ thể của một cuốn sách (VD: "Sách X nói về gì?", "Nội dung chính của sách Y là gì?", "Sách Z có ý nghĩa gì?")

Chỉ trả lời bằng một từ duy nhất: 'search' hoặc 'content'.
"""
    try:
        response = generate_gemini_response(prompt)
        question_type = response.strip().lower()
        if question_type not in ['search', 'content']:
            return 'search'  # Default to search for invalid responses
        return question_type
    except Exception as e:
        print(f"Error classifying question: {e}")
        return 'search'  # Default to search on error

def rag_pipeline(user_query: str, top_k_retrieval: int = 15):
    """Orchestrates the RAG process."""
    if not can_run_rag:
        return "The RAG system is not ready. Please check logs for loading errors.", []

    print(f"\nProcessing query: '{user_query}'")

    # Classify question type
    question_type = classify_question_type(user_query)
    print(f"Question type: {question_type}")

    # Query Expansion
    expanded_queries = generate_expanded_queries(user_query)
    print(f"Expanded queries: {expanded_queries}")

    all_retrieved_products = []
    seen_product_ids = set()

    # Retrieve products for each expanded query
    for query in expanded_queries:
        query_embedding = get_query_embedding(query)
        if query_embedding is None:
            continue

        distances, indices = search_faiss_index(query_embedding, top_k=top_k_retrieval)

        for i, idx in enumerate(indices):
            if isinstance(idx, (int, np.integer)) and 0 <= idx < len(df_rag_input):
                product_id = df_rag_input.iloc[idx]['product_id']
                if product_id not in seen_product_ids:
                    seen_product_ids.add(product_id)
                    product_info = df_rag_input.iloc[idx].to_dict()
                    product_info['distance'] = float(distances[i])
                    all_retrieved_products.append(product_info)

    if not all_retrieved_products:
        no_context_prompt = f"""User asks: "{user_query}".
        As an AI assistant for Tiki book store, please inform that you couldn't find any specific product information
        in the database relevant to this request.
        Suggest the user to try a different query or provide more details.
        Answer in Vietnamese.
        Assistant:"""
        print("No products found in the database. Generating a general response...")
        response_text = generate_gemini_response(no_context_prompt)
        return response_text, []

    # Rerank results
    query_embedding = get_query_embedding(user_query)
    reranked_products = rerank_results(all_retrieved_products, query_embedding)[:top_k_retrieval]

    # Format context from reranked products
    context_parts = []
    for i, product in enumerate(reranked_products, 1):
        context_parts.append(f"""
Product {i}:
- Product ID: {product['product_id']}
- Name: {product['product_name']}
- Author/Brand: {product['author_brand_name']}
- Price: {product['price']:,.0f} VND
- Quantity Sold: {product.get('quantity_sold', 'N/A')}
- Categories: {product.get('joined_category_name', 'N/A')}
- Retrieval Relevance (distance): {product['distance']:.4f}
- Sales Score: {product.get('final_score', 0):.4f}
- Product URL: {product['product_url_path']}
- Detailed Description: {product['rag_description']}
""")

    retrieved_context = "\n".join(context_parts)

    # Different prompts for different question types
    if question_type == 'content':
        final_prompt = f"""You are a knowledgeable AI assistant for the Tiki book website.
Your task is to answer the user's specific question about book content: "{user_query}"

[PRODUCT CONTEXT]
{retrieved_context}
[END PRODUCT CONTEXT]

Please provide a detailed response in Vietnamese that:
1. Directly answers the user's question about the book's content
2. Uses both the provided book description AND your general knowledge about Buddhism, philosophy, and related topics
3. If the exact answer isn't in the description, use your knowledge to provide relevant insights while noting that you're supplementing beyond the book's description
4. Keep the focus on helping the user understand the book's content and meaning
5. Make the response conversational and engaging
6. Don't mention prices or sales statistics

Assistant's response (in Vietnamese):"""
    else:  # search type
        final_prompt = f"""You are a professional and friendly AI assistant for the Tiki book website.
Your task is to provide a brief, friendly introduction for book recommendations based on the user's question: "{user_query}"

[PRODUCT CONTEXT]
{retrieved_context}
[END PRODUCT CONTEXT]

Please provide a VERY BRIEF response in Vietnamese that:
1. Greets the user warmly
2. Acknowledges their interest in the topic
3. Mentions that you have some book recommendations
4. Ends with a friendly closing line

Keep it under 3 sentences. DO NOT list or describe the books - that will be handled by the UI.
DO NOT mention prices, authors, or any specific book details.
Just provide a friendly introduction.

Example:
"Chào bạn! Mình thấy bạn đang quan tâm đến sách về [topic]. Mình có một số gợi ý rất hay về chủ đề này. Hy vọng bạn sẽ tìm được cuốn sách phù hợp nhất!"

Assistant's response (in Vietnamese):"""

    response_text = generate_gemini_response(final_prompt)

    # Only prepare book output for search queries
    if question_type == 'search':
        # Limit to top 5 products for search queries
        retrieved_products_info_for_output = []
        for product in reranked_products[:5]:  # Limit to top 5 products
            # Create snippet from full description
            description = product['rag_description']
            snippet = description[:500] + "..." if len(description) > 500 else description
            
            product_info = {
                'product_id': product['product_id'],
                'product_name': product['product_name'],
                'author_brand_name': product['author_brand_name'],
                'price': product['price'],
                'quantity_sold': product.get('quantity_sold', 0),
                'rating': product.get('rating', 0),
                'review_count': product.get('review_count', 0),
                'categories': product.get('joined_category_name', ''),
                'product_url_path': product['product_url_path'],
                'rag_description_snippet': snippet
            }
            retrieved_products_info_for_output.append(product_info)
    else:
        # Return empty list for content queries
        retrieved_products_info_for_output = []

    return response_text, retrieved_products_info_for_output

def load_test_cases():
    """Load test cases for benchmarking."""
    test_cases = {
        'Python programming': [33112, 29268, 32476],
        'AI/ML': [161602, 33435, 33444],
        'Vietnamese cooking': [52009, 51802, 163451],
        'Personal finance': [172145, 45684, 45109],
        'Vietnamese history': [36138, 155597, 155320]
    }
    return test_cases

def get_book_info(idx):
    """Get book information from dataframe by index."""
    book = df_rag_input.iloc[idx]
    description = book['rag_description'] if not pd.isna(book['rag_description']) else ''
    # Giới hạn mô tả trong khoảng 100 ký tự và thêm dấu ... nếu bị cắt
    short_desc = description[:100] + '...' if len(description) > 100 else description
    
    return {
        'title': book['product_name'],
        'price': float(book['price']) if not pd.isna(book['price']) else 0.0,
        'rating_average': float(book['rating_average']) if not pd.isna(book['rating_average']) else 0.0,
        'categories': book['joined_category_name'] if not pd.isna(book['joined_category_name']) else 'Chưa phân loại',
        'url': book['product_url_path'] if not pd.isna(book['product_url_path']) else '',
        'description': short_desc
    }