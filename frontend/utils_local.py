# utils_local.py
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import time

# --- Configuration (cập nhật đường dẫn cho môi trường local) ---
# Đảm bảo thư mục 'data' nằm cùng cấp với root project directory
LOCAL_DATA_PATH = '../data'
PROCESSED_FILE_PATH = os.path.join(LOCAL_DATA_PATH, 'tiki_books_processed_for_rag.csv')
FAISS_INDEX_FILE_PATH = os.path.join(LOCAL_DATA_PATH, 'tiki_books_faiss.index')
EMBEDDING_MODEL_NAME = 'VoVanPhuc/sup-SimCSE-VietNamese-phobert-base'
GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'

# Global variables to store loaded models and data
df_rag_input_local = None # Đổi tên để tránh xung đột nếu có df_rag_input từ các file khác
faiss_index_local = None
embedding_model_local = None
gemini_llm_model_local = None
can_run_rag_local = False

def load_rag_components_for_local_demo():
    """
    Loads the processed DataFrame, FAISS index, and embedding model for local demo.
    This function should be called once when the Streamlit app starts up.
    """
    global df_rag_input_local, faiss_index_local, embedding_model_local, gemini_llm_model_local, can_run_rag_local

    print("--- Loading data and models for local Streamlit demo ---")

    # 1. Load DataFrame
    try:
        df_rag_input_local = pd.read_csv(PROCESSED_FILE_PATH)
        df_rag_input_local['rag_description'] = df_rag_input_local['rag_description'].astype(str)
        print(f"✅ Loaded processed data: {len(df_rag_input_local)} products.")
    except FileNotFoundError:
        print(f"❌ ERROR: Processed data file not found at {PROCESSED_FILE_PATH}. Please ensure it's in ./data folder.")
        can_run_rag_local = False
        return
    except Exception as e:
        print(f"❌ ERROR loading processed data: {e}")
        can_run_rag_local = False
        return

    # 2. Load FAISS index
    try:
        faiss_index_local = faiss.read_index(FAISS_INDEX_FILE_PATH)
        print(f"✅ Loaded FAISS index with {faiss_index_local.ntotal} vectors.")
        if faiss_index_local.ntotal != len(df_rag_input_local):
            print("⚠️ WARNING: FAISS index vector count does not match DataFrame row count.")
    except FileNotFoundError:
        print(f"❌ ERROR: FAISS index file not found at {FAISS_INDEX_FILE_PATH}. Please ensure it's in ./data folder.")
        can_run_rag_local = False
        return
    except Exception as e:
        print(f"❌ ERROR loading FAISS index: {e}")
        can_run_rag_local = False
        return

    # 3. Load Embedding Model
    try:
        embedding_model_local = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"✅ Loaded embedding model: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        print(f"❌ ERROR loading embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        can_run_rag_local = False
        return

    # 4. Configure Gemini API
    try:
        from dotenv import load_dotenv
        load_dotenv() # Load .env file
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in a .env file or directly.")
        genai.configure(api_key=gemini_api_key)
        gemini_llm_model_local = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"✅ Configured Gemini API and loaded model: {GEMINI_MODEL_NAME}")
    except Exception as e:
        print(f"❌ ERROR configuring Gemini API or loading model: {e}")
        can_run_rag_local = False
        return

    can_run_rag_local = True
    print("--- All RAG components loaded successfully for local demo! ---")

def get_query_embedding_local(query_text: str):
    """Generates an embedding for the given query text."""
    if embedding_model_local is None:
        print("Error: embedding_model_local not loaded.")
        return None
    try:
        return embedding_model_local.encode([query_text])[0].astype('float32')
    except Exception as e:
        print(f"Error generating embedding for query: {e}")
        return None

def search_faiss_index_local(query_embedding: np.ndarray, top_k: int = 5):
    """Searches the FAISS index for top_k similar vectors."""
    if faiss_index_local is None:
        print("Error: faiss_index_local not loaded.")
        return np.array([]), np.array([])
    if query_embedding is None:
        return np.array([]), np.array([])
    try:
        query_vector = np.array([query_embedding])
        distances, indices = faiss_index_local.search(query_vector, top_k)
        return distances[0], indices[0]
    except Exception as e:
        print(f"Error searching FAISS index: {e}")
        return np.array([]), np.array([])

def generate_gemini_response_local(prompt_text: str):
    """Generates a response using the Gemini LLM."""
    if gemini_llm_model_local is None:
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
            temperature=0.7 # Có thể thử nghiệm với temperature
        )
        response = gemini_llm_model_local.generate_content(
            prompt_text,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"Xin lỗi, đã có lỗi xảy ra khi kết nối với dịch vụ AI. Chi tiết: {str(e)}"

def generate_expanded_queries_local(user_query, num_queries=3):
    if gemini_llm_model_local is None:
        print("Lỗi: Mô hình Gemini chưa sẵn sàng để mở rộng truy vấn.")
        return [user_query] # Fallback to original query

    prompt = f"""Người dùng muốn tìm kiếm sách với câu hỏi sau: "{user_query}".
    Hãy phân tích câu hỏi này và tạo ra TỐI ĐA {num_queries} cụm từ/câu hỏi tìm kiếm có thể dùng để truy vấn cơ sở dữ liệu sách.
    Mỗi cụm từ/câu hỏi nên tập trung vào các khía cạnh khác nhau của ý định tìm kiếm của người dùng, hoặc các từ khóa liên quan.
    Chỉ trả lời bằng một danh sách các cụm từ/câu hỏi, mỗi cụm từ/câu hỏi trên một dòng mới.
    Ví dụ:
    Nếu người dùng hỏi: "Sách về kinh tế"
    Trả lời:
    sách kinh tế
    sách tài chính
    sách đầu tư
    ---
    Nếu người dùng hỏi: "Truyện thiếu nhi"
    Trả lời:
    truyện thiếu nhi
    sách cho trẻ em
    sách tuổi thơ
    ---
    Nếu người dùng hỏi: "{user_query}"
    Trả lời:
    """
    try:
        response_text = generate_gemini_response_local(prompt)
        expanded_queries = [q.strip() for q in response_text.split('\n') if q.strip()]
        if user_query not in expanded_queries:
            expanded_queries.insert(0, user_query)
        return expanded_queries[:num_queries]
    except Exception as e:
        print(f"Lỗi khi tạo truy vấn mở rộng: {e}")
        return [user_query]

def rag_pipeline_local(user_query: str, top_k_retrieval: int = 15):
    if not can_run_rag_local:
        return "Hệ thống RAG chưa sẵn sàng. Vui lòng kiểm tra log lỗi khi tải.", []

    # print(f"\nProcessing query: '{user_query}'") # Có thể comment cho Streamlit để output gọn hơn

    # Query Expansion
    expanded_queries = generate_expanded_queries_local(user_query, num_queries=3)
    # print(f"  Các truy vấn mở rộng: {expanded_queries}")

    all_retrieved_product_ids = set()
    all_retrieved_products_with_distance = []

    for q_text in expanded_queries:
        query_embedding = get_query_embedding_local(q_text)
        if query_embedding is None:
            # print(f"  Không thể tạo embedding cho truy vấn mở rộng: '{q_text}'. Bỏ qua.")
            continue
        distances, indices = search_faiss_index_local(query_embedding, top_k=top_k_retrieval)

        for i, doc_index in enumerate(indices):
            try:
                # Kiểm tra doc_index có hợp lệ không
                doc_index = int(doc_index)
                if 0 <= doc_index < len(df_rag_input_local):
                    product_id = df_rag_input_local.iloc[doc_index]['product_id']
                    if product_id not in all_retrieved_product_ids:
                        all_retrieved_product_ids.add(product_id)
                        all_retrieved_products_with_distance.append({
                            'product_id': product_id,
                            'distance': distances[i],
                            'original_index': doc_index
                        })
                # else: # Có thể comment để không in quá nhiều log nếu có index không hợp lệ
                    # print(f"Warning: Retrieved index {doc_index} out of bounds for df_rag_input_local.")
            except Exception as e:
                print(f"Error processing retrieved product from index {doc_index}: {e}")
                continue

    all_retrieved_products_with_distance.sort(key=lambda x: x['distance'])

    final_retrieved_products_info = []
    for item in all_retrieved_products_with_distance[:top_k_retrieval]:
        final_retrieved_products_info.append(df_rag_input_local.iloc[item['original_index']])


    if not final_retrieved_products_info:
        no_context_prompt = f"""Người dùng hỏi: "{user_query}".
        Là một trợ lý AI cho trang bán sách Tiki, hãy thông báo rằng bạn không tìm thấy thông tin sản phẩm cụ thể nào
        trong cơ sở dữ liệu phù hợp với yêu cầu này.
        Đề nghị người dùng thử một truy vấn khác hoặc cung cấp thêm chi tiết.
        Trả lời bằng tiếng Việt.
        Trợ lý:"""
        response_text = generate_gemini_response_local(no_context_prompt)
        return response_text, []

    # Format retrieved context for LLM
    context = "Dưới đây là thông tin về một số sản phẩm sách có thể liên quan từ cơ sở dữ liệu Tiki:\n\n"
    for i, product_info in enumerate(final_retrieved_products_info):
        name = product_info.get('product_name', 'N/A')
        author = product_info.get('author_brand_name', 'Không rõ')
        price = product_info.get('price', 0)
        quantity_sold = product_info.get('quantity_sold', 0)
        rating_average = product_info.get('rating_average', 0.0)
        review_count = product_info.get('review_count', 0)
        product_link = product_info.get('product_url_path', '#')
        description_for_llm = product_info.get('rag_description', 'Không có mô tả.')

        context += f"--- Sản phẩm {i+1} (ID: {product_info['product_id']}) ---\n"
        context += f"   Tên sách: {name}\n"
        context += f"   Tác giả/Thương hiệu: {author}\n"
        context += f"   Giá bán: {price:,} đồng\n"
        context += f"   Số lượt mua ước tính: {quantity_sold}\n"
        context += f"   Đánh giá trung bình: {rating_average:.1f} sao ({review_count} lượt)\n"
        context += f"   Link sản phẩm: {product_link}\n"
        context += f"   Mô tả chi tiết: {description_for_llm}\n\n"

    # Updated Prompt for intelligent selection
    final_prompt = f"""Bạn là một trợ lý AI chuyên nghiệp và thân thiện của trang web bán sách Tiki.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách chính xác và hữu ích, CHỈ DỰA TRÊN thông tin sản phẩm được cung cấp trong phần [NGỮ CẢNH SẢN PHẨM].
Tuyệt đối không được bịa đặt hoặc sử dụng kiến thức bên ngoài ngữ cảnh này.

HƯỚNG DẪN TRẢ LỜI:
1.  Đọc kỹ câu hỏi của người dùng: "{user_query}"
2.  Xem xét cẩn thận thông tin trong [NGỮ CẢNH SẢN PHẨM].
3.  Mục tiêu là giới thiệu TỐI ĐA 3 quyển sách phù hợp nhất với câu hỏi của người dùng từ ngữ cảnh được cung cấp.
4.  **KHI LỰA CHỌN SÁCH ĐỂ GIỚI THIỆU, hãy ưu tiên các tiêu chí sau (theo thứ tự ưu tiên):**
    * **Độ liên quan cao nhất** đến câu hỏi của người dùng.
    * **Độ phổ biến cao:** Ưu tiên các sách có 'Số lượt mua ước tính' cao và/hoặc 'Đánh giá trung bình' cao (trên 4.0 sao) với nhiều lượt nhận xét (trên 5 lượt).
    * **Tránh trùng lặp:** Tuyệt đối không giới thiệu các phiên bản khác nhau của CÙNG MỘT TỰA SÁCH (ví dụ: cùng tên sách nhưng khác nhà xuất bản hoặc bìa). Hãy chọn phiên bản tốt nhất (ví dụ: có nhiều lượt mua, đánh giá cao, hoặc giá tốt hơn) làm đại diện.
    * **Đa dạng thể loại/tác giả (nếu có thể):** Nếu có nhiều lựa chọn tốt, hãy cố gắng giới thiệu sách từ các tác giả hoặc thể loại hơi khác nhau để cung cấp nhiều lựa chọn hơn cho người dùng.
5.  Với MỖI quyển sách bạn chọn để giới thiệu, hãy trình bày các thông tin sau một cách rõ ràng và mạch lạc:
    * Tên sách đầy đủ.
    * Tác giả hoặc thương hiệu.
    * Một đoạn mô tả ngắn gọn và hấp dẫn về sách (dựa vào thông tin trong "Mô tả chi tiết" của sản phẩm đó).
    * Số lượt mua ước tính (nếu có thông tin và lớn hơn 0).
    * Giá bán hiện tại.
    * Link sản phẩm.
6.  Hãy trình bày thông tin của từng quyển sách một cách riêng biệt, có thể dùng gạch đầu dòng hoặc đánh số.
7.  Nếu sau khi xem xét kỹ ngữ cảnh, bạn chỉ tìm thấy 1 hoặc 2 quyển sách thực sự phù hợp, thì chỉ giới thiệu số lượng đó.
8.  Nếu người dùng hỏi một chi tiết cụ thể mà không có trong mô tả sản phẩm, hãy trả lời rằng thông tin đó không có sẵn.
9.  Nếu không có sản phẩm nào trong ngữ cảnh thực sự phù hợp với câu hỏi, hãy lịch sự thông báo và có thể gợi ý họ thử tìm kiếm với từ khóa khác.
10. Trả lời bằng tiếng Việt, giọng điệu tự nhiên, thân thiện và kết thúc một cách lịch sự.

[NGỮ CẢNH SẢN PHẨM]
{context}
[HẾT NGỮ CẢNH SẢN PHẨM]

Câu trả lời của trợ lý (bằng tiếng Việt, giới thiệu tối đa 3 sách nếu phù hợp, mỗi sách có đủ thông tin yêu cầu):
"""
    response_text = generate_gemini_response_local(final_prompt)

    retrieved_products_info_for_output = []
    if final_retrieved_products_info:
        for product_data_series in final_retrieved_products_info:
            try:
                product_dict = product_data_series[['product_id', 'product_name', 'author_brand_name', 'price', 'quantity_sold', 'product_url_path']].to_dict()
                product_dict['rag_description_snippet'] = product_data_series['rag_description'][:150] + '...' if len(product_data_series['rag_description']) > 150 else product_data_series['rag_description']
                retrieved_products_info_for_output.append(product_dict)
            except Exception as e_info:
                print(f"Lỗi khi lấy thông tin sản phẩm truy xuất cho output: {e_info}")

    return response_text, retrieved_products_info_for_output