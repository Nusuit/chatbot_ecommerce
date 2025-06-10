import streamlit as st
from utils import rag_pipeline, load_models_and_data
import time

# Set page config MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Tiki Book Chatbot",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'last_query_time' not in st.session_state:
    st.session_state.last_query_time = 0
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Initialize RAG system with caching
@st.cache_resource(show_spinner=False)
def initialize_rag():
    """Initialize RAG system with caching to prevent multiple loads"""
    load_models_and_data()

# Initialize RAG system
with st.spinner('Đang khởi tạo hệ thống...'):
    initialize_rag()

# Custom CSS
st.markdown("""
<style>
    .stTextInput > div > div > input {
        font-size: 20px;
    }
    .book-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .book-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .book-title {
        color: #6B46C1;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 12px;
        line-height: 1.4;
    }
    .book-info {
        color: #4A5568;
        font-size: 15px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .book-info:before {
        content: "•";
        color: #6B46C1;
    }
    .book-categories {
        color: #38A169;
        font-size: 14px;
        font-style: italic;
        margin: 8px 0;
    }
    .book-description {
        color: #2D3748;
        font-size: 15px;
        margin: 12px 0;
        line-height: 1.6;
        padding: 16px;
        background-color: #F7FAFC;
        border-radius: 8px;
        border-left: 4px solid #6B46C1;
    }
    .book-stats {
        display: flex;
        gap: 16px;
        margin: 12px 0;
        color: #4A5568;
        font-size: 14px;
    }
    .book-stat {
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .book-stat-icon {
        color: #6B46C1;
    }
    .book-link {
        color: #6B46C1;
        text-decoration: none;
        font-weight: 500;
        padding: 8px 16px;
        border: 1px solid #6B46C1;
        border-radius: 20px;
        display: inline-block;
        margin-top: 12px;
        transition: all 0.2s ease;
    }
    .book-link:hover {
        background-color: #6B46C1;
        color: white;
        text-decoration: none;
    }
    .chat-history-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .bot-message {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .error-message {
        color: #dc3545;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📚 Tiki Book Recommendation Chatbot")
st.markdown("""
Chào mừng bạn đến với Chatbot gợi ý sách! Hãy nhập câu hỏi của bạn về loại sách bạn muốn tìm,
ví dụ như:
- "Tôi muốn tìm sách về lập trình Python"
- "Có sách nào về nấu ăn Việt Nam không?"
- "Gợi ý cho tôi sách về quản lý tài chính"
""")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "books" in message:
            for book in message["books"]:
                # Create URL with proper format
                url = book['product_url_path']
                if not url.startswith('http'):
                    url = 'https://tiki.vn/' + url.lstrip('/')
                
                # Format rating stars
                rating = book.get('rating', 0)
                stars = "⭐" * int(rating) + "☆" * (5 - int(rating))
                
                # Create a card for each book
                st.markdown(f"""
                <div class="book-card">
                    <div class="book-title">{book['product_name']}</div>
                    <div class="book-info">Tác giả/Thương hiệu: {book['author_brand_name']}</div>
                    <div class="book-info">Giá: {book['price']:,.0f}đ</div>
                    <div class="book-categories">{book.get('categories', '')}</div>
                    <div class="book-description">{book['rag_description_snippet']}</div>
                    <div class="book-stats">
                        <div class="book-stat">
                            <span class="book-stat-icon">📊</span>
                            <span>Đã bán: {book.get('quantity_sold', 0)}</span>
                        </div>
                        <div class="book-stat">
                            <span class="book-stat-icon">⭐</span>
                            <span>{book.get('rating', 0)}/5 ({book.get('review_count', 0)} đánh giá)</span>
                        </div>
                    </div>
                    <a href="{url}" target="_blank" class="book-link">Xem trên Tiki</a>
                </div>
                """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Nhập câu hỏi của bạn:", key="chat_input"):
    # Rate limiting - 1 request per second
    current_time = time.time()
    if current_time - st.session_state.last_query_time < 1:
        st.warning("Vui lòng đợi một chút trước khi gửi câu hỏi tiếp theo.")
    else:
        st.session_state.last_query_time = current_time
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        try:
            # Show assistant response with loading spinner
            with st.chat_message("assistant"):
                with st.spinner('Đang tìm kiếm sách phù hợp...'):
                    response_text, retrieved_books = rag_pipeline(prompt)
                    st.markdown(response_text)
                    
                    # Only display book cards if there are books to show
                    if retrieved_books:
                        for i, book in enumerate(retrieved_books, 1):
                            url = book['product_url_path']
                            if not url.startswith('http'):
                                url = 'https://tiki.vn/' + url.lstrip('/')
                            
                            st.markdown(f"""
                            <div class="book-card">
                                <div style="display: flex; align-items: center; gap: 12px;">
                                    <div style="
                                        background-color: #6B46C1;
                                        color: white;
                                        width: 28px;
                                        height: 28px;
                                        border-radius: 50%;
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        font-weight: bold;
                                        font-size: 16px;">
                                        {i}
                                    </div>
                                    <div class="book-title">{book['product_name']}</div>
                                </div>
                                <div class="book-info">Tác giả/Thương hiệu: {book['author_brand_name']}</div>
                                <div class="book-info">Giá: {book['price']:,.0f}đ</div>
                                <div class="book-categories">{book.get('categories', '')}</div>
                                <div class="book-description">{book['rag_description_snippet']}</div>
                                <div class="book-stats">
                                    <div class="book-stat">
                                        <span class="book-stat-icon">📊</span>
                                        <span>Đã bán: {book.get('quantity_sold', 0)}</span>
                                    </div>
                                    <div class="book-stat">
                                        <span class="book-stat-icon">⭐</span>
                                        <span>{book.get('rating', 0)}/5 ({book.get('review_count', 0)} đánh giá)</span>
                                    </div>
                                </div>
                                <a href="{url}" target="_blank" class="book-link">Xem trên Tiki</a>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text,
                "books": retrieved_books
            })

            # Auto-scroll to the latest message
            st.markdown("""
                <script>
                    var messages = document.getElementsByClassName('stChatMessage');
                    if (messages.length > 0) {
                        var latest = messages[messages.length - 1];
                        latest.scrollIntoView();
                    }
                </script>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            error_msg = str(e)
            if "quota exceeded" in error_msg.lower():
                error_msg = "Xin lỗi, hệ thống đang tạm thời quá tải. Vui lòng thử lại sau ít phút."
            st.error(error_msg)
            # Remove the user message if there was an error
            st.session_state.chat_history.pop()

# Clear chat button
if st.button("Xóa lịch sử trò chuyện"):
    st.session_state.chat_history = []
    st.rerun() 