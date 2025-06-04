# streamlit_app.py
import streamlit as st
import requests
import json
import pandas as pd
import os

# Import các hàm RAG từ file utils_local.py
from utils_local import load_rag_components_for_local_demo, rag_pipeline_local, can_run_rag_local

st.set_page_config(page_title="Tiki Book Chatbot Demo (Local)", layout="wide")

st.title("📚 Tiki Book Chatbot (Local Demo)")
st.markdown("Chào mừng bạn đến với Chatbot Tiki Sách! Hãy hỏi mình về các loại sách, tác giả, giá cả, hoặc bất kỳ thông tin nào về sách mà bạn quan tâm. Mình sẽ cố gắng giới thiệu những cuốn sách phù hợp nhất từ cơ sở dữ liệu của Tiki.")
st.markdown("---")

# Load RAG components once when the Streamlit app starts
# Sử dụng st.cache_resource để chỉ tải một lần duy nhất
@st.cache_resource
def load_components():
    load_rag_components_for_local_demo()
    return can_run_rag_local

rag_is_ready = load_components()

if not rag_is_ready:
    st.error("❌ Hệ thống chatbot chưa sẵn sàng. Vui lòng kiểm tra lỗi tải mô hình hoặc dữ liệu ở console.")
    st.stop() # Dừng ứng dụng nếu không tải được components

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant_products" and isinstance(message["content"], list):
            # Tái tạo phần hiển thị sản phẩm tham khảo nếu muốn
            st.markdown("---")
            st.markdown("**Các sản phẩm được tham khảo để trả lời (thông tin nội bộ):**")
            for i, product in enumerate(message["content"]):
                st.markdown(f"**{i+1}. {product.get('product_name', 'N/A')}**")
                st.markdown(f"   * ID: {product.get('product_id', 'N/A')}")
                st.markdown(f"   * Tác giả/Thương hiệu: {product.get('author_brand_name', 'Không rõ')}")
                st.markdown(f"   * Giá: {product.get('price', 0):,} đồng")
                st.markdown(f"   * Số lượng đã bán: {product.get('quantity_sold', 0)}")
                st.markdown(f"   * [Link sản phẩm]({product.get('product_url_path', '#')})")
                st.markdown("")


# React to user input
if prompt := st.chat_input("Bạn muốn tìm sách gì?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call the RAG pipeline directly
    try:
        with st.spinner("Đang tìm kiếm và tạo câu trả lời..."):
            # Gọi trực tiếp hàm rag_pipeline_local
            # Tăng top_k_retrieval để LLM có nhiều ngữ cảnh hơn để chọn lọc thông minh
            chatbot_response, retrieved_products = rag_pipeline_local(prompt, top_k_retrieval=20) # Tăng lên 20

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(chatbot_response)
            # Không hiển thị phần "Các sản phẩm được tham khảo để trả lời:"
            # Nếu bạn muốn xem lại, hãy bỏ comment phần code trong vòng lặp messages ở trên
            # và thêm dòng này vào đây:
            # if retrieved_products:
            #     st.session_state.messages.append({"role": "assistant_products", "content": retrieved_products})


        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": chatbot_response})
        # LƯU Ý: Nếu bạn muốn lưu thông tin sản phẩm đã tham khảo vào lịch sử chat
        # để hiển thị lại khi refresh trang, bạn cần thêm dòng này:
        if retrieved_products:
            st.session_state.messages.append({"role": "assistant_products", "content": retrieved_products})


    except Exception as e:
        error_message = f"❌ Đã xảy ra lỗi không mong muốn: {e}"
        st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})