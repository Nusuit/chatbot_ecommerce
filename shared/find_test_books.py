import pandas as pd
from utils import get_query_embedding, df_rag_input, faiss_index
import numpy as np

# Test queries
queries = [
    "Sách dạy lập trình Python cho người mới bắt đầu",
    "Sách về machine learning và AI",
    "Sách dạy nấu ăn Việt Nam",
    "Sách về quản lý tài chính cá nhân",
    "Sách về lịch sử Việt Nam"
]

print("Finding relevant books for each query...")
for query in queries:
    print(f"\nQuery: {query}")
    
    # Get query embedding
    query_vector = get_query_embedding(query).reshape(1, -1).astype('float32')
    
    # Search
    D, I = faiss_index.search(query_vector, 5)
    
    # Print results
    print("Top 5 results:")
    for i, idx in enumerate(I[0]):
        book = df_rag_input.iloc[idx]
        print(f"{idx}: {book['product_name']} (distance: {D[0][i]:.4f})")

print("\nUse these IDs to update the test_cases in utils.py") 