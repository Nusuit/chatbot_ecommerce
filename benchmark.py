import pandas as pd
import numpy as np
from utils import rag_pipeline, load_models_and_data
import time

# Test queries
test_queries = [
    "Sách dạy lập trình Python cho người mới bắt đầu",
    "Sách về machine learning và AI",
    "Sách dạy nấu ăn Việt Nam",
    "Sách về quản lý tài chính cá nhân",
    "Sách về lịch sử Việt Nam",
    "Cuốn sách Đắc Nhân Tâm nói về điều gì?",
    "Nội dung chính của sách Nhà Giả Kim là gì?",
    "Sách về phát triển bản thân",
    "Truyện ngắn Nguyễn Nhật Ánh",
    "Sách về y học cổ truyền"
]

def run_benchmark():
    print("Đang khởi tạo các thành phần RAG...")
    try:
        # Initialize RAG components
        load_models_and_data()
        
        # Prepare results storage
        results = []
        
        # Run tests
        for query in test_queries:
            print(f"\nTesting query: {query}")
            
            # Measure retrieval time
            start_time = time.time()
            response_text, retrieved_books = rag_pipeline(query)
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            retrieval_time = total_time  # For now, we'll use total time as retrieval time
            
            # Calculate a simple relevance score (placeholder)
            # In a real scenario, this would be more sophisticated
            relevance_score = len(retrieved_books) / 5.0  # Normalize by expected number of results
            
            # Store results
            results.append({
                'query': query,
                'response_text': response_text,
                'num_results': len(retrieved_books),
                'response_time': total_time,
                'retrieval_time': retrieval_time,
                'relevance_score': relevance_score
            })
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Save results
        df_results.to_csv('benchmark_results.csv', index=False)
        print("\n✅ Benchmark completed successfully!")
        print(f"Tested {len(test_queries)} queries")
        print(f"Average response time: {df_results['response_time'].mean():.2f} seconds")
        print(f"Average relevance score: {df_results['relevance_score'].mean():.2f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during benchmark: {str(e)}")
        return False

if __name__ == "__main__":
    run_benchmark() 