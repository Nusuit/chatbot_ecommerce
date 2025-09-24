import faiss
import numpy as np
import pandas as pd
import time
import os
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
from utils import load_test_cases, get_query_embedding, df_rag_input, faiss_index

# Configuration
DATA_PATH = './data'
FLAT_INDEX_PATH = os.path.join(DATA_PATH, 'tiki_books_faiss.index')
IVF_INDEX_PATH = os.path.join(DATA_PATH, 'tiki_books_faiss_ivf.index')
HNSW_INDEX_PATH = os.path.join(DATA_PATH, 'tiki_books_faiss_hnsw.index')

def load_indices():
    """Load all three indices."""
    indices = {}
    print("Loading indices...")
    indices['flat'] = faiss.read_index(FLAT_INDEX_PATH)
    indices['ivf'] = faiss.read_index(IVF_INDEX_PATH)
    indices['hnsw'] = faiss.read_index(HNSW_INDEX_PATH)
    return indices

def evaluate_index(index, query_vectors, true_labels, k=5):
    """Evaluate an index using precision, recall, and search time."""
    t0 = time.time()
    D, I = index.search(query_vectors, k)
    search_time = (time.time() - t0) / len(query_vectors)
    
    # Convert predictions to binary format for precision/recall calculation
    y_true = []
    y_pred = []
    for i in range(len(query_vectors)):
        true_set = set(true_labels[i])
        pred_set = set(I[i])
        
        # For each possible book id
        for book_id in true_set.union(pred_set):
            y_true.append(1 if book_id in true_set else 0)
            y_pred.append(1 if book_id in pred_set else 0)
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'avg_search_time_ms': search_time * 1000,
        'avg_distance': D.mean()
    }

def main():
    # Load test cases
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases")
    
    # Get query embeddings
    query_vectors = []
    true_labels = []
    
    print("Processing test cases...")
    for case in tqdm(test_cases):
        query = case['query']
        expected_books = case['expected_books']
        
        # Get query embedding
        query_vectors.append(get_query_embedding(query))
        true_labels.append(expected_books)
    
    query_vectors = np.array(query_vectors).astype('float32')
    
    # Load indices
    indices = {
        'flat': faiss_index,  # Original flat index
        'ivf': faiss.read_index(IVF_INDEX_PATH),
        'hnsw': faiss.read_index(HNSW_INDEX_PATH)
    }
    
    # Evaluate each index
    results = {}
    for name, index in indices.items():
        print(f"\nEvaluating {name.upper()} index...")
        results[name] = evaluate_index(index, query_vectors, true_labels)
    
    # Print results
    print("\nBenchmark Results:")
    print("-" * 80)
    print(f"{'Index Type':<15} {'Precision':<12} {'Recall':<12} {'Search Time':<15} {'Avg Distance'}")
    print("-" * 80)
    
    for name, metrics in results.items():
        print(
            f"{name.upper():<15} "
            f"{metrics['precision']:.2f}%{'':>6} "
            f"{metrics['recall']:.2f}%{'':>6} "
            f"{metrics['avg_search_time_ms']:.2f}ms{'':>6} "
            f"{metrics['avg_distance']:.4f}"
        )

if __name__ == "__main__":
    main() 