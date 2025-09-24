import faiss
import numpy as np
import os
import time
from tqdm import tqdm

# Configuration
DATA_PATH = './data'
OLD_INDEX_PATH = os.path.join(DATA_PATH, 'tiki_books_faiss.index')

def extract_vectors(index_path):
    """Extract vectors from existing index."""
    print(f"\nExtracting vectors from {index_path}...")
    index = faiss.read_index(index_path)
    print(f"Index loaded: {type(index).__name__} with {index.ntotal} vectors of dimension {index.d}")
    
    # Extract vectors one by one using reconstruct
    vectors = np.zeros((index.ntotal, index.d), dtype='float32')
    print("Extracting vectors...")
    for i in tqdm(range(index.ntotal)):
        vectors[i] = index.reconstruct(i)
    
    print(f"Extracted vectors shape: {vectors.shape}")
    return vectors

def build_ivf_flat(vectors, nlist=None):
    """Build IVF-Flat index."""
    print("\nBuilding IVF-Flat index...")
    dimension = vectors.shape[1]
    num_vectors = vectors.shape[0]
    
    # Calculate number of clusters if not provided
    if nlist is None:
        nlist = min(4096, int(np.sqrt(num_vectors)))
    
    # Create index
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    # Train and add vectors
    print(f"Training index with {nlist} clusters...")
    t0 = time.time()
    index.train(vectors)
    print(f"Training time: {time.time() - t0:.2f}s")
    
    print("Adding vectors...")
    t0 = time.time()
    index.add(vectors)
    print(f"Adding time: {time.time() - t0:.2f}s")
    
    # Set search parameters
    index.nprobe = min(64, nlist)  # Number of clusters to search
    return index

def build_hnsw(vectors, M=32, efConstruction=40):
    """Build HNSW index."""
    print("\nBuilding HNSW index...")
    dimension = vectors.shape[1]
    
    # Create index
    index = faiss.IndexHNSWFlat(dimension, M)  # M: number of connections per layer
    index.hnsw.efConstruction = efConstruction  # Build time/quality trade-off
    index.hnsw.efSearch = 16  # Search time/quality trade-off
    
    # Add vectors
    print("Adding vectors...")
    t0 = time.time()
    index.add(vectors)
    print(f"Adding time: {time.time() - t0:.2f}s")
    return index

def test_index(index, vectors, queries=None):
    """Test index with sample queries."""
    if queries is None:
        # Use some random vectors as queries
        nq = 5
        queries = vectors[np.random.choice(len(vectors), nq)]
    
    print("\nTesting index...")
    t0 = time.time()
    D, I = index.search(queries, k=5)
    search_time = time.time() - t0
    
    print(f"Average search time: {(search_time / len(queries)) * 1000:.2f}ms per query")
    print(f"Average distances: {D.mean():.4f}")
    return D, I

def main():
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    # Extract vectors from old index
    vectors = extract_vectors(OLD_INDEX_PATH)
    
    # Build and test IVF-Flat index
    ivf_index = build_ivf_flat(vectors)
    test_index(ivf_index, vectors)
    output_path = os.path.join(DATA_PATH, 'tiki_books_faiss_ivf.index')
    faiss.write_index(ivf_index, output_path)
    print(f"Saved IVF-Flat index to: {output_path}")
    
    # Build and test HNSW index
    hnsw_index = build_hnsw(vectors)
    test_index(hnsw_index, vectors)
    output_path = os.path.join(DATA_PATH, 'tiki_books_faiss_hnsw.index')
    faiss.write_index(hnsw_index, output_path)
    print(f"Saved HNSW index to: {output_path}")
    
    print("\nDone! You can now use either index by updating the FAISS_INDEX_FILE_PATH in utils.py")
    print("Recommended steps:")
    print("1. Test both indices to see which gives better results")
    print("2. Update FAISS_INDEX_FILE_PATH in utils.py to use the better index")
    print("3. For IVF-Flat, you may want to adjust nprobe (currently set to 64)")
    print("4. For HNSW, you may want to adjust efSearch (currently set to 16)")

if __name__ == "__main__":
    main() 