import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm

# Configuration
DATA_PATH = './data'
PROCESSED_FILE_PATH = os.path.join(DATA_PATH, 'tiki_books_processed_for_rag.csv')
FAISS_INDEX_FILE_PATH = os.path.join(DATA_PATH, 'tiki_books_faiss.index')
EMBEDDING_MODEL_NAME = 'VoVanPhuc/sup-SimCSE-VietNamese-phobert-base'

def create_embeddings(df, model):
    """Creates embeddings for all products in the DataFrame."""
    print("Creating embeddings...")
    embeddings = []
    
    # Combine title and description for better semantic search
    texts = []
    for _, row in df.iterrows():
        text = f"{row['product_name']} {row['author_brand_name']} {row['rag_description']}"
        texts.append(text)
    
    # Create embeddings in batches
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings).astype('float32')

def build_faiss_index(embeddings, index_type='IVFFlat'):
    """Builds a FAISS index with the specified configuration."""
    print(f"Building FAISS index of type {index_type}...")
    
    dimension = embeddings.shape[1]
    num_vectors = embeddings.shape[0]
    
    if index_type == 'Flat':
        # Simple but exact nearest neighbor search
        index = faiss.IndexFlatL2(dimension)
    
    elif index_type == 'IVFFlat':
        # IVF with flat storage - good balance of speed and accuracy
        nlist = min(4096, int(np.sqrt(num_vectors)))  # Number of clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(embeddings)
    
    elif index_type == 'HNSW':
        # Hierarchical NSW - very fast and memory efficient
        index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections per layer
        index.hnsw.efConstruction = 40  # Higher value = better accuracy but slower build
        index.hnsw.efSearch = 16  # Higher value = better accuracy but slower search
    
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Add vectors to index
    index.add(embeddings)
    return index

def main():
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created directory: {DATA_PATH}")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(PROCESSED_FILE_PATH)
    df['rag_description'] = df['rag_description'].astype(str)
    print(f"Loaded {len(df)} products.")
    
    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Create embeddings
    embeddings = create_embeddings(df, model)
    print(f"Created embeddings with shape: {embeddings.shape}")
    
    # Build and save indices
    index_types = ['Flat', 'IVFFlat', 'HNSW']
    for index_type in index_types:
        print(f"\nBuilding {index_type} index...")
        index = build_faiss_index(embeddings, index_type)
        
        # Save index
        output_path = os.path.join(DATA_PATH, f'tiki_books_faiss_{index_type.lower()}.index')
        faiss.write_index(index, output_path)
        print(f"Saved {index_type} index to: {output_path}")
        
        # Test search
        print(f"Testing {index_type} index...")
        query = "sách dạy nấu ăn"
        query_embedding = model.encode([query]).astype('float32')
        distances, indices = index.search(query_embedding, k=5)
        
        print(f"\nTest search results for query: '{query}'")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            print(f"{i+1}. Distance: {dist:.4f}")
            print(f"   Title: {df.iloc[idx]['product_name']}")
            print(f"   Author: {df.iloc[idx]['author_brand_name']}")
            print()

if __name__ == "__main__":
    main() 