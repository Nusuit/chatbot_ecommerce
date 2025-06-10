import faiss
import os

# Configuration
DATA_PATH = './data'
FAISS_INDEX_FILE_PATH = os.path.join(DATA_PATH, 'tiki_books_faiss.index')

def main():
    print("Loading FAISS index...")
    index = faiss.read_index(FAISS_INDEX_FILE_PATH)
    
    print(f"\nIndex Information:")
    print(f"Type: {type(index).__name__}")
    print(f"Total vectors: {index.ntotal}")
    print(f"Dimension: {index.d}")

if __name__ == "__main__":
    main() 