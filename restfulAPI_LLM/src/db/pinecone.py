import pinecone
import os
from dotenv import load_dotenv

# Tải các biến môi trường từ file .env
load_dotenv()

# Lấy Pinecone API key từ biến môi trường
api_key = os.getenv('PINECONE_API_KEY')

def get_pinecone_index(index_name):
    if api_key:
        pinecone.init(api_key=api_key, environment="us-east-1")
        return pinecone.Index(index_name)
    else:
        raise ValueError("Pinecone API key is not set")
