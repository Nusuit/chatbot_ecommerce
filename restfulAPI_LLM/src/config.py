import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    MYSQL_USERNAME = os.environ.get('MYSQL_USERNAME')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD')
    MYSQL_HOST = os.environ.get('MYSQL_HOST')
    MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE')

    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{MYSQL_USERNAME}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY') or None
