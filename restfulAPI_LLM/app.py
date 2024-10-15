from flask import Flask
from src.db import init_db
from src.api import api_blueprint
from dotenv import load_dotenv
import os

load_dotenv()  # Tải các biến môi trường từ tệp .env


def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    
    init_db(app)
    app.register_blueprint(api_blueprint, url_prefix='/api')
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
