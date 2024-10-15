from flask import Flask
from .config import Config
from .api import api_blueprint

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Đăng ký các blueprint cho API
    app.register_blueprint(api_blueprint, url_prefix='/api')
    
    return app
