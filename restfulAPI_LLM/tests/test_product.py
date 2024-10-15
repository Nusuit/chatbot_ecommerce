import pytest
from app import create_app
from src.db import db
from src.models.product import Product
from src.services.ecommerce_service import get_product_info

@pytest.fixture(scope='module')
def test_client():
    flask_app = create_app()
    flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    flask_app.config['TESTING'] = True

    with flask_app.test_client() as testing_client:
        with flask_app.app_context():
            db.create_all()
            yield testing_client
            db.drop_all()

def test_get_product_info(test_client):
    product = Product(name="Test Product", price=10.99, category="Test", description="Test Description", url="http://example.com")
    db.session.add(product)
    db.session.commit()

    result = get_product_info(product.id)
    assert result['name'] == "Test Product"
    assert result['price'] == 10.99
