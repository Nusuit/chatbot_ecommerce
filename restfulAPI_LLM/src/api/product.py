from flask import request, jsonify
from . import api_blueprint
from ..services.ecommerce_service import get_product_info

@api_blueprint.route('/product/<int:product_id>', methods=['GET'])
def get_product(product_id):
    product_info = get_product_info(product_id)
    if not product_info:
        return jsonify({'error': 'Product not found'}), 404
    
    return jsonify(product_info)
