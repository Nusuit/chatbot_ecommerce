from src.models.product import Product

def get_product_info(product_id):
    product = Product.query.get(product_id)
    if not product:
        return None
    return {
        'id': product.id,
        'name': product.name,
        'price': product.price,
        'category': product.category,
        'description': product.description,
        'url': product.url,
        'sizes': [size.size for size in product.sizes],
        'colors': [color.color for color in product.colors]
    }
