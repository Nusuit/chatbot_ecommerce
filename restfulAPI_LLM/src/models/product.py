from src.db import db

class Product(db.Model):
    __tablename__ = 'products'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    price = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    url = db.Column(db.String(255), nullable=False)

    sizes = db.relationship('Size', backref='product', lazy=True)
    colors = db.relationship('Color', backref='product', lazy=True)

class Size(db.Model):
    __tablename__ = 'sizes'

    id = db.Column(db.Integer, primary_key=True)
    size = db.Column(db.String(10), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)

class Color(db.Model):
    __tablename__ = 'colors'

    id = db.Column(db.Integer, primary_key=True)
    color = db.Column(db.String(50), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
