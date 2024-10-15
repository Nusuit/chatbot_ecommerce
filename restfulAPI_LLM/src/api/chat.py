from flask import request, jsonify
from . import api_blueprint
from ..services.llm_service import generate_response

@api_blueprint.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    response = generate_response(question)
    return jsonify({'response': response})
