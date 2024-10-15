import pytest
from app import create_app
from flask import json
from src.services.llm_service import generate_response

@pytest.fixture(scope='module')
def test_client():
    flask_app = create_app()
    flask_app.config['TESTING'] = True

    with flask_app.test_client() as testing_client:
        with flask_app.app_context():
            yield testing_client

def test_chat_api(test_client, mocker):
    # Giả lập (mock) hàm generate_response để kiểm tra API
    mock_response = "This is a test response."
    mocker.patch('app.services.llm_service.generate_response', return_value=mock_response)

    # Dữ liệu gửi đến API
    data = {
        'question': 'What is the capital of France?'
    }

    # Gửi yêu cầu POST đến endpoint /chat
    response = test_client.post('/api/chat', 
                                data=json.dumps(data), 
                                content_type='application/json')

    # Kiểm tra phản hồi
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['response'] == mock_response

def test_chat_api_no_question(test_client):
    # Trường hợp không có câu hỏi trong dữ liệu
    data = {}

    response = test_client.post('/api/chat',
                                data=json.dumps(data),
                                content_type='application/json')

    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data['error'] == 'No question provided'
