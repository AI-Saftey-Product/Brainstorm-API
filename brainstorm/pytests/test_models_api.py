from fastapi.testclient import TestClient
from brainstorm.core.main import app

client = TestClient(app)


openai_api_key = ''

def test_create_or_update_model():
    payload = {
        'model_id': 'new_model_1',
        'name': 'new model',
        'modality': 'NLP',
        'sub_type': "Text Generation",
        'provider': 'OpenAI',  # this ends up being called provider not source
        'provider_model': 'GPT-3.5-Turbo',
        'endpoint_url': 'no_url',
        'api_key': 'openapi_key_placeholder',
    }
    response = client.post("api/models/create_or_update_model", json=payload)
    breakpoint()
    assert response.status_code == 200
    # this actually works and sends out test request, verified via usage in openai's account

