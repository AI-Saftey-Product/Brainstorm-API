import json
from unittest import mock

from fastapi.testclient import TestClient
from brainstorm.core.main import app

client = TestClient(app)


def test_model_creation_and_listing():
    # todo: add fixtures to reset db on each test run

    payload = {
        'model_id': 'new_model_1',
        'name': 'new model',
        'modality': 'NLP',
        'sub_type': "TEXT_GENERATION",
        'provider': 'OPENAI',  # this ends up being called provider not source
        'provider_model': 'gpt-4o-mini',
        'endpoint_url': 'no_url',
        'api_key': 'openapi_key_placeholder',
        'parameters': json.dumps({"temperature": 0.7})

    }
    response = client.post("api/models/create_or_update_model", json=payload)
    assert response.status_code == 200

    response = client.get("api/models/get_models", params={'model_id': 'new_model_1'})
    assert response.status_code == 200
    models = response.json()

    assert len(models) == 1
    assert models[0]['model_id'] == 'new_model_1'


@mock.patch("brainstorm.core.adapters.openai_adapter.AsyncOpenAI")
def test_model_validation_openai(mock_openai_client):
    mock_client = mock.AsyncMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "Hello"

    mock_openai_client.return_value = mock_client

    response = client.post("api/models/validate_model", params={"model_id": "new_model_1"})
    assert response.status_code == 200


def test_delete_models():
    # todo: add fixtures to reset db on each test run

    response = client.post("api/models/delete_models", json=['new_model_1'],)
    assert response.status_code == 200
