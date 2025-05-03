import json
from unittest import mock

from fastapi.testclient import TestClient
from brainstorm.core.main import app

client = TestClient(app)


def test_dataset_creation_and_listing():
    # todo: add fixtures to reset db on each test run

    payload = {
        'dataset_id': 'new_dataset_1',
        'name': 'test dataset',
        'description': 'a dataset for testing',
        'modality': "NLP",
        'dataset_adapter': 'MuSR',  # this ends up being called provider not source
        'sample_size': 3,
        'parameters': json.dumps({"temperature": 0.7})

    }
    response = client.post("api/datasets/create_or_update_dataset", json=payload)
    assert response.status_code == 200

    response = client.get("api/datasets/get_datasets", params={'dataset_id': 'new_dataset_1'})
    assert response.status_code == 200
    datasets = response.json()

    assert len(datasets) == 1
    assert datasets[0]['dataset_id'] == 'new_dataset_1'


def test_delete_datasets():
    # todo: add fixtures to reset db on each test run

    response = client.post("api/datasets/delete_datasets", json=['new_dataset_1'],)
    assert response.status_code == 200
