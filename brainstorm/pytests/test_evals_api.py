import json
from unittest import mock

from fastapi.testclient import TestClient
from brainstorm.core.main import app

client = TestClient(app)


def test_eval_creation_and_listing():
    # todo: add fixtures to reset db on each test run

    payload = {
        'eval_id': 'new_eval_1',
        'dataset_id': 'new_dataset_1',
        'model_id': 'llama_4_scout_vanilla',

        'scorer': 'ExactStringMatchScorer',
        'scorer_agg_functions': ['accuracy', 'negative_count'],
        'scorer_agg_dimensions': [],

        'name': 'test eval',
        'description': 'an eval for testing',
    }
    response = client.post("api/evals/create_or_update_eval", json=payload)
    assert response.status_code == 200

    response = client.get("api/evals/get_evals", params={'eval_id': 'new_eval_1'})
    assert response.status_code == 200
    evals = response.json()

    assert len(evals) == 1
    assert evals[0]['eval_id'] == 'new_eval_1'


def test_delete_evals():
    # todo: add fixtures to reset db on each test run

    response = client.post("api/evals/delete_evals", json=['new_eval_1'],)
    assert response.status_code == 200


def test_run_eval():
    # todo: add fixtures to reset db on each test run

    with client.stream("POST", "api/evals/run_eval", params={'eval_id': 'new_eval_1'}) as response:
        assert response.status_code == 200
        chunks = list(response.iter_bytes(chunk_size=None))

    decoded = b"".join(chunks).decode()
    breakpoint()
    dict_objects = [json.loads(i) for i in decoded.split("\n")[:-1]]
    scores = [i.get('score') for i in dict_objects]
    breakpoint()
