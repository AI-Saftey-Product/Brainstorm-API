
### Installation

Install dependencies:
   ```
   pip install -r requirements.txt
   ```
Run the API:
   ```
   uvicorn brainstorm.core.main:app --reload 
   ```

## API Endpoints

### Models

- `POST /api/v1/models` - Register a new model
- `GET /api/v1/models` - List all registered models
- `GET /api/v1/models/{model_id}` - Get model details
- `PUT /api/v1/models/{model_id}` - Update model configuration
- `DELETE /api/v1/models/{model_id}` - Remove a model
- `POST /api/v1/models/{model_id}/validate` - Validate model connection

### Tests

- `GET /api/v1/tests/registry` - Get available tests
- `POST /api/v1/tests/runs` - Create a new test run
- `GET /api/v1/tests/runs` - List all test runs
- `GET /api/v1/tests/runs/{run_id}` - Get test run details
- `GET /api/v1/tests/runs/{run_id}/results` - Get test results

## Development

### Adding New Tests

To add new tests, update the test registry in `app/test_registry/registry.py`.
