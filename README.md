# AI Safety Testing API

An API service for testing AI models across different modalities for safety concerns.

## Features

- Test AI models across multiple modalities (NLP, Vision, Audio, Multimodal, Tabular)
- Comprehensive test registry organized by safety domains
- Flexible model parameter handling
- Detailed test reporting and analysis

## Current Status

This API currently supports NLP model testing with plans to expand to other modalities.

## Setup

### Prerequisites

- Python 3.8+
- PostgreSQL

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure environment variables:
   ```
   cp .env.example .env
   # Edit .env with your configuration
   ```
4. Create the database:
   ```
   # In PostgreSQL
   CREATE DATABASE ai_safety_testing;
   ```
5. Run database migrations:
   ```
   alembic upgrade head
   ```
6. Run the API:
   ```
   uvicorn app.main:app --reload
   ```

## API Documentation

Once the server is running, access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

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

### Adding Support for New Modalities

1. Create a new model adapter in `app/model_adapters/`
2. Update the factory function in `app/model_adapters/__init__.py`
3. Add parameter schemas in `app/schemas/parameters/`
4. Register tests for the new modality in the test registry

## License

[Your License Here] 