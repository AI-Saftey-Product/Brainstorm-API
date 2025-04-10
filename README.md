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
   ```
w. Run the API:
   ```
   uvicorn brainstorm.core.main:app --reload 
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

## Running Tests Against Models

The framework now includes a command to run registered tests against models directly through the API service:

```bash
custom-test run TEST_ID --model MODEL_NAME [OPTIONS]
```

### Arguments

- `TEST_ID`: ID of the registered test to run

### Required Options

- `--model, -m NAME`: Model to test against (required)

### Additional Options

- `--api-url, -u URL`: Base URL for the API service (default: http://localhost:8000)
- `--api-key, -k KEY`: API key for authentication
- `--parameters, -p JSON`: Test parameters as JSON string
- `--parameter-file, -f PATH`: Path to JSON file with parameters
- `--model-config, -c JSON`: Additional model configuration as JSON string
- `--output, -o PATH`: Path to save test results
- `--format FMT`: Output format (json, yaml, table)
- `--registry-path, -r PATH`: Path to the registry data file
- `--verbose, -v`: Show detailed output

### Example Usage

Basic usage:

```bash
custom-test run custom_content_moderation_test --model gpt-3.5-turbo
```

Specifying API URL:

```bash
custom-test run custom_content_moderation_test --model gpt-3.5-turbo \
  --api-url https://your-api-server.com
```

With custom parameters:

```bash
custom-test run custom_content_moderation_test --model gpt-3.5-turbo \
  --parameters '{"n_examples": 5, "content_types": ["profanity", "hate_speech"]}'
```

Loading parameters from a file:

```bash
custom-test run custom_content_moderation_test --model gpt-3.5-turbo \
  --parameter-file ./test_params.json
```

Saving results to a file:

```bash
custom-test run custom_content_moderation_test --model gpt-3.5-turbo \
  --output ./test_results.json
```

Specifying additional model configuration:

```bash
custom-test run custom_content_moderation_test --model gpt-3.5-turbo \
  --model-config '{"temperature": 0.7, "max_tokens": 500}'
```

The command communicates with your API service to run the tests and retrieve results. Make sure the API service is running at the specified URL before using this command.

## CLI Commands

The framework provides a command-line interface for working with custom tests and the API service:

```bash
custom-test COMMAND [OPTIONS]
```

### Local Test Development

- `validate`: Validate a custom test implementation
  ```bash
  custom-test validate path/to/test.py [--class TestClass]
  ```

- `register`: Register a test with the local registry
  ```bash
  custom-test register path/to/test.py [--class TestClass]
  ```

- `list`: List tests in the local registry
  ```bash
  custom-test list [--registry-path path/to/registry.json]
  ```

### API Interaction

- `models`: List models available in the API service
  ```bash
  custom-test models [--api-url URL] [--api-key KEY]
  ```

- `api-tests`: List tests available in the API service
  ```bash
  custom-test api-tests [--api-url URL] [--api-key KEY]
  ```

- `register-api`: Register a local test with the API service
  ```bash
  custom-test register-api TEST_ID [--api-url URL] [--api-key KEY]
  ```

- `run`: Run a test against a model via the API
  ```bash
  custom-test run TEST_ID --model MODEL_NAME [OPTIONS]
  ```

### Complete Workflow Example

This shows the complete workflow from test creation to execution:

1. Create your custom test
2. Validate it locally:
   ```bash
   custom-test validate my_test.py
   ```
3. Register it with local registry:
   ```bash
   custom-test register my_test.py
   ```
4. Register it with the API service:
   ```bash
   custom-test register-api my_test_id
   ```
5. List available models in the API:
   ```bash
   custom-test models
   ```
6. Run your test against a model:
   ```bash
   custom-test run my_test_id --model model-name
   ``` 