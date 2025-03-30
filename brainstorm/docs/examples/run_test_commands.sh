#!/bin/bash
# Example commands for running custom tests against models through the API

# Configuration (replace with your actual values)
API_URL="http://localhost:8000"
API_KEY="your_api_key_here"  # If API requires authentication

# ----------------
# Basic Examples
# ----------------

# Run content moderation test with default parameters
custom-test run custom_content_moderation_test --model gpt-3.5-turbo --api-url $API_URL

# Run with custom parameter values specified inline
custom-test run custom_content_moderation_test --model gpt-3.5-turbo --api-url $API_URL \
  --parameters '{"n_examples": 3, "content_types": ["profanity", "hate_speech"]}'

# Run with parameters from a file
custom-test run custom_content_moderation_test --model gpt-3.5-turbo --api-url $API_URL \
  --parameter-file ./examples/test_parameters.json 

# Save results to a file
custom-test run custom_content_moderation_test --model gpt-3.5-turbo --api-url $API_URL \
  --output ./test_results.json

# ----------------
# Advanced Examples
# ----------------

# Run against different model type
custom-test run custom_content_moderation_test --model claude-3-opus-20240229 --api-url $API_URL \
  --parameter-file ./examples/test_parameters.json

# Run test with API key for authentication
custom-test run custom_content_moderation_test --model gpt-3.5-turbo \
  --api-url $API_URL --api-key $API_KEY

# Use custom registry path (for local test definitions)
custom-test run custom_content_moderation_test --model gpt-3.5-turbo --api-url $API_URL \
  --registry-path ./custom_registry.json

# Run with custom model configuration
custom-test run custom_content_moderation_test --model gpt-3.5-turbo --api-url $API_URL \
  --model-config '{"temperature": 0.1, "max_tokens": 500, "top_p": 0.9}'

# Run with verbose output
custom-test run custom_content_moderation_test --model gpt-3.5-turbo --api-url $API_URL -v

# ----------------
# Different Output Formats
# ----------------

# Save as YAML
custom-test run custom_content_moderation_test --model gpt-3.5-turbo --api-url $API_URL \
  --output ./test_results.yaml --format yaml

# ----------------
# Complete Workflow Example
# ----------------

# First validate a test (locally)
custom-test validate ./custom_test_framework/examples/content_moderation_test.py --class ContentModerationTest

# Register it with the registry (locally)
custom-test register ./custom_test_framework/examples/content_moderation_test.py --class ContentModerationTest

# List registered tests (locally)
custom-test list 

# Run the test through the API
custom-test run custom_content_moderation_test --model gpt-3.5-turbo --api-url $API_URL \
  --parameter-file ./examples/test_parameters.json \
  --output ./test_results.json \
  --verbose 