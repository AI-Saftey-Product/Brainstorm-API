#!/bin/bash
# Complete workflow example from test creation to execution

# Configuration
API_URL="http://localhost:8000"  # Replace with your API URL
API_KEY="your_api_key_here"      # If your API requires authentication
TEST_FILE="./custom_test_framework/examples/content_moderation_test.py"
TEST_CLASS="ContentModerationTest"

# Step 1: Validate the test locally
echo "Step 1: Validating test..."
custom-test validate $TEST_FILE --class $TEST_CLASS
if [ $? -ne 0 ]; then
    echo "❌ Validation failed. Please fix errors and try again."
    exit 1
fi
echo "✅ Test validation successful"
echo

# Step 2: Register the test with local registry
echo "Step 2: Registering test with local registry..."
custom-test register $TEST_FILE --class $TEST_CLASS
if [ $? -ne 0 ]; then
    echo "❌ Local registration failed."
    exit 1
fi
echo "✅ Test registered with local registry"
echo

# Step 3: List local registry to get test ID
echo "Step 3: Listing tests in local registry..."
custom-test list
echo

# Prompt for test ID
echo "Enter the test ID from above (e.g., custom_content_moderation_test):"
read TEST_ID
if [ -z "$TEST_ID" ]; then
    echo "❌ No test ID provided."
    exit 1
fi

# Step 4: Register test with API service
echo "Step 4: Registering test with API service..."
custom-test register-api $TEST_ID --api-url $API_URL --api-key $API_KEY
if [ $? -ne 0 ]; then
    echo "❌ API registration failed."
    exit 1
fi
echo "✅ Test registered with API service"
echo

# Step 5: List available models in API
echo "Step 5: Listing available models in API..."
custom-test models --api-url $API_URL --api-key $API_KEY
echo

# Prompt for model ID
echo "Enter a model ID from above to test against:"
read MODEL_ID
if [ -z "$MODEL_ID" ]; then
    echo "❌ No model ID provided."
    exit 1
fi

# Step 6: Prepare test parameters
echo "Step 6: Preparing test parameters..."
cat > test_params.json << EOL
{
  "n_examples": 3,
  "content_types": ["profanity", "hate_speech"],
  "severity_levels": ["low", "medium"],
  "model_parameters": {
    "temperature": 0.2,
    "max_tokens": 250
  }
}
EOL
echo "Created test_params.json"
echo

# Step 7: Run the test
echo "Step 7: Running test against model..."
custom-test run $TEST_ID --model $MODEL_ID \
  --api-url $API_URL --api-key $API_KEY \
  --parameter-file ./test_params.json \
  --output ./test_results.json \
  --verbose

if [ $? -ne 0 ]; then
    echo "❌ Test execution failed."
    exit 1
fi
echo "✅ Test execution completed"
echo

# Step 8: Examine results
echo "Step 8: Examining results..."
if [ -f "./test_results.json" ]; then
    echo "Results saved to ./test_results.json"
    echo "Summary of results:"
    cat ./test_results.json | grep -E '"status"|"score"|"issues_found"'
else
    echo "❌ Results file not found."
fi

echo
echo "Complete workflow demonstration finished" 