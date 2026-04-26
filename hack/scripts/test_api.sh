#!/bin/bash

echo "Testing PUMA OpenAI-Compatible API"
echo "===================================="
echo

# Base URL
BASE_URL="http://localhost:8000"

echo "1. Health Check"
curl -s "$BASE_URL/health"
echo -e "\n"

echo "2. List Models"
curl -s "$BASE_URL/v1/models" | jq '.'
echo

echo "3. Chat Completion (Non-streaming)"
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-model",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 50
  }' | jq '.'
echo

echo "4. Chat Completion (Streaming)"
curl -s -N "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-model",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true,
    "max_tokens": 50
  }'
echo -e "\n"

echo "5. Legacy Text Completion"
curl -s "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-model",
    "prompt": "Once upon a time",
    "max_tokens": 50
  }' | jq '.'
echo

echo "Done!"
