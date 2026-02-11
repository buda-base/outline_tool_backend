#!/bin/bash
# Development server script
# Can be run directly or inside a screen session for debugging

echo "Starting OpenSearch Volume API in development mode..."
echo "API will be available at: http://localhost:8000/api/v1"
echo "Swagger docs at: http://localhost:8000/api/v1/docs"
echo ""

# Run uvicorn with auto-reload for development
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
