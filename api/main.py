"""
FastAPI application for OpenSearch volume data management.

Provides REST API endpoints for fetching and saving volume documents.
"""

from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from opensearchpy.exceptions import NotFoundError, OpenSearchException

from api.config import opensearch_client, index_name, Config


# Initialize FastAPI app with subpath mount
app = FastAPI(
    title="OpenSearch Volume API",
    description="Thin API layer for managing volume documents in OpenSearch",
    version="0.1.0",
    root_path="/api/v1",
)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify API and OpenSearch connectivity.
    
    Returns:
        dict: Status information
    """
    try:
        # Ping OpenSearch to verify connectivity
        if opensearch_client.ping():
            return {
                "status": "healthy",
                "opensearch": "connected",
                "index": index_name,
            }
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unhealthy",
                    "opensearch": "disconnected",
                },
            )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
            },
        )


@app.get("/volume/{doc_id}")
async def get_volume_data(doc_id: str) -> Dict[str, Any]:
    """
    Fetch a volume document by ID from OpenSearch.
    
    Args:
        doc_id: The document ID to retrieve
        
    Returns:
        dict: The document data
        
    Raises:
        HTTPException: 404 if document not found, 500 for other errors
    """
    try:
        response = opensearch_client.get(index=index_name, id=doc_id)
        return response["_source"]
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{doc_id}' not found in index '{index_name}'",
        )
    except OpenSearchException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OpenSearch error: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@app.put("/volume/{doc_id}")
async def put_volume_data(doc_id: str, document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save or update a volume document in OpenSearch.
    
    Args:
        doc_id: The document ID to save
        document: The complete document data to store
        
    Returns:
        dict: Operation result with document ID and status
        
    Raises:
        HTTPException: 400 for validation errors, 500 for other errors
    """
    try:
        # Validate that document is not empty
        if not document:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document body cannot be empty",
            )
        
        # Index the document (creates or updates)
        response = opensearch_client.index(
            index=index_name,
            id=doc_id,
            body=document,
            refresh=True,  # Make immediately searchable
        )
        
        return {
            "id": doc_id,
            "result": response["result"],  # "created" or "updated"
            "version": response["_version"],
        }
    except OpenSearchException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OpenSearch error: {str(e)}",
        )
    except HTTPException:
        # Re-raise our own HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint with API information.
    
    Returns:
        dict: API metadata
    """
    return {
        "name": "OpenSearch Volume API",
        "version": "0.1.0",
        "docs": "/api/v1/docs",
        "health": "/api/v1/health",
    }
