"""
Configuration and OpenSearch client setup.
"""

import os
from typing import Optional
from opensearchpy import OpenSearch
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""
    
    OPENSEARCH_HOST: str = os.getenv("OPENSEARCH_HOST", "localhost")
    OPENSEARCH_PORT: int = int(os.getenv("OPENSEARCH_PORT", "9200"))
    OPENSEARCH_INDEX: str = os.getenv("OPENSEARCH_INDEX", "bec")
    OPENSEARCH_USE_SSL: bool = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"
    OPENSEARCH_VERIFY_CERTS: bool = os.getenv("OPENSEARCH_VERIFY_CERTS", "true").lower() == "true"
    OPENSEARCH_USER: Optional[str] = os.getenv("OPENSEARCH_USER")
    OPENSEARCH_PASSWORD: Optional[str] = os.getenv("OPENSEARCH_PASSWORD")
    
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))


def get_opensearch_client() -> OpenSearch:
    """
    Create and return an OpenSearch client instance.
    
    Returns:
        OpenSearch: Configured OpenSearch client
    """
    config = Config()
    
    # Build connection parameters
    connection_params = {
        "hosts": [{"host": config.OPENSEARCH_HOST, "port": config.OPENSEARCH_PORT}],
        "use_ssl": config.OPENSEARCH_USE_SSL,
        "verify_certs": config.OPENSEARCH_VERIFY_CERTS,
        "ssl_show_warn": False,
    }
    
    # Add authentication if provided
    if config.OPENSEARCH_USER and config.OPENSEARCH_PASSWORD:
        connection_params["http_auth"] = (config.OPENSEARCH_USER, config.OPENSEARCH_PASSWORD)
    
    return OpenSearch(**connection_params)


# Global client instance (reused across requests)
opensearch_client = get_opensearch_client()
index_name = Config.OPENSEARCH_INDEX
