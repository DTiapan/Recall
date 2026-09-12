"""REST API and Web UI layer for Recall."""

from recall.api.app import create_app
from recall.api.service import RAGService

__all__ = [
    "create_app",
    "RAGService",
]
