"""Health check route for Twinergy API.

Provides a fast, lightweight liveness and readiness probe that does not
require reading the dataset or initializing heavy model artifacts.
"""

from fastapi import APIRouter
from src.api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns the current operational status and version of the Twinergy API service.",
)
def get_health() -> HealthResponse:
    """Fast service health check without dataset loading."""
    return HealthResponse(
        status="healthy",
        service="Twinergy API",
        version="2.0.0",
    )
