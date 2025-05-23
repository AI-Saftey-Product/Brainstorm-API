"""API router for the application."""
from fastapi import APIRouter

from brainstorm.api.v1.endpoints import models, tests
from brainstorm.core.config import settings


api_router = APIRouter()

# Include routers for different endpoints
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(tests.router, prefix="/tests", tags=["tests"]) 