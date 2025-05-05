"""API router for the application."""
from fastapi import APIRouter, Depends

from brainstorm.api.v1.endpoints import models, datasets, evals, auth
from brainstorm.api.v1.endpoints.auth import get_current_user
from brainstorm.core.config import settings


api_router = APIRouter(dependencies=[Depends(get_current_user)])

# Include routers for different endpoints
api_router.include_router(models.router, prefix="/models", tags=["models"])
# api_router.include_router(tests.router, prefix="/tests", tags=["tests"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(evals.router, prefix="/evals", tags=["evals"])


auth_router = APIRouter()
auth_router.include_router(auth.router, tags=["auth"])
