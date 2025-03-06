"""Main application module."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.api import api_router
from app.core.config import settings


# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you would specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router with version prefix
app.include_router(api_router, prefix=f"/api/{settings.API_VERSION}")

# Include API router without version prefix for compatibility with frontend
app.include_router(api_router, prefix="/api")


@app.get("/", tags=["status"])
async def root():
    """Root endpoint for health check."""
    return {"status": "ok", "message": "AI Safety Testing API is running"}


@app.get("/health", tags=["status"])
async def health():
    """Health check endpoint."""
    return {"status": "ok"} 