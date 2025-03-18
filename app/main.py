"""Main application module."""
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Set
import json
from uuid import UUID
import os

from app.api.api import api_router
from app.core.config import settings
from app.core.websocket import manager as websocket_manager


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

# Get CORS settings from environment variables
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
allowed_methods = os.getenv("ALLOWED_METHODS", "GET,POST,PUT,DELETE,OPTIONS").split(",")
allowed_headers = os.getenv("ALLOWED_HEADERS", "Content-Type,Authorization,X-Requested-With").split(",")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=allowed_methods,
    allow_headers=allowed_headers,
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

@app.websocket("/ws/tests/{test_run_id}")
async def websocket_endpoint(websocket: WebSocket, test_run_id: str):
    """WebSocket endpoint for test result notifications."""
    await websocket_manager.connect(websocket, test_run_id)
    try:
        while True:
            # Keep the connection alive, waiting for backend notifications
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, test_run_id) 