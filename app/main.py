"""Main application module."""
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
from uuid import UUID, uuid4
import os

from app.api.api import api_router
from app.core.config import settings
from app.core.websocket import manager as websocket_manager
from app.api.endpoints import tests, models

# Configure logging
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create FastAPI app
app = FastAPI()

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

@app.websocket("/ws/tests")
async def websocket_endpoint_no_id(websocket: WebSocket):
    """WebSocket endpoint for test notifications without a test run ID."""
    try:
        # Generate a new test run ID
        test_run_id = str(uuid4())
        logger.info(f"Generated new test run ID: {test_run_id}")
        
        # Connect the WebSocket with the new test run ID (this includes accepting the connection)
        await websocket_manager.connect(websocket, test_run_id)
        
        # Send the test run ID to the client
        await websocket.send_json({
            "type": "connection_established",
            "test_run_id": test_run_id,
            "message": "WebSocket connection established with new test run ID"
        })
        logger.info(f"Sent test run ID to client: {test_run_id}")
        
        # Keep the connection alive until the client disconnects
        while True:
            try:
                data = await websocket.receive_text()
                # Handle any messages from the client if needed
                logger.debug(f"Received message from client: {data}")
            except WebSocketDisconnect:
                # Using non-async disconnect method
                websocket_manager.disconnect(websocket, test_run_id)
                logger.info(f"Client disconnected from test run: {test_run_id}")
                break
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        try:
            await websocket.close()
        except:
            pass

@app.websocket("/ws/tests/{test_run_id}")
async def websocket_endpoint(websocket: WebSocket, test_run_id: str):
    """WebSocket endpoint for test notifications with a specific test run ID."""
    try:
        logger.info(f"WebSocket connection request for test run ID: {test_run_id}")
        
        # Connect the WebSocket with the provided test run ID (this includes accepting the connection)
        await websocket_manager.connect(websocket, test_run_id)
        
        # Send confirmation to the client
        await websocket.send_json({
            "type": "connection_established",
            "test_run_id": test_run_id,
            "message": "WebSocket connection established with existing test run ID"
        })
        logger.info(f"WebSocket connection established for test run ID: {test_run_id}")
        
        # Keep the connection alive until the client disconnects
        while True:
            try:
                data = await websocket.receive_text()
                # Handle any messages from the client if needed
                logger.debug(f"Received message from client: {data}")
            except WebSocketDisconnect:
                # Using non-async disconnect method
                websocket_manager.disconnect(websocket, test_run_id)
                logger.info(f"Client disconnected from test run: {test_run_id}")
                break
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        try:
            await websocket.close()
        except:
            pass 