"""WebSocket connection manager for real-time notifications."""
import logging
from typing import Dict, Set, Any
from fastapi import WebSocket, APIRouter
import time
from datetime import datetime
import json

logger = logging.getLogger(__name__)

def _serialize_datetime(obj: Any) -> Any:
    """Recursively serialize datetime objects in a dictionary or list."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _serialize_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    return obj

class ConnectionManager:
    """Manages WebSocket connections for real-time notifications."""
    
    def __init__(self):
        """Initialize the connection manager with empty connections."""
        # Map test_run_id to set of connected websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Map test_run_id to last connection time
        self.last_connection_time: Dict[str, float] = {}
        logger.info("WebSocket ConnectionManager initialized")
        
    async def connect(self, websocket: WebSocket, test_run_id: str):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        if test_run_id not in self.active_connections:
            self.active_connections[test_run_id] = set()
        self.active_connections[test_run_id].add(websocket)
        self.last_connection_time[test_run_id] = time.time()
        logger.info(f"New WebSocket connection for test run {test_run_id}. Total connections for this run: {len(self.active_connections[test_run_id])}")
        logger.info(f"Active test runs with connections: {list(self.active_connections.keys())}")
        logger.info(f"Connection times for active runs: {self.last_connection_time}")
        
    def disconnect(self, websocket: WebSocket, test_run_id: str):
        """Remove a WebSocket connection."""
        if test_run_id in self.active_connections:
            self.active_connections[test_run_id].discard(websocket)
            logger.info(f"WebSocket disconnected for test run {test_run_id}. Remaining connections for this run: {len(self.active_connections[test_run_id])}")
            if not self.active_connections[test_run_id]:
                del self.active_connections[test_run_id]
                if test_run_id in self.last_connection_time:
                    del self.last_connection_time[test_run_id]
                logger.info(f"Removed empty connection set for test run {test_run_id}")
            logger.info(f"Remaining test runs with connections: {list(self.active_connections.keys())}")
            logger.info(f"Connection times for remaining runs: {self.last_connection_time}")
                
    async def send_notification(self, test_run_id: str, message: dict):
        """Send a notification to all clients connected to a test run."""
        logger.info(f"Attempting to send notification of type '{message.get('type', 'unknown')}' to test run {test_run_id}")
        logger.info(f"Message content: {json.dumps(message, indent=2)}")
        logger.info(f"Current active connections: {list(self.active_connections.keys())}")
        logger.info(f"Current connection times: {self.last_connection_time}")
        
        if test_run_id in self.active_connections:
            logger.info(f"Found active connections for test run {test_run_id}. Total connections: {len(self.active_connections[test_run_id])}")
            
            # Serialize the message to handle datetime objects
            serialized_message = _serialize_datetime(message)
            logger.info(f"Serialized message: {json.dumps(serialized_message, indent=2)}")
            
            disconnected_websockets = set()
            for connection in self.active_connections[test_run_id]:
                try:
                    await connection.send_json(serialized_message)
                    logger.info(f"Successfully sent notification to a client for test run {test_run_id}")
                except Exception as e:
                    logger.error(f"Error sending WebSocket message: {e}")
                    logger.error(f"Failed message content: {json.dumps(serialized_message, indent=2)}")
                    disconnected_websockets.add(connection)
            
            # Clean up any disconnected websockets
            for websocket in disconnected_websockets:
                self.disconnect(websocket, test_run_id)
            
            if disconnected_websockets:
                logger.info(f"Removed {len(disconnected_websockets)} disconnected WebSockets")
        else:
            logger.warning(f"Attempted to send notification to test run {test_run_id}, but no active connections exist for this run")
            logger.warning(f"Current active connections: {list(self.active_connections.keys())}")
            logger.warning(f"Current connection times: {self.last_connection_time}")

# Create a singleton instance
manager = ConnectionManager()

# Create router for WebSocket endpoints
websocket_router = APIRouter()

@websocket_router.websocket("/ws/tests/{test_run_id}")
async def websocket_endpoint(websocket: WebSocket, test_run_id: str):
    """WebSocket endpoint for test run notifications."""
    try:
        await manager.connect(websocket, test_run_id)
        await websocket.send_json({
            "type": "connection_established",
            "test_run_id": test_run_id,
            "message": "WebSocket connection established successfully",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        try:
            while True:
                # Keep the connection alive and handle incoming messages
                data = await websocket.receive_json()
                await websocket.send_json({
                    "type": "message_received",
                    "test_run_id": test_run_id,
                    "data": data,
                    "timestamp": datetime.utcnow().isoformat()
                })
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}")
    finally:
        manager.disconnect(websocket, test_run_id)