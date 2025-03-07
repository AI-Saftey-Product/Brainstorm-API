"""WebSocket connection manager for real-time notifications."""
import logging
from typing import Dict, Set
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time notifications."""
    
    def __init__(self):
        """Initialize the connection manager with empty connections."""
        # Map test_run_id to set of connected websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, test_run_id: str):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        if test_run_id not in self.active_connections:
            self.active_connections[test_run_id] = set()
        self.active_connections[test_run_id].add(websocket)
        logger.info(f"New WebSocket connection for test run {test_run_id}")
        
    def disconnect(self, websocket: WebSocket, test_run_id: str):
        """Remove a WebSocket connection."""
        if test_run_id in self.active_connections:
            self.active_connections[test_run_id].discard(websocket)
            if not self.active_connections[test_run_id]:
                del self.active_connections[test_run_id]
            logger.info(f"WebSocket disconnected for test run {test_run_id}")
                
    async def send_notification(self, test_run_id: str, message: dict):
        """Send a notification to all clients connected to a test run."""
        if test_run_id in self.active_connections:
            disconnected_websockets = set()
            for connection in self.active_connections[test_run_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending WebSocket message: {e}")
                    disconnected_websockets.add(connection)
            
            # Clean up any disconnected websockets
            for websocket in disconnected_websockets:
                self.disconnect(websocket, test_run_id)
            
            if disconnected_websockets:
                logger.info(f"Removed {len(disconnected_websockets)} disconnected WebSockets")

# Create a singleton instance
manager = ConnectionManager() 