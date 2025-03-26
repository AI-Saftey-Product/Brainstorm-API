"""WebSocket connection manager for real-time notifications."""
import logging
from typing import Dict, Set
from fastapi import WebSocket
import time

logger = logging.getLogger(__name__)

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
        logger.info(f"Current active connections: {list(self.active_connections.keys())}")
        logger.info(f"Current connection times: {self.last_connection_time}")
        
        if test_run_id in self.active_connections:
            logger.info(f"Found active connections for test run {test_run_id}. Total connections: {len(self.active_connections[test_run_id])}")
            
            disconnected_websockets = set()
            for connection in self.active_connections[test_run_id]:
                try:
                    await connection.send_json(message)
                    logger.debug(f"Successfully sent notification to a client for test run {test_run_id}")
                except Exception as e:
                    logger.error(f"Error sending WebSocket message: {e}")
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