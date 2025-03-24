"""WebSocket manager for handling real-time communication."""
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class WebsocketManager:
    """Manages WebSocket connections and message broadcasting."""
    
    _active_connections: List[WebSocket] = []
    _connection_ids: Dict[WebSocket, str] = {}
    
    @classmethod
    async def connect(cls, websocket: WebSocket, connection_id: Optional[str] = None) -> None:
        """Accept a new websocket connection."""
        await websocket.accept()
        cls._active_connections.append(websocket)
        if connection_id:
            cls._connection_ids[websocket] = connection_id
        logger.info(f"New websocket connection established. Total connections: {len(cls._active_connections)}")
    
    @classmethod
    async def disconnect(cls, websocket: WebSocket) -> None:
        """Remove a websocket connection."""
        try:
            cls._active_connections.remove(websocket)
            if websocket in cls._connection_ids:
                del cls._connection_ids[websocket]
            logger.info(f"Websocket connection closed. Remaining connections: {len(cls._active_connections)}")
        except ValueError:
            logger.warning("Attempted to disconnect non-existent websocket connection")
    
    @classmethod
    async def broadcast_text(cls, message: str) -> None:
        """Broadcast a text message to all connected clients."""
        disconnected = []
        for connection in cls._active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to websocket: {str(e)}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            await cls.disconnect(connection)
    
    @classmethod
    async def broadcast_json(cls, message: Dict[str, Any]) -> None:
        """Broadcast a JSON message to all connected clients."""
        try:
            json_str = json.dumps(message)
            await cls.broadcast_text(json_str)
        except Exception as e:
            logger.error(f"Error broadcasting JSON message: {str(e)}")
    
    @classmethod
    async def send_personal_message(cls, message: str, websocket: WebSocket) -> None:
        """Send a message to a specific client."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {str(e)}")
            await cls.disconnect(websocket)
    
    @classmethod
    async def broadcast_to_others(cls, message: str, websocket: WebSocket) -> None:
        """Broadcast a message to all clients except the sender."""
        disconnected = []
        for connection in cls._active_connections:
            if connection != websocket:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to others: {str(e)}")
                    disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            await cls.disconnect(connection)
    
    @classmethod
    def get_active_connections(cls) -> List[WebSocket]:
        """Get list of active websocket connections."""
        return cls._active_connections
    
    @classmethod
    def get_connection_count(cls) -> int:
        """Get count of active connections."""
        return len(cls._active_connections)
    
    @classmethod
    def get_connection_id(cls, websocket: WebSocket) -> Optional[str]:
        """Get the ID associated with a websocket connection."""
        return cls._connection_ids.get(websocket) 