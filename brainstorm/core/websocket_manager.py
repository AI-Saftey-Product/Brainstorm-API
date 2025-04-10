"""WebSocket manager for handling real-time communication."""
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import WebSocket
from datetime import datetime

logger = logging.getLogger(__name__)

def _serialize_datetime(obj: Any) -> Any:
    """Recursively serialize datetime objects in a dictionary or list."""
    try:
        if isinstance(obj, datetime):
            logger.debug(f"Found datetime object: {obj}")
            return obj.isoformat()
        elif isinstance(obj, dict):
            logger.debug(f"Processing dictionary with keys: {list(obj.keys())}")
            return {k: _serialize_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            logger.debug(f"Processing list of length: {len(obj)}")
            return [_serialize_datetime(item) for item in obj]
        elif obj is not None:
            logger.debug(f"Processing object of type: {type(obj)}")
        return obj
    except Exception as e:
        logger.error(f"Error in _serialize_datetime: {str(e)}")
        logger.error(f"Object type: {type(obj)}")
        logger.error(f"Object value: {obj}")
        raise

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
            logger.debug(f"Original message type: {type(message)}")
            logger.debug(f"Original message keys: {list(message.keys()) if isinstance(message, dict) else 'not a dict'}")
            
            # Serialize datetime objects before converting to JSON
            logger.debug("Starting datetime serialization...")
            serialized_message = _serialize_datetime(message)
            logger.debug("Datetime serialization completed")
            
            logger.debug("Converting to JSON string...")
            json_str = json.dumps(serialized_message)
            logger.debug("JSON conversion completed")
            
            logger.debug("Broadcasting message...")
            await cls.broadcast_text(json_str)
            logger.debug("Broadcast completed")
        except Exception as e:
            logger.error(f"Error broadcasting JSON message: {str(e)}")
            logger.error(f"Message type: {type(message)}")
            logger.error(f"Message content: {message}")
            logger.error(f"Exception details: {str(e)}")
            raise
    
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