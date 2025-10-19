from typing import Dict, Any, List, Tuple

class ProtocolSession:
    """Base class for a QKD protocol session running inside a room."""

    def __init__(self, room_id: str):
        self.room_id = room_id
        self.completed = False

    async def handle(self, sender_id: str, message: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Process a message from sender and return list of (recipient_id, payload) to send.
        If recipient_id is '*', send to all except sender."""
        raise NotImplementedError
