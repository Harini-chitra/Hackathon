from typing import Dict, List, Tuple, Any
from fastapi import WebSocket
from uuid import uuid4
from ..protocols.bb84 import BB84Session
from ..protocols.cow import COWSession
from ..protocols.e91 import E91Session

class Room:
    """A room represents a QKD session. Handles WebSocket connections and basic broadcast."""

    def __init__(self, max_participants: int, protocol: str = "bb84", room_id: str | None = None):
        self.max_participants = max_participants
        self.protocol_name = protocol
        self.clients: Dict[str, WebSocket] = {}
        self.eve_id: str | None = None  # Track Eve client
        # create protocol session instance (only bb84 for now)
        protocol_map = {
            "bb84": BB84Session,
            "cow": COWSession,
            "e91": E91Session,
        }
        cls = protocol_map.get(protocol.lower())
        self.session = cls(room_id or "temp") if cls else None

    def is_full(self) -> bool:
        return len(self.clients) >= self.max_participants

    async def connect(self, client_id: str, websocket: WebSocket):
        if self.is_full():
            raise ValueError("Room is full")
        await websocket.accept()
        self.clients[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.clients:
            del self.clients[client_id]
            # Clear Eve slot if she left
            if self.eve_id == client_id:
                self.eve_id = None

    async def broadcast(self, sender_id: str, message):
        # First let protocol session process
        if self.session and not self.session.completed:
            try:
                responses: List[Tuple[str, Dict[str, Any]]] = await self.session.handle(sender_id, message)
                for recipient, payload in responses:
                    if recipient == "*":
                        for cid, ws in self.clients.items():
                            if cid != sender_id:
                                await ws.send_json({"from": "server", "payload": payload})
                    elif recipient == "eve" and self.eve_id:
                        ws = self.clients.get(self.eve_id)
                        if ws:
                            await ws.send_json({"from": "server", "payload": payload})
                    else:
                        ws = self.clients.get(recipient)
                        if ws:
                            await ws.send_json({"from": "server", "payload": payload})
            except Exception as exc:
                print("Protocol handling error", exc)
        # Echo original message to others
        for cid, ws in self.clients.items():
            if cid != sender_id:
                await ws.send_json({"from": sender_id, "payload": message})

class RoomManager:
    """Keeps track of active rooms and provides helper methods."""

    def __init__(self):
        self.rooms: Dict[str, Room] = {}

    def create_room(self, participants: int, protocol: str = "bb84") -> str:
        room_id = uuid4().hex[:8]
        self.rooms[room_id] = Room(max_participants=participants, protocol=protocol, room_id=room_id)
        return room_id

    def list_rooms(self):
        return {rid: {"participants": len(room.clients), "capacity": room.max_participants, "protocol": room.protocol_name} for rid, room in self.rooms.items()}

    async def connect(self, room_id: str, client_id: str, websocket: WebSocket):
        if room_id not in self.rooms:
            raise ValueError("Room not found")
        await self.rooms[room_id].connect(client_id, websocket)

    def disconnect(self, room_id: str, client_id: str):
        if room_id in self.rooms:
            self.rooms[room_id].disconnect(client_id)
            # remove empty room
            if not self.rooms[room_id].clients:
                del self.rooms[room_id]

    async def broadcast(self, room_id: str, sender_id: str, message):
        if room_id in self.rooms:
            await self.rooms[room_id].broadcast(sender_id, message)
