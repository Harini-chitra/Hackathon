from typing import Dict, Any, List, Tuple
from .room_manager import Room
from ..protocols.bb84 import BB84Session
from ..protocols.cow import COWSession
from ..protocols.e91 import E91Session

class RoomWithProtocol(Room):
    def __init__(self, max_participants: int, protocol: str = 'bb84'):
        super().__init__(max_participants)
        self.protocol_name = protocol
        if protocol == 'bb84':
            self.session = BB84Session(room_id='temp')
        elif protocol == 'cow':
            self.session = COWSession(room_id='temp')
        elif protocol == 'e91':
            self.session = E91Session(room_id='temp')
        else:
            self.session = BB84Session(room_id='temp')  # default

    async def handle_message(self, sender_id: str, message: Dict[str, Any]):
        if self.session:
            try:
                # Pass Eve presence info to session
                message['_eve_present'] = self.eve_id is not None
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
