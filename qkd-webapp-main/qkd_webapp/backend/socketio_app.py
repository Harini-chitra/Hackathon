import asyncio
import os
import random
import string
from typing import Dict, Any, List, Tuple

import socketio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from uuid import uuid4

# Import config
from qkd_webapp.config import Config

# Import after FastAPI setup to avoid circular imports
# from .app import app as core_app  # noqa: E402
# from .websocket.room_manager import RoomManager

# ---------------------------------------------------------------------------
# Socket.IO server setup
# ---------------------------------------------------------------------------

# Initialize FastAPI with CORS
fastapi_app = FastAPI(title="QKD Web Application", version="1.0.0")

# Configure CORS
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Socket.IO
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",  # In production, replace with specific origins
    logger=True,
    engineio_logger=True
)

# Serve the HTML file
frontend_dir = Path(__file__).resolve().parents[2] / "qkd_webapp" / "templates"

@fastapi_app.get("/")
async def root():
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "QKD Web API is running"}

# Mount Socket.IO on FastAPI
app = socketio.ASGIApp(sio, fastapi_app)

# Import here to avoid circular imports
from .websocket.room_manager import RoomManager
room_manager = RoomManager()

# mapping from sid to (room_id, role)
clients: Dict[str, Dict] = {}

# utilities -----------------------------------------------------------------

def random_id(k: int = 6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))

# Socket.IO events ----------------------------------------------------------

@sio.event
async def connect(sid, environ):
    await sio.emit('connected', {'participant_id': sid}, to=sid)
    print(f"Client {sid} connected")

@sio.event
async def disconnect(sid):
    info = clients.get(sid)
    if info:
        room_id = info['room']
        room = room_manager.rooms.get(room_id)
        if room:
            await sio.emit('session_status', _room_status(room_id), room=room_id)
        del clients[sid]
    print(f"Client {sid} disconnected")

# helper to broadcast participants status
def _room_status(room_id):
    room = room_manager.rooms.get(room_id)
    participants = {role: False for role in ['alice', 'bob', 'eve']}
    if room:
        for cid, ws in room.clients.items():
            role = clients.get(cid, {}).get('role')
            if role:
                participants[role] = True
    return {'participants': participants}

# create session ------------------------------------------------------------

@sio.event
async def create_session(sid, data):
    protocol = data.get('protocol', 'bb84').lower()
    room_id = room_manager.create_room(3, protocol)  # capacity 3 (alice,bob,eve)
    await sio.emit('session_created', {'session_id': room_id}, to=sid)
    print(f"Session {room_id} created for protocol {protocol}")

# join session --------------------------------------------------------------

@sio.event
async def join_session(sid, data):
    room_id = data['session_id']
    role = data['role']
    if room_id not in room_manager.rooms:
        await sio.emit('error', {'message': 'Room not found'}, to=sid)
        return
    clients[sid] = {'room': room_id, 'role': role}
    await sio.enter_room(sid, room_id)
    room = room_manager.rooms[room_id]
    
    # Set Eve ID if this is Eve joining
    if role == 'eve':
        room.eve_id = sid
        
    await sio.emit('joined_session', {
        'session_id': room_id, 
        'role': role, 
        'protocol': room.protocol_name
    }, to=sid)
    await sio.emit('session_status', _room_status(room_id), room=room_id)
    print(f"Client {sid} joined session {room_id} as {role}")

# start protocol ------------------------------------------------------------

@sio.event
async def start_protocol(sid, data):
    room_id = data['session_id']
    num_qubits = data.get('num_qubits', 9)
    room = room_manager.rooms.get(room_id)
    if not room:
        await sio.emit('error', {'message': 'Invalid room'}, to=sid)
        return
    
    # delegate to protocol
    responses = await room.session.handle('alice', {'action': 'start', 'num_qubits': num_qubits})
    await _dispatch(room_id, responses)

@sio.event
async def quantum_action(sid, data):
    room_id = data['session_id']
    action = data['action']
    role = clients.get(sid, {}).get('role')
    room = room_manager.rooms.get(room_id)
    if not room:
        return
    responses = await room.session.handle(role, {'action': action})
    await _dispatch(room_id, responses)

# dispatch helper -----------------------------------------------------------

async def _dispatch(room_id, responses):
    for recipient, payload in responses:
        event_name = payload.get('event', 'server_event')
        if recipient == '*':
            await sio.emit(event_name, payload, room=room_id)
        else:
            # find sid of recipient
            target_sid = None
            for sid, info in clients.items():
                if info['room'] == room_id and info['role'] == recipient:
                    target_sid = sid
                    break
            if target_sid:
                await sio.emit(event_name, payload, to=target_sid)
