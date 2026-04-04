"""
WebSocket Server — handles source/target language switching
"""

import asyncio
import queue
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

_msg_queue:  queue.Queue = None
_transcriber             = None

def set_message_queue(q: queue.Queue):
    global _msg_queue
    _msg_queue = q

def set_transcriber(t):
    global _transcriber
    _transcriber = t


class ConnectionManager:
    def __init__(self):
        self.connections: set[WebSocket] = set()
        self.latest_info: dict | None = None
        self.latest_status: dict | None = None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.add(ws)
        if self.latest_info:
            await ws.send_json(self.latest_info)
        if self.latest_status:
            await ws.send_json(self.latest_status)

    def disconnect(self, ws: WebSocket):
        self.connections.discard(ws)

    async def broadcast(self, data: dict):
        if data.get("type") == "info":
            self.latest_info = data
        elif data.get("type") == "status":
            self.latest_status = data
            
        dead = set()
        for ws in self.connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self.connections -= dead

manager = ConnectionManager()


def _loop_exception_handler(loop, context):
    exc = context.get("exception")
    if isinstance(exc, ConnectionResetError) and getattr(exc, "winerror", None) == 10054:
        return
    loop.default_exception_handler(context)


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "set_source" and _transcriber:
                lang = data.get("language", "vi")
                _transcriber.set_source_lang(lang)
                await manager.broadcast({"type": "source_changed", "language": lang})

            elif msg_type == "set_target" and _transcriber:
                lang = data.get("language", "none")
                _transcriber.set_target_lang(lang)
                await manager.broadcast({"type": "target_changed", "language": lang})

            elif msg_type == "set_save_transcript" and _transcriber:
                enabled = data.get("enabled", True)
                _transcriber.set_save_transcript(enabled)
                await manager.broadcast({"type": "save_transcript_changed", "enabled": enabled})

            elif msg_type == "set_log_terminal" and _transcriber:
                enabled = data.get("enabled", True)
                _transcriber.set_log_terminal(enabled)
                await manager.broadcast({"type": "log_terminal_changed", "enabled": enabled})

            elif msg_type == "set_model" and _transcriber:
                model = data.get("model", "base")
                _transcriber.set_model(model)

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


@app.on_event("startup")
async def start_queue_poller():
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_loop_exception_handler)
    asyncio.create_task(_poll_queue())


async def _poll_queue():
    while True:
        try:
            if _msg_queue is not None:
                while not _msg_queue.empty():
                    msg = _msg_queue.get_nowait()
                    await manager.broadcast(msg)
        except Exception:
            pass
        await asyncio.sleep(0.005)  # 5ms → faster response
