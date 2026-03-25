#!/usr/bin/env python3
"""
Web application — HTTP API + WebSocket for real-time audio mixer control.
Serves React frontend and provides API for recordings and mixer state.
"""

import asyncio
import json
import os
import time

from aiohttp import web

from mixer import AudioMixer

RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", ".")
WEB_PORT = int(os.environ.get("WEB_PORT", "8080"))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class WebApp:
    def __init__(self):
        self.mixer = AudioMixer(recordings_dir=RECORDINGS_DIR)
        self.ws_clients: set[web.WebSocketResponse] = set()
        self._broadcast_task = None

    async def start(self):
        self.mixer.discover_devices()
        auto = self.mixer.get_auto_devices()
        if auto:
            self.mixer.start([d.index for d in auto])
        else:
            self.mixer.start()

    async def cleanup(self):
        self.mixer.stop()
        if self._broadcast_task:
            self._broadcast_task.cancel()

    # ── REST API ─────────────────────────────────────────────────────

    async def api_status(self, request):
        return web.json_response(self.mixer.get_status())

    async def api_devices(self, request):
        devs = [{"index": d.index, "name": d.name} for d in self.mixer.devices]
        return web.json_response(devs)

    async def api_record_start(self, request):
        self.mixer.start_recording()
        return web.json_response({"ok": True, "recording": True})

    async def api_record_stop(self, request):
        path = self.mixer.stop_recording()
        return web.json_response({"ok": True, "path": path})

    async def api_dominant_toggle(self, request):
        self.mixer.toggle_dominant()
        return web.json_response({"ok": True, "dominant_mode": self.mixer.dominant_mode})

    async def api_select_devices(self, request):
        data = await request.json()
        indices = data.get("devices", [])
        self.mixer.stop()
        self.mixer.start(indices)
        return web.json_response({"ok": True})

    async def api_recordings(self, request):
        recs = self.mixer.get_recordings()
        for r in recs:
            r.pop("path", None)
        return web.json_response(recs)

    async def api_recording_file(self, request):
        filename = request.match_info["filename"]
        filepath = os.path.join(self.mixer.recordings_dir, os.path.basename(filename))
        if not os.path.exists(filepath):
            return web.Response(status=404)
        return web.FileResponse(filepath, headers={
            "Content-Type": "audio/wav",
            "Accept-Ranges": "bytes",
        })

    async def api_recording_delete(self, request):
        filename = request.match_info["filename"]
        ok = self.mixer.delete_recording(filename)
        return web.json_response({"ok": ok})

    # ── WebSocket ────────────────────────────────────────────────────

    async def ws_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws_clients.add(ws)

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        action = data.get("action")
                        if action == "record_start":
                            self.mixer.start_recording()
                        elif action == "record_stop":
                            self.mixer.stop_recording()
                        elif action == "toggle_dominant":
                            self.mixer.toggle_dominant()
                    except json.JSONDecodeError:
                        pass
                elif msg.type == web.WSMsgType.ERROR:
                    break
        finally:
            self.ws_clients.discard(ws)

        return ws

    async def broadcast_loop(self):
        """Send mixer state to all WebSocket clients at ~20fps."""
        while True:
            if self.ws_clients and self.mixer.is_live:
                self.mixer.decay_peaks()
                status = self.mixer.get_status()
                payload = json.dumps(status)
                dead = set()
                for ws in self.ws_clients:
                    try:
                        await ws.send_str(payload)
                    except Exception:
                        dead.add(ws)
                self.ws_clients -= dead
            await asyncio.sleep(0.05)

    # ── Static files ─────────────────────────────────────────────────

    async def index(self, request):
        return web.FileResponse(os.path.join(STATIC_DIR, "index.html"))

    # ── HA Ingress support ───────────────────────────────────────────

    async def ingress_handler(self, request):
        """Handle HA ingress — same as index but with base path awareness."""
        return web.FileResponse(os.path.join(STATIC_DIR, "index.html"))


def create_app():
    app_instance = WebApp()
    app = web.Application()

    # API routes
    app.router.add_get("/api/status", app_instance.api_status)
    app.router.add_get("/api/devices", app_instance.api_devices)
    app.router.add_post("/api/record/start", app_instance.api_record_start)
    app.router.add_post("/api/record/stop", app_instance.api_record_stop)
    app.router.add_post("/api/dominant/toggle", app_instance.api_dominant_toggle)
    app.router.add_post("/api/devices/select", app_instance.api_select_devices)
    app.router.add_get("/api/recordings", app_instance.api_recordings)
    app.router.add_get("/api/recordings/{filename}", app_instance.api_recording_file)
    app.router.add_delete("/api/recordings/{filename}", app_instance.api_recording_delete)

    # WebSocket
    app.router.add_get("/ws", app_instance.ws_handler)

    # Static / SPA
    app.router.add_get("/", app_instance.index)
    if os.path.isdir(STATIC_DIR):
        app.router.add_static("/static", STATIC_DIR)

    async def on_startup(app):
        await app_instance.start()
        app_instance._broadcast_task = asyncio.create_task(app_instance.broadcast_loop())

    async def on_cleanup(app):
        await app_instance.cleanup()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    return app


if __name__ == "__main__":
    app = create_app()
    print(f"Starting Audio Mixer Web UI on port {WEB_PORT}")
    print(f"Recordings dir: {RECORDINGS_DIR}")
    web.run_app(app, host="0.0.0.0", port=WEB_PORT)
