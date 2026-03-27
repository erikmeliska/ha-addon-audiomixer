#!/usr/bin/env python3
"""
Web application — HTTP API + WebSocket for real-time audio mixer control.
Serves React frontend and provides API for recordings and mixer state.
"""

import asyncio
import json
import os
import time
from typing import Optional

from aiohttp import web

from mixer import AudioMixer
from remote_session import RemoteSession, SessionConfig
from deepgram_streamer import DeepgramConfig

RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", ".")
WEB_PORT = int(os.environ.get("WEB_PORT", "8080"))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class WebApp:
    def __init__(self):
        self.mixer = AudioMixer(recordings_dir=RECORDINGS_DIR)
        self.ws_clients: set[web.WebSocketResponse] = set()
        self._broadcast_task = None
        self.remote_session: Optional[RemoteSession] = None

    async def start(self):
        self.mixer.discover_devices()
        auto = self.mixer.get_auto_devices()
        if auto:
            self.mixer.start([d.index for d in auto])
        else:
            self.mixer.start()

    async def cleanup(self):
        if self.remote_session and self.remote_session.status != "stopped":
            await self.remote_session.stop()
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

    async def api_set_mode(self, request):
        data = await request.json()
        mode = data.get("mode", "smart")
        self.mixer.set_mode(mode)
        return web.json_response({"ok": True, "mode": self.mixer.mode_name})

    async def api_auto_record_toggle(self, request):
        data = await request.json()
        enabled = data.get("enabled", not self.mixer.auto_record_enabled)
        self.mixer.set_auto_record(enabled)
        return web.json_response({"ok": True, "auto_record": self.mixer.auto_record_enabled})

    async def api_set_mic_volume(self, request):
        data = await request.json()
        source = data.get("source_name", "")
        volume = int(data.get("volume", 100))
        volume = max(0, min(150, volume))
        self.mixer.set_mic_volume(source, volume)
        return web.json_response({"ok": True, "volume": volume})

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

    # ── Remote Session API ─────────────────────────────────────────

    async def api_health(self, request):
        """Health check endpoint for meeting dashboard."""
        mics_info = []
        for idx, mic in self.mixer.mics.items():
            mics_info.append({
                "index": idx,
                "name": mic.device.name,
                "is_dead": mic.is_dead,
            })
        return web.json_response({
            "status": "ok",
            "deviceName": "Audio Mixer RPi",
            "version": "1.0.0",
            "activeMics": len([m for m in self.mixer.mics.values() if not m.is_dead]),
            "mics": mics_info,
            "is_live": self.mixer.is_live,
            "remote_session": {
                "active": self.remote_session is not None and self.remote_session.status != "stopped",
                "status": self.remote_session.status if self.remote_session else None,
            }
        })

    async def api_remote_pair(self, request):
        """Pair with a meeting backend session."""
        data = await request.json()

        # Expected: full session config from meeting dashboard
        session_id = data.get("sessionId")
        control_token = data.get("controlToken")
        control_url = data.get("controlUrl")
        transcript_ingest_url = data.get("transcriptIngestUrl")
        events_url = data.get("eventsUrl")
        dg_config = data.get("deepgramConfig", {})
        meeting_id = data.get("meetingId", "")

        if not all([session_id, control_token, control_url, transcript_ingest_url]):
            return web.json_response({"error": "Missing required fields"}, status=400)

        # Stop existing session if any
        if self.remote_session and self.remote_session.status != "stopped":
            await self.remote_session.stop()

        config = SessionConfig(
            session_id=session_id,
            control_token=control_token,
            control_url=control_url,
            transcript_ingest_url=transcript_ingest_url,
            events_url=events_url or "",
            meeting_id=meeting_id,
            deepgram_config=DeepgramConfig(
                key=dg_config.get("key", ""),
                url=dg_config.get("url", "wss://api.deepgram.com/v1/listen"),
                model=dg_config.get("model", "nova-2"),
                language=dg_config.get("language", "sk"),
                diarize=dg_config.get("diarize", True),
                punctuate=dg_config.get("punctuate", True),
                interim_results=dg_config.get("interimResults", True),
                sample_rate=dg_config.get("sampleRate", 16000),
            )
        )

        self.remote_session = RemoteSession(config, mixer=self.mixer)

        # Hook audio feed into mixer callbacks
        self._setup_remote_audio_feed()

        await self.remote_session.start()

        return web.json_response({
            "ok": True,
            "sessionId": session_id,
            "status": self.remote_session.status,
        })

    async def api_remote_status(self, request):
        """Get remote session status."""
        if not self.remote_session:
            return web.json_response({"active": False, "status": None})

        return web.json_response({
            "active": self.remote_session.status != "stopped",
            "status": self.remote_session.status,
            "sessionId": self.remote_session.config.session_id,
            "meetingId": self.remote_session.config.meeting_id,
        })

    async def api_remote_stop(self, request):
        """Stop remote session."""
        if self.remote_session:
            await self.remote_session.stop()
        return web.json_response({"ok": True})

    def _setup_remote_audio_feed(self):
        """Hook mixed audio feed to remote session."""
        def audio_sink(audio_data):
            if self.remote_session and self.remote_session.status == "live":
                self.remote_session.feed_audio(audio_data)

        self.mixer._audio_sink = audio_sink

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
                        elif action == "set_mode":
                            self.mixer.set_mode(data.get("mode", "smart"))
                        elif action == "toggle_auto_record":
                            self.mixer.set_auto_record(not self.mixer.auto_record_enabled)
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
    app.router.add_post("/api/mode", app_instance.api_set_mode)
    app.router.add_post("/api/auto-record", app_instance.api_auto_record_toggle)
    app.router.add_post("/api/mic/volume", app_instance.api_set_mic_volume)
    app.router.add_post("/api/devices/select", app_instance.api_select_devices)
    app.router.add_get("/api/recordings", app_instance.api_recordings)
    app.router.add_get("/api/recordings/{filename}", app_instance.api_recording_file)
    app.router.add_delete("/api/recordings/{filename}", app_instance.api_recording_delete)

    # Remote session / health
    app.router.add_get("/api/health", app_instance.api_health)
    app.router.add_post("/api/remote-session/pair", app_instance.api_remote_pair)
    app.router.add_get("/api/remote-session/status", app_instance.api_remote_status)
    app.router.add_post("/api/remote-session/stop", app_instance.api_remote_stop)

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
