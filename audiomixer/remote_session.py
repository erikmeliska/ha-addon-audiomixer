#!/usr/bin/env python3
"""
Remote session client — connects to meeting backend, receives commands,
streams audio via Deepgram, and forwards transcripts back.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
import numpy as np

from deepgram_streamer import DeepgramConfig, DeepgramStreamer, TranscriptEvent

logger = logging.getLogger(__name__)

TRANSCRIPT_BATCH_INTERVAL = 0.3  # seconds between transcript POSTs


@dataclass
class SessionConfig:
    """Configuration received from meeting backend."""
    session_id: str
    control_token: str
    control_url: str
    transcript_ingest_url: str
    events_url: str
    deepgram_config: DeepgramConfig
    meeting_id: str = ""


class RemoteSession:
    """
    Manages a remote audio session with the meeting backend.

    Lifecycle:
    1. Dashboard creates session -> returns SessionConfig
    2. RPi connects and starts polling for commands
    3. On 'start' command -> connect to Deepgram, stream audio
    4. Transcripts forwarded to backend via POST
    5. On 'pause' -> pause Deepgram streaming
    6. On 'stop' -> disconnect everything
    """

    def __init__(self, config: SessionConfig, mixer=None):
        self.config = config
        self.mixer = mixer

        self._streamer: Optional[DeepgramStreamer] = None
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._transcript_task: Optional[asyncio.Task] = None
        self._running = False
        self._status = "armed"

        # Transcript buffer for batching
        self._transcript_queue: asyncio.Queue = asyncio.Queue()

        # Local transcript storage
        self._local_transcripts: list[dict] = []
        self._local_audio_path: Optional[str] = None

        # Callbacks
        self.on_status_change: Optional[callable] = None

    @property
    def status(self) -> str:
        return self._status

    async def start(self):
        """Start the remote session - begin polling for commands."""
        self._http_session = aiohttp.ClientSession()
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_commands())
        self._transcript_task = asyncio.create_task(self._transcript_sender())
        logger.info(f"Remote session started: {self.config.session_id}")

    async def stop(self):
        """Stop the remote session completely."""
        self._running = False

        if self._streamer and self._streamer.is_connected:
            await self._streamer.disconnect()
            self._streamer = None

        for task in [self._poll_task, self._transcript_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

        self._set_status("stopped")

        # Save local transcripts
        self._save_local_transcripts()

        logger.info("Remote session stopped")

    def _set_status(self, new_status: str):
        old = self._status
        self._status = new_status
        if old != new_status:
            logger.info(f"Session status: {old} -> {new_status}")
            if self.on_status_change:
                self.on_status_change(new_status)

    # -- Audio Integration ------------------------------------------------

    def _setup_audio_hook(self):
        """Hook into the mixer to get audio data."""
        if not self.mixer:
            logger.warning("No mixer attached - cannot capture audio")
            return

        def audio_callback(event, data=None):
            """Called by mixer for various events - we don't need this for audio."""
            pass

        self.mixer.add_listener(audio_callback)

    def feed_audio(self, audio_data: np.ndarray):
        """
        Feed mixed audio data to the Deepgram streamer.
        Called from the mixer's audio callback.
        """
        if self._streamer and self._streamer.is_connected:
            self._streamer.send_audio(audio_data)

    # -- Deepgram Integration ---------------------------------------------

    def _on_transcript(self, event: TranscriptEvent):
        """Handle transcript from Deepgram."""
        # Only forward final transcripts to backend
        if event.is_final and event.transcript.strip():
            transcript_data = {
                "transcript": event.transcript,
                "speaker": event.speaker,
                "isFinal": True,
                "confidence": event.confidence,
                "timestamp": int(event.timestamp * 1000),
            }

            # Queue for sending to backend
            try:
                self._transcript_queue.put_nowait(transcript_data)
            except asyncio.QueueFull:
                logger.warning("Transcript queue full, dropping event")

            # Store locally
            self._local_transcripts.append(transcript_data)

    async def _start_streaming(self):
        """Start streaming audio to Deepgram."""
        self._streamer = DeepgramStreamer(
            config=self.config.deepgram_config,
            on_transcript=self._on_transcript,
            on_error=lambda e: logger.error(f"Deepgram error: {e}"),
        )
        await self._streamer.connect()

        # Also start recording locally if mixer supports it
        if self.mixer and not self.mixer.is_recording:
            self.mixer.start_recording()

        self._set_status("live")

    async def _pause_streaming(self):
        """Pause audio streaming."""
        if self._streamer:
            self._streamer.pause()
        self._set_status("paused")

    async def _resume_streaming(self):
        """Resume audio streaming."""
        if self._streamer:
            self._streamer.resume()
        self._set_status("live")

    async def _stop_streaming(self):
        """Stop streaming and disconnect."""
        if self._streamer:
            await self._streamer.disconnect()
            self._streamer = None

        # Stop local recording
        if self.mixer and self.mixer.is_recording:
            path = self.mixer.stop_recording()
            self._local_audio_path = path

        self._set_status("stopped")

    # -- Command Polling --------------------------------------------------

    async def _poll_commands(self):
        """Poll the backend for commands."""
        try:
            while self._running:
                try:
                    url = f"{self.config.control_url}?token={self.config.control_token}"
                    async with self._http_session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            backend_status = data.get("status", "")

                            # React to status changes from backend
                            if backend_status != self._status:
                                await self._handle_status_change(backend_status)
                except aiohttp.ClientError as e:
                    logger.warning(f"Poll error: {e}")
                except asyncio.TimeoutError:
                    pass

                await asyncio.sleep(1)  # Poll every 1 second
        except asyncio.CancelledError:
            pass

    async def _handle_status_change(self, new_status: str):
        """Handle a status change from the backend."""
        if new_status == "live" and self._status in ("armed", "paused"):
            if self._status == "armed":
                await self._start_streaming()
            else:
                await self._resume_streaming()
        elif new_status == "paused" and self._status == "live":
            await self._pause_streaming()
        elif new_status == "stopped":
            await self._stop_streaming()

    # -- Transcript Forwarding --------------------------------------------

    async def _transcript_sender(self):
        """Send transcript events to backend in batches."""
        try:
            while self._running:
                events = []

                # Collect all pending events
                try:
                    while True:
                        event = self._transcript_queue.get_nowait()
                        events.append(event)
                except asyncio.QueueEmpty:
                    pass

                if events:
                    # Send each event individually (backend expects one at a time)
                    for event in events:
                        try:
                            payload = {
                                "controlToken": self.config.control_token,
                                **event
                            }
                            async with self._http_session.post(
                                self.config.transcript_ingest_url,
                                json=payload,
                                timeout=aiohttp.ClientTimeout(total=5)
                            ) as resp:
                                if resp.status != 200:
                                    logger.warning(f"Transcript POST failed: {resp.status}")
                        except Exception as e:
                            logger.warning(f"Transcript send error: {e}")

                await asyncio.sleep(TRANSCRIPT_BATCH_INTERVAL)
        except asyncio.CancelledError:
            pass

    # -- Local Storage ----------------------------------------------------

    def _save_local_transcripts(self):
        """Save transcripts locally for backup."""
        if not self._local_transcripts:
            return

        data_dir = os.environ.get("RECORDINGS_DIR", ".")
        meeting_dir = os.path.join(data_dir, "meetings", self.config.meeting_id or self.config.session_id)
        os.makedirs(meeting_dir, exist_ok=True)

        transcript_path = os.path.join(meeting_dir, "transcript.json")
        meta_path = os.path.join(meeting_dir, "meta.json")

        import json as json_mod
        with open(transcript_path, "w") as f:
            json_mod.dump(self._local_transcripts, f, indent=2, ensure_ascii=False)

        meta = {
            "sessionId": self.config.session_id,
            "meetingId": self.config.meeting_id,
            "transcriptCount": len(self._local_transcripts),
            "audioPath": self._local_audio_path,
            "savedAt": time.time(),
        }
        with open(meta_path, "w") as f:
            json_mod.dump(meta, f, indent=2)

        logger.info(f"Saved {len(self._local_transcripts)} transcripts to {meeting_dir}")
