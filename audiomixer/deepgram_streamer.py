#!/usr/bin/env python3
"""
Deepgram WebSocket streaming client.
Sends PCM audio to Deepgram and receives transcription results.
"""

import asyncio
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DeepgramConfig:
    """Configuration for Deepgram connection."""
    key: str
    url: str = "wss://api.deepgram.com/v1/listen"
    model: str = "nova-2"
    language: str = "sk"
    diarize: bool = True
    punctuate: bool = True
    interim_results: bool = True
    encoding: str = "linear16"
    sample_rate: int = 16000
    channels: int = 1

    def ws_url(self) -> str:
        params = (
            f"model={self.model}"
            f"&language={self.language}"
            f"&punctuate={str(self.punctuate).lower()}"
            f"&diarize={str(self.diarize).lower()}"
            f"&interim_results={str(self.interim_results).lower()}"
            f"&utterance_end_ms=1000"
            f"&vad_events=true"
            f"&smart_format=true"
            f"&encoding={self.encoding}"
            f"&sample_rate={self.sample_rate}"
            f"&channels={self.channels}"
        )
        return f"{self.url}?{params}"


@dataclass
class TranscriptEvent:
    """A single transcript event from Deepgram."""
    transcript: str
    speaker: Optional[int]
    is_final: bool
    confidence: float
    timestamp: float = field(default_factory=time.time)


class DeepgramStreamer:
    """
    Streams audio to Deepgram via WebSocket and receives transcripts.

    Usage:
        streamer = DeepgramStreamer(config, on_transcript=my_callback)
        await streamer.connect()
        streamer.send_audio(pcm_bytes)  # call from audio callback
        await streamer.disconnect()
    """

    def __init__(
        self,
        config: DeepgramConfig,
        on_transcript: Optional[Callable[[TranscriptEvent], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self.config = config
        self.on_transcript = on_transcript
        self.on_error = on_error

        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._send_task: Optional[asyncio.Task] = None
        self._connected = False
        self._paused = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None and not self._ws.closed

    async def connect(self):
        """Connect to Deepgram WebSocket."""
        try:
            self._session = aiohttp.ClientSession()
            headers = {"Authorization": f"Token {self.config.key}"}

            self._ws = await self._session.ws_connect(
                self.config.ws_url(),
                headers=headers,
                heartbeat=30,
            )
            self._connected = True

            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())
            self._send_task = asyncio.create_task(self._send_loop())

            logger.info("Connected to Deepgram")
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")
            if self.on_error:
                self.on_error(e)
            raise

    async def disconnect(self):
        """Disconnect from Deepgram."""
        self._connected = False

        for task in [self._receive_task, self._keepalive_task, self._send_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._ws and not self._ws.closed:
            # Send close signal
            try:
                await self._ws.send_bytes(b'')  # Empty byte message signals end
                await self._ws.close()
            except Exception:
                pass

        if self._session and not self._session.closed:
            await self._session.close()

        self._ws = None
        self._session = None
        logger.info("Disconnected from Deepgram")

    def send_audio(self, audio_data: np.ndarray):
        """
        Queue audio data for sending to Deepgram.
        audio_data: numpy array of float32 samples (-1.0 to 1.0)
        Called from audio callback thread - must be thread-safe.
        """
        if not self._connected or self._paused:
            return

        # Convert float32 to int16 PCM bytes
        int16_data = (audio_data * 32767).astype(np.int16)
        pcm_bytes = int16_data.tobytes()

        try:
            self._audio_queue.put_nowait(pcm_bytes)
        except asyncio.QueueFull:
            pass  # Drop audio if queue is full (backpressure)

    def pause(self):
        """Pause sending audio (keep connection alive)."""
        self._paused = True
        logger.info("Deepgram streaming paused")

    def resume(self):
        """Resume sending audio."""
        self._paused = False
        logger.info("Deepgram streaming resumed")

    async def _send_loop(self):
        """Send queued audio data to Deepgram."""
        try:
            while self._connected:
                try:
                    pcm_bytes = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.1
                    )
                    if self._ws and not self._ws.closed and not self._paused:
                        await self._ws.send_bytes(pcm_bytes)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Send loop error: {e}")

    async def _receive_loop(self):
        """Receive transcription results from Deepgram."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        if data.get("type") == "Results":
                            alt = data.get("channel", {}).get("alternatives", [{}])[0]
                            transcript = alt.get("transcript", "")
                            if transcript:
                                words = alt.get("words", [])
                                speaker = words[0].get("speaker") if words else None
                                confidence = alt.get("confidence", 0)
                                is_final = data.get("is_final", False)

                                event = TranscriptEvent(
                                    transcript=transcript,
                                    speaker=speaker,
                                    is_final=is_final,
                                    confidence=confidence,
                                )

                                if self.on_transcript:
                                    self.on_transcript(event)
                    except json.JSONDecodeError:
                        pass
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receive loop error: {e}")
            if self.on_error:
                self.on_error(e)

    async def _keepalive_loop(self):
        """Send keepalive messages to Deepgram."""
        try:
            while self._connected:
                await asyncio.sleep(10)
                if self._ws and not self._ws.closed:
                    await self._ws.send_str(json.dumps({"type": "KeepAlive"}))
        except asyncio.CancelledError:
            pass
