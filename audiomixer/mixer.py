#!/usr/bin/env python3
"""
Shared AudioMixer — device discovery, stream management,
dominant mic detection, recording, and level tracking.
Used by both TUI and Web interfaces.
"""

import glob
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from audio_backend import AudioBackend, AudioDevice

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 1024
NOISE_GATE_RMS = 0.005
DEAD_MIC_BLOCKS = 30
DOMINANT_RELEASE_RMS = 0.008
DOMINANT_MIN_HOLD = 0.5
DOMINANT_SWITCH_RATIO = 1.5
AUTO_SELECT_KEYWORD = "fifine"


@dataclass
class MicState:
    """Real-time state of a single microphone."""
    device: AudioDevice
    level: float = 0.0
    peak: float = 0.0
    smoothed_rms: float = 0.0
    is_dominant: bool = False
    is_muted: bool = False
    is_dead: bool = False
    below_gate: bool = False
    gain: float = 0.0
    _zero_blocks: int = 0


class AudioMixer:
    """Core audio mixer with dominant mic detection."""

    def __init__(self, recordings_dir: str = "."):
        self.backend = AudioBackend()
        self.recordings_dir = recordings_dir
        os.makedirs(self.recordings_dir, exist_ok=True)

        self.devices: list[AudioDevice] = []
        self.mics: dict[int, MicState] = {}
        self.streams: list = []

        self.is_live = False
        self.is_recording = False
        self.dominant_mode = False
        self.dominant_idx: int | None = None

        self._record_buffers: dict[int, list] = {}
        self._record_start_time: float = 0
        self._rms_per_block: dict[int, float] = {}
        self._dominant_hold_until: float = 0

        self._lock = threading.Lock()
        self._listeners: list = []  # callbacks for state changes

    def add_listener(self, callback):
        """Add a callback(event, data) for state change notifications."""
        self._listeners.append(callback)

    def _notify(self, event: str, data=None):
        for cb in self._listeners:
            try:
                cb(event, data)
            except Exception:
                pass

    def discover_devices(self) -> list[AudioDevice]:
        self.devices = self.backend.discover()
        return self.devices

    def get_auto_devices(self) -> list[AudioDevice]:
        return [d for d in self.devices if AUTO_SELECT_KEYWORD in d.name.lower()]

    def start(self, device_indices: list[int] | None = None):
        """Start monitoring selected devices."""
        if self.is_live:
            self.stop()

        selected = []
        if device_indices is None:
            selected = self.get_auto_devices() or self.devices
        else:
            selected = [d for d in self.devices if d.index in device_indices]

        if not selected:
            return

        self.mics.clear()
        for dev in selected:
            self.mics[dev.index] = MicState(device=dev)
            self._record_buffers[dev.index] = []
            self._rms_per_block[dev.index] = 0.0

        self.is_live = True
        for dev in selected:
            stream = self.backend.open_stream(dev, self._make_callback(dev.index))
            stream.start()
            self.streams.append(stream)

        self._notify("started")

    def stop(self):
        """Stop all streams."""
        self.is_live = False
        if self.is_recording:
            self.stop_recording()
        for s in self.streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self.streams.clear()
        self.dominant_idx = None
        self._notify("stopped")

    def start_recording(self):
        if not self.is_live or self.is_recording:
            return
        self.is_recording = True
        self._record_start_time = time.time()
        for idx in self._record_buffers:
            self._record_buffers[idx] = []
        self._notify("recording_started")

    def stop_recording(self) -> str | None:
        if not self.is_recording:
            return None
        self.is_recording = False
        elapsed = time.time() - self._record_start_time
        path = self._save_recording(elapsed)
        self._notify("recording_stopped", {"path": path, "duration": elapsed})
        return path

    def toggle_dominant(self):
        self.dominant_mode = not self.dominant_mode
        if not self.dominant_mode:
            self.dominant_idx = None
            for mic in self.mics.values():
                mic.is_dominant = False
                mic.is_muted = False
        self._notify("dominant_toggled", {"enabled": self.dominant_mode})

    def get_status(self) -> dict:
        mics_data = []
        for idx, mic in self.mics.items():
            mics_data.append({
                "index": idx,
                "name": mic.device.name,
                "level": mic.level,
                "peak": mic.peak,
                "smoothed_rms": mic.smoothed_rms,
                "is_dominant": mic.is_dominant,
                "is_muted": mic.is_muted,
                "is_dead": mic.is_dead,
                "below_gate": mic.below_gate,
            })
        return {
            "is_live": self.is_live,
            "is_recording": self.is_recording,
            "dominant_mode": self.dominant_mode,
            "dominant_idx": self.dominant_idx,
            "recording_duration": (time.time() - self._record_start_time) if self.is_recording else 0,
            "mics": mics_data,
        }

    def get_recordings(self) -> list[dict]:
        files = sorted(glob.glob(os.path.join(self.recordings_dir, "*.wav")), reverse=True)
        recordings = []
        for f in files:
            try:
                info = sf.info(f)
                stat = os.stat(f)
                recordings.append({
                    "filename": os.path.basename(f),
                    "path": f,
                    "duration": info.frames / info.samplerate,
                    "size": stat.st_size,
                    "created": stat.st_mtime,
                })
            except Exception:
                pass
        return recordings

    def delete_recording(self, filename: str) -> bool:
        path = os.path.join(self.recordings_dir, os.path.basename(filename))
        try:
            os.remove(path)
            return True
        except OSError:
            return False

    # ── Internal ─────────────────────────────────────────────────────

    def _make_callback(self, dev_idx: int):
        def callback(indata, frames, time_info, status):
            mono = indata[:, 0] if indata.shape[1] > 1 else indata.flatten()
            rms = float(np.sqrt(np.mean(mono ** 2)))
            scaled = min(rms * 10, 1.0)

            mic = self.mics.get(dev_idx)
            if mic is None:
                return

            mic.level = scaled
            self._rms_per_block[dev_idx] = rms

            # Dead mic detection
            if rms < 1e-10:
                mic._zero_blocks += 1
            else:
                mic._zero_blocks = 0
            mic.is_dead = mic._zero_blocks >= DEAD_MIC_BLOCKS

            # Smooth RMS (EMA)
            alpha = 0.3 if rms > mic.smoothed_rms else 0.05
            mic.smoothed_rms = alpha * rms + (1 - alpha) * mic.smoothed_rms

            if scaled > mic.peak:
                mic.peak = scaled

            # Dominant mic determination
            if self.dominant_mode:
                self._update_dominant(dev_idx)

            # Update UI state
            below_gate = mic.smoothed_rms < NOISE_GATE_RMS
            mic.below_gate = below_gate and not mic.is_dead
            if self.dominant_mode and not mic.is_dead:
                mic.is_dominant = (dev_idx == self.dominant_idx) and not below_gate
                mic.is_muted = (dev_idx != self.dominant_idx) and not below_gate
            else:
                mic.is_dominant = False
                mic.is_muted = False

            # Recording
            if self.is_recording:
                if self.dominant_mode:
                    if self.dominant_idx is None:
                        target = 0.0
                    elif dev_idx == self.dominant_idx:
                        target = 1.0
                    else:
                        target = 0.0
                    mic.gain = mic.gain + 0.15 * (target - mic.gain)
                    self._record_buffers[dev_idx].append(mono * mic.gain)
                else:
                    self._record_buffers[dev_idx].append(mono.copy())

        return callback

    def _update_dominant(self, dev_idx: int):
        now = time.monotonic()
        active_rms = {
            idx: mic.smoothed_rms
            for idx, mic in self.mics.items()
            if mic.smoothed_rms >= NOISE_GATE_RMS and not mic.is_dead
        }

        current_dom = self.dominant_idx

        if not active_rms:
            if now >= self._dominant_hold_until:
                self.dominant_idx = None
        elif current_dom is None or current_dom not in {m.device.index for m in self.mics.values()}:
            loudest = max(active_rms, key=active_rms.get)
            self.dominant_idx = loudest
            self._dominant_hold_until = now + DOMINANT_MIN_HOLD
        else:
            current_rms_val = self.mics[current_dom].smoothed_rms if current_dom in self.mics else 0
            loudest = max(active_rms, key=active_rms.get)
            loudest_rms = active_rms[loudest]

            all_alive = {idx for idx, mic in self.mics.items() if not mic.is_dead}
            above_gate_ratio = len(active_rms) / max(len(all_alive), 1)

            if above_gate_ratio >= 0.5 and loudest != current_dom:
                if loudest_rms > current_rms_val * 1.3:
                    self.dominant_idx = loudest
                    self._dominant_hold_until = now + DOMINANT_MIN_HOLD
            else:
                dom_still_speaking = current_rms_val >= DOMINANT_RELEASE_RMS
                if dom_still_speaking:
                    self._dominant_hold_until = now + DOMINANT_MIN_HOLD
                elif now >= self._dominant_hold_until:
                    if loudest != current_dom and loudest_rms > current_rms_val * DOMINANT_SWITCH_RATIO:
                        self.dominant_idx = loudest
                        self._dominant_hold_until = now + DOMINANT_MIN_HOLD
                    elif current_rms_val < NOISE_GATE_RMS:
                        self.dominant_idx = loudest if active_rms else None

    def _save_recording(self, elapsed: float) -> str | None:
        all_arrays = {}
        max_len = 0
        for idx, chunks in self._record_buffers.items():
            if chunks:
                arr = np.concatenate(chunks)
                all_arrays[idx] = arr
                max_len = max(max_len, len(arr))

        if not all_arrays or max_len == 0:
            return None

        mixed = np.zeros(max_len, dtype=np.float32)
        for arr in all_arrays.values():
            padded = np.zeros(max_len, dtype=np.float32)
            padded[:len(arr)] = arr
            mixed += padded
        mixed /= len(all_arrays)

        peak = np.max(np.abs(mixed))
        if peak > 0:
            mixed = mixed / peak * 0.95

        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.recordings_dir, f"mixed_{ts}.wav")
        sf.write(output_path, mixed, SAMPLE_RATE)
        return output_path

    def decay_peaks(self):
        """Call periodically to decay peak indicators."""
        for mic in self.mics.values():
            mic.peak = max(mic.peak * 0.95, mic.level)
