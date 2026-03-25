#!/usr/bin/env python3
"""
Shared AudioMixer with pluggable mixing modes.
Modes: mix_all, dominant, dugan, dugan_gated, dugan_strict, smart.
"""

import collections
import glob
import os
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from audio_backend import AudioBackend, AudioDevice

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 1024
NOISE_GATE_RMS = 0.005
DEAD_MIC_BLOCKS = 30
AUTO_SELECT_KEYWORD = "fifine"

# Smart mode correlation
CORR_WINDOW = 2048  # samples for cross-correlation
CORR_UPDATE_INTERVAL = 0.15  # seconds between correlation updates
CORR_ECHO_THRESHOLD = 0.6  # above this = same source (echo)
CORR_INDEPENDENT_THRESHOLD = 0.3  # below this = independent speakers


@dataclass
class MicState:
    device: AudioDevice
    level: float = 0.0
    peak: float = 0.0
    smoothed_rms: float = 0.0
    is_dominant: bool = False
    is_muted: bool = False
    is_dead: bool = False
    below_gate: bool = False
    gain: float = 0.0
    weight: float = 0.0
    correlation: float = 0.0  # correlation with dominant mic (0-1)
    _zero_blocks: int = 0


# ── Mixing Modes ─────────────────────────────────────────────────

class MixMode(ABC):
    name: str = "base"
    label: str = "Base"

    def reset(self):
        pass

    def feed_audio(self, dev_idx: int, samples: np.ndarray):
        """Override to receive raw audio per block."""
        pass

    @abstractmethod
    def compute_weights(self, mics: dict[int, MicState]) -> dict[int, float]:
        ...

    def update_ui_state(self, mics: dict[int, MicState], weights: dict[int, float]):
        for idx, mic in mics.items():
            mic.weight = weights.get(idx, 0)
            mic.below_gate = mic.smoothed_rms < NOISE_GATE_RMS and not mic.is_dead


class MixAllMode(MixMode):
    name = "mix_all"
    label = "Mix All"

    def compute_weights(self, mics):
        alive = {idx for idx, m in mics.items() if not m.is_dead}
        if not alive:
            return {idx: 0.0 for idx in mics}
        w = 1.0 / len(alive)
        return {idx: (w if idx in alive else 0.0) for idx in mics}

    def update_ui_state(self, mics, weights):
        super().update_ui_state(mics, weights)
        for mic in mics.values():
            mic.is_dominant = False
            mic.is_muted = False


class DominantMode(MixMode):
    name = "dominant"
    label = "Dominant"

    def __init__(self):
        self.dominant_idx = None
        self._hold_until = 0.0

    def reset(self):
        self.dominant_idx = None
        self._hold_until = 0.0

    def compute_weights(self, mics):
        now = time.monotonic()
        active = {
            idx: m.smoothed_rms for idx, m in mics.items()
            if m.smoothed_rms >= NOISE_GATE_RMS and not m.is_dead
        }

        if not active:
            if now >= self._hold_until:
                self.dominant_idx = None
        elif self.dominant_idx is None or self.dominant_idx not in active:
            self.dominant_idx = max(active, key=active.get)
            self._hold_until = now + 0.5
        else:
            current_rms = mics[self.dominant_idx].smoothed_rms
            loudest = max(active, key=active.get)
            if current_rms >= 0.008:
                self._hold_until = now + 0.5
            elif now >= self._hold_until:
                if loudest != self.dominant_idx and active[loudest] > current_rms * 1.5:
                    self.dominant_idx = loudest
                    self._hold_until = now + 0.5
                elif current_rms < NOISE_GATE_RMS:
                    self.dominant_idx = loudest if active else None

        return {idx: (1.0 if idx == self.dominant_idx else 0.0) for idx in mics}

    def update_ui_state(self, mics, weights):
        super().update_ui_state(mics, weights)
        for idx, mic in mics.items():
            below = mic.smoothed_rms < NOISE_GATE_RMS
            mic.is_dominant = (idx == self.dominant_idx) and not below and not mic.is_dead
            mic.is_muted = (idx != self.dominant_idx) and not below and not mic.is_dead


class DuganMode(MixMode):
    """Dugan automixing with optional weight gating."""
    name = "dugan"
    label = "Dugan"
    EXPONENT = 6
    GATE_RATIO = 0.0

    def compute_weights(self, mics):
        alive = {idx: m for idx, m in mics.items() if not m.is_dead}
        if not alive:
            return {idx: 0.0 for idx in mics}

        energies = {}
        for idx, m in alive.items():
            if m.smoothed_rms < NOISE_GATE_RMS:
                energies[idx] = 0.0
            else:
                energies[idx] = m.smoothed_rms ** self.EXPONENT

        total = sum(energies.values())
        if total < 1e-30:
            w = 1.0 / len(alive)
            return {idx: (w if idx in alive else 0.0) for idx in mics}

        weights = {idx: (energies.get(idx, 0.0) / total if idx in alive else 0.0) for idx in mics}

        if self.GATE_RATIO > 0:
            max_w = max(weights.values())
            threshold = max_w * self.GATE_RATIO
            for idx in weights:
                if weights[idx] < threshold:
                    weights[idx] = 0.0
            total_w = sum(weights.values())
            if total_w > 0:
                weights = {idx: w / total_w for idx, w in weights.items()}

        return weights

    def update_ui_state(self, mics, weights):
        super().update_ui_state(mics, weights)
        if not weights:
            return
        max_w = max(weights.values())
        for idx, mic in mics.items():
            w = weights.get(idx, 0)
            mic.is_dominant = (w == max_w and w > 0.3) and not mic.is_dead
            mic.is_muted = (w < 0.05) and not mic.is_dead and not mic.below_gate


class DuganGatedMode(DuganMode):
    name = "dugan_gated"
    label = "Dugan Gated"
    EXPONENT = 6
    GATE_RATIO = 0.5


class DuganStrictMode(DuganMode):
    name = "dugan_strict"
    label = "Dugan Strict"
    EXPONENT = 8
    GATE_RATIO = 0.7


class SmartMode(MixMode):
    """Smart mode: Dugan + correlation-based echo detection.

    For each pair of active mics, computes cross-correlation:
    - High correlation (>0.6) = same source heard by both → gate the weaker one
    - Low correlation (<0.3) = different speakers → allow both with Dugan weights

    This lets multiple simultaneous speakers through while suppressing echo.
    """
    name = "smart"
    label = "Smart"
    EXPONENT = 6

    def __init__(self):
        self._ring_buffers: dict[int, collections.deque] = {}
        self._correlations: dict[tuple[int, int], float] = {}
        self._last_corr_update: float = 0

    def reset(self):
        self._ring_buffers.clear()
        self._correlations.clear()
        self._last_corr_update = 0

    def feed_audio(self, dev_idx: int, samples: np.ndarray):
        if dev_idx not in self._ring_buffers:
            self._ring_buffers[dev_idx] = collections.deque(maxlen=CORR_WINDOW)
        self._ring_buffers[dev_idx].extend(samples.tolist())

    def _update_correlations(self, mics: dict[int, MicState]):
        now = time.monotonic()
        if now - self._last_corr_update < CORR_UPDATE_INTERVAL:
            return
        self._last_corr_update = now

        active_ids = [
            idx for idx, m in mics.items()
            if m.smoothed_rms >= NOISE_GATE_RMS and not m.is_dead
        ]

        for i, idx_a in enumerate(active_ids):
            for idx_b in active_ids[i + 1:]:
                buf_a = self._ring_buffers.get(idx_a)
                buf_b = self._ring_buffers.get(idx_b)
                if not buf_a or not buf_b:
                    continue

                n = min(len(buf_a), len(buf_b), CORR_WINDOW)
                if n < BLOCKSIZE:
                    continue

                a = np.array(list(buf_a)[-n:], dtype=np.float32)
                b = np.array(list(buf_b)[-n:], dtype=np.float32)

                # Normalized cross-correlation (Pearson)
                a_mean = a - np.mean(a)
                b_mean = b - np.mean(b)
                norm_a = np.sqrt(np.sum(a_mean ** 2))
                norm_b = np.sqrt(np.sum(b_mean ** 2))

                if norm_a < 1e-10 or norm_b < 1e-10:
                    corr = 0.0
                else:
                    corr = float(np.sum(a_mean * b_mean) / (norm_a * norm_b))
                    corr = max(0.0, corr)  # only positive correlation matters

                self._correlations[(idx_a, idx_b)] = corr
                self._correlations[(idx_b, idx_a)] = corr

    def _get_correlation(self, idx_a: int, idx_b: int) -> float:
        return self._correlations.get((idx_a, idx_b), 0.0)

    def compute_weights(self, mics):
        self._update_correlations(mics)

        alive = {idx: m for idx, m in mics.items() if not m.is_dead}
        if not alive:
            return {idx: 0.0 for idx in mics}

        # Step 1: Compute base Dugan weights
        energies = {}
        for idx, m in alive.items():
            if m.smoothed_rms < NOISE_GATE_RMS:
                energies[idx] = 0.0
            else:
                energies[idx] = m.smoothed_rms ** self.EXPONENT

        total = sum(energies.values())
        if total < 1e-30:
            w = 1.0 / len(alive)
            return {idx: (w if idx in alive else 0.0) for idx in mics}

        weights = {idx: energies.get(idx, 0.0) / total for idx in alive}

        # Step 2: For each mic pair, check correlation
        # If highly correlated (echo), suppress the weaker one
        # If uncorrelated (different speakers), keep both
        active_ids = [idx for idx in alive if weights.get(idx, 0) > 0.01]
        suppressed = set()

        for i, idx_a in enumerate(active_ids):
            for idx_b in active_ids[i + 1:]:
                corr = self._get_correlation(idx_a, idx_b)

                if corr > CORR_ECHO_THRESHOLD:
                    # Same source — suppress the weaker mic
                    if weights[idx_a] >= weights[idx_b]:
                        suppressed.add(idx_b)
                    else:
                        suppressed.add(idx_a)
                # If corr < CORR_INDEPENDENT_THRESHOLD → independent, keep both
                # If between thresholds → partial suppression via weight reduction
                elif corr > CORR_INDEPENDENT_THRESHOLD:
                    # Transition zone: reduce weaker mic proportionally
                    blend = (corr - CORR_INDEPENDENT_THRESHOLD) / (CORR_ECHO_THRESHOLD - CORR_INDEPENDENT_THRESHOLD)
                    weaker = idx_b if weights[idx_a] >= weights[idx_b] else idx_a
                    weights[weaker] *= (1.0 - blend * 0.9)  # reduce up to 90%

        # Apply suppression
        for idx in suppressed:
            weights[idx] = 0.0

        # Renormalize
        final = {idx: 0.0 for idx in mics}
        for idx in alive:
            final[idx] = weights.get(idx, 0.0)
        total_w = sum(final.values())
        if total_w > 0:
            final = {idx: w / total_w for idx, w in final.items()}

        return final

    def update_ui_state(self, mics, weights):
        super().update_ui_state(mics, weights)
        if not weights:
            return
        max_w = max(weights.values()) if weights else 0
        for idx, mic in mics.items():
            w = weights.get(idx, 0)
            mic.is_dominant = (w == max_w and w > 0.3) and not mic.is_dead
            mic.is_muted = (w < 0.05) and not mic.is_dead and not mic.below_gate
            # Show correlation with dominant mic
            if mic.is_dominant:
                mic.correlation = 1.0
            else:
                dominant_idx = next((i for i, m in mics.items() if m.is_dominant), None)
                if dominant_idx is not None:
                    mic.correlation = self._get_correlation(idx, dominant_idx)
                else:
                    mic.correlation = 0.0


# ── Mode registry ────────────────────────────────────────────────

MIXING_MODES: dict[str, type[MixMode]] = {
    "mix_all": MixAllMode,
    "dominant": DominantMode,
    "dugan": DuganMode,
    "dugan_gated": DuganGatedMode,
    "dugan_strict": DuganStrictMode,
    "smart": SmartMode,
}

GAIN_SMOOTHING = 0.15


# ── AudioMixer ───────────────────────────────────────────────────

class AudioMixer:
    """Core audio mixer with pluggable mixing modes."""

    def __init__(self, recordings_dir: str = "."):
        self.backend = AudioBackend()
        self.recordings_dir = recordings_dir
        os.makedirs(self.recordings_dir, exist_ok=True)

        self.devices: list[AudioDevice] = []
        self.mics: dict[int, MicState] = {}
        self.streams: list = []

        self.is_live = False
        self.is_recording = False

        self._modes: dict[str, MixMode] = {
            name: cls() for name, cls in MIXING_MODES.items()
        }
        self._current_mode_name = "smart"
        self._mode: MixMode = self._modes[self._current_mode_name]

        self._record_buffers: dict[int, list] = {}
        self._record_start_time: float = 0

        self._lock = threading.Lock()
        self._listeners: list = []

    @property
    def mode_name(self) -> str:
        return self._current_mode_name

    @property
    def available_modes(self) -> list[dict]:
        return [{"name": name, "label": cls.label} for name, cls in MIXING_MODES.items()]

    def set_mode(self, mode_name: str):
        if mode_name not in self._modes:
            return
        self._current_mode_name = mode_name
        self._mode = self._modes[mode_name]
        self._mode.reset()
        for mic in self.mics.values():
            mic.is_dominant = False
            mic.is_muted = False
            mic.gain = 0.0
            mic.weight = 0.0
            mic.correlation = 0.0
        self._notify("mode_changed", {"mode": mode_name})

    def set_mic_volume(self, source_name: str, volume_pct: int):
        """Set PulseAudio source volume (0-100%)."""
        try:
            subprocess.run(
                ["pactl", "set-source-volume", source_name, f"{volume_pct}%"],
                capture_output=True, timeout=3,
            )
        except Exception:
            pass

    def add_listener(self, callback):
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

        self._mode.reset()
        self.is_live = True

        for dev in selected:
            stream = self.backend.open_stream(dev, self._make_callback(dev.index))
            stream.start()
            self.streams.append(stream)

        self._notify("started")

    def stop(self):
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

    def get_status(self) -> dict:
        mics_data = []
        for idx, mic in self.mics.items():
            mics_data.append({
                "index": idx,
                "name": mic.device.name,
                "source_name": mic.device.source_name,
                "level": mic.level,
                "peak": mic.peak,
                "smoothed_rms": mic.smoothed_rms,
                "is_dominant": mic.is_dominant,
                "is_muted": mic.is_muted,
                "is_dead": mic.is_dead,
                "below_gate": mic.below_gate,
                "weight": round(mic.weight, 3),
                "correlation": round(mic.correlation, 2),
            })
        return {
            "is_live": self.is_live,
            "is_recording": self.is_recording,
            "mode": self._current_mode_name,
            "mode_label": self._mode.label,
            "available_modes": self.available_modes,
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

    # ── Internal ─────────────────────────────────────────────────

    def _make_callback(self, dev_idx: int):
        def callback(indata, frames, time_info, status):
            mono = indata[:, 0] if indata.shape[1] > 1 else indata.flatten()
            rms = float(np.sqrt(np.mean(mono ** 2)))
            scaled = min(rms * 10, 1.0)

            mic = self.mics.get(dev_idx)
            if mic is None:
                return

            mic.level = scaled

            if rms < 1e-10:
                mic._zero_blocks += 1
            else:
                mic._zero_blocks = 0
            mic.is_dead = mic._zero_blocks >= DEAD_MIC_BLOCKS

            alpha = 0.3 if rms > mic.smoothed_rms else 0.05
            mic.smoothed_rms = alpha * rms + (1 - alpha) * mic.smoothed_rms

            if scaled > mic.peak:
                mic.peak = scaled

            # Feed raw audio to mode (for correlation etc.)
            self._mode.feed_audio(dev_idx, mono)

            # Compute mixing weights
            weights = self._mode.compute_weights(self.mics)
            self._mode.update_ui_state(self.mics, weights)

            # Recording: apply smoothed gain
            if self.is_recording:
                target = weights.get(dev_idx, 0.0)
                mic.gain += GAIN_SMOOTHING * (target - mic.gain)
                self._record_buffers[dev_idx].append(mono * mic.gain)

        return callback

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

        peak = np.max(np.abs(mixed))
        if peak > 0:
            mixed = mixed / peak * 0.95

        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.recordings_dir, f"mixed_{ts}.wav")
        sf.write(output_path, mixed, SAMPLE_RATE)
        return output_path

    def decay_peaks(self):
        for mic in self.mics.values():
            mic.peak = max(mic.peak * 0.95, mic.level)
