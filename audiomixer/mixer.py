#!/usr/bin/env python3
"""
Shared AudioMixer with pluggable mixing modes.
Modes: mix_all, dominant, dugan (more can be added).
"""

import glob
import os
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
    weight: float = 0.0  # mixing weight (0-1), shown in UI
    _zero_blocks: int = 0


# ── Mixing Modes ─────────────────────────────────────────────────

class MixMode(ABC):
    """Base class for mixing modes."""
    name: str = "base"
    label: str = "Base"

    def reset(self):
        """Called when mode is activated or mics change."""
        pass

    @abstractmethod
    def compute_weights(self, mics: dict[int, MicState]) -> dict[int, float]:
        """Return target weight (0-1) for each mic index.
        Called from audio callback — must be fast."""
        ...

    def update_ui_state(self, mics: dict[int, MicState], weights: dict[int, float]):
        """Update mic UI flags based on computed weights."""
        for idx, mic in mics.items():
            w = weights.get(idx, 0)
            mic.weight = w
            mic.below_gate = mic.smoothed_rms < NOISE_GATE_RMS and not mic.is_dead


class MixAllMode(MixMode):
    """Simple equal mix — all mics at equal weight."""
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
    """Single dominant mic with hysteresis and hold time."""
    name = "dominant"
    label = "Dominant"

    RELEASE_RMS = 0.008
    MIN_HOLD = 0.5
    SWITCH_RATIO = 1.5

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
            self._hold_until = now + self.MIN_HOLD
        else:
            current_rms = mics[self.dominant_idx].smoothed_rms
            loudest = max(active, key=active.get)
            loudest_rms = active[loudest]

            all_alive = {idx for idx, m in mics.items() if not m.is_dead}
            above_gate_ratio = len(active) / max(len(all_alive), 1)

            if above_gate_ratio >= 0.5 and loudest != self.dominant_idx:
                if loudest_rms > current_rms * 1.3:
                    self.dominant_idx = loudest
                    self._hold_until = now + self.MIN_HOLD
            else:
                dom_speaking = current_rms >= self.RELEASE_RMS
                if dom_speaking:
                    self._hold_until = now + self.MIN_HOLD
                elif now >= self._hold_until:
                    if loudest != self.dominant_idx and loudest_rms > current_rms * self.SWITCH_RATIO:
                        self.dominant_idx = loudest
                        self._hold_until = now + self.MIN_HOLD
                    elif current_rms < NOISE_GATE_RMS:
                        self.dominant_idx = loudest if active else None

        weights = {}
        for idx in mics:
            if self.dominant_idx is None:
                weights[idx] = 0.0
            elif idx == self.dominant_idx:
                weights[idx] = 1.0
            else:
                weights[idx] = 0.0
        return weights

    def update_ui_state(self, mics, weights):
        super().update_ui_state(mics, weights)
        for idx, mic in mics.items():
            below = mic.smoothed_rms < NOISE_GATE_RMS
            mic.is_dominant = (idx == self.dominant_idx) and not below and not mic.is_dead
            mic.is_muted = (idx != self.dominant_idx) and not below and not mic.is_dead


class DuganMode(MixMode):
    """Dugan automixing — gain proportional to energy.

    weight_i = rms_i^n / Σ(rms_j^n)

    Higher exponent = sharper focus on loudest mic.
    n=2: gentle (original Dugan), n=6: sharp (reduces phase between mics).
    """
    name = "dugan"
    label = "Dugan Automix"

    # Exponent controls focus sharpness:
    # 2 = classic Dugan (gentle), 4 = moderate, 6 = sharp, 8 = very sharp
    EXPONENT = 6

    def compute_weights(self, mics):
        weights = {}
        alive = {idx: m for idx, m in mics.items() if not m.is_dead}

        if not alive:
            return {idx: 0.0 for idx in mics}

        energies = {}
        for idx, m in alive.items():
            rms = m.smoothed_rms
            if rms < NOISE_GATE_RMS:
                energies[idx] = 0.0
            else:
                energies[idx] = rms ** self.EXPONENT

        total_energy = sum(energies.values())

        if total_energy < 1e-30:
            w = 1.0 / len(alive)
            for idx in mics:
                weights[idx] = w if idx in alive else 0.0
        else:
            for idx in mics:
                if idx in alive:
                    weights[idx] = energies[idx] / total_energy
                else:
                    weights[idx] = 0.0

        return weights

    def update_ui_state(self, mics, weights):
        super().update_ui_state(mics, weights)
        # In Dugan, dominant = highest weight, muted = weight < 10%
        if not weights:
            return
        max_w = max(weights.values()) if weights else 0
        for idx, mic in mics.items():
            w = weights.get(idx, 0)
            mic.is_dominant = (w == max_w and w > 0.3) and not mic.is_dead
            mic.is_muted = (w < 0.1) and not mic.is_dead and not mic.below_gate


class DuganSharpMode(DuganMode):
    """Dugan with very high exponent — nearly single-mic selection
    but with smooth transitions instead of hard switching."""
    name = "dugan_sharp"
    label = "Dugan Sharp"
    EXPONENT = 12


# ── Mode registry ────────────────────────────────────────────────

MIXING_MODES: dict[str, type[MixMode]] = {
    "mix_all": MixAllMode,
    "dominant": DominantMode,
    "dugan": DuganMode,
    "dugan_sharp": DuganSharpMode,
}

GAIN_SMOOTHING = 0.15  # how fast gain transitions (per block)


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

        # Mixing mode
        self._modes: dict[str, MixMode] = {
            name: cls() for name, cls in MIXING_MODES.items()
        }
        self._current_mode_name = "dugan"  # default
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
        # Reset mic UI state
        for mic in self.mics.values():
            mic.is_dominant = False
            mic.is_muted = False
            mic.gain = 0.0
            mic.weight = 0.0
        self._notify("mode_changed", {"mode": mode_name})

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
                "level": mic.level,
                "peak": mic.peak,
                "smoothed_rms": mic.smoothed_rms,
                "is_dominant": mic.is_dominant,
                "is_muted": mic.is_muted,
                "is_dead": mic.is_dead,
                "below_gate": mic.below_gate,
                "weight": round(mic.weight, 3),
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

            # Dead mic detection
            if rms < 1e-10:
                mic._zero_blocks += 1
            else:
                mic._zero_blocks = 0
            mic.is_dead = mic._zero_blocks >= DEAD_MIC_BLOCKS

            # Smooth RMS (EMA: fast attack, slow release)
            alpha = 0.3 if rms > mic.smoothed_rms else 0.05
            mic.smoothed_rms = alpha * rms + (1 - alpha) * mic.smoothed_rms

            if scaled > mic.peak:
                mic.peak = scaled

            # Compute mixing weights via current mode
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

        # Normalize (weights already sum to ~1, but normalize for safety)
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
