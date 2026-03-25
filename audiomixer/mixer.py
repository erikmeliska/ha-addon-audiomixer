#!/usr/bin/env python3
"""
Shared AudioMixer with pluggable mixing modes.
Modes: mix_all, dominant, dugan, dugan_sharp, dugan_aligned (phase-aligned).
"""

import collections
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

# Phase alignment
MAX_DELAY_SAMPLES = 48  # ~3ms at 16kHz, covers ~1m distance
CORRELATION_WINDOW = 2048  # samples used for cross-correlation
DELAY_UPDATE_INTERVAL = 0.1  # seconds between delay re-estimation


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
    weight: float = 0.0
    delay: int = 0  # estimated delay in samples vs reference mic
    _zero_blocks: int = 0


# ── Mixing Modes ─────────────────────────────────────────────────

class MixMode(ABC):
    name: str = "base"
    label: str = "Base"
    needs_raw_buffers: bool = False  # if True, mixer stores raw audio for post-processing

    def reset(self):
        pass

    @abstractmethod
    def compute_weights(self, mics: dict[int, MicState]) -> dict[int, float]:
        ...

    def update_ui_state(self, mics: dict[int, MicState], weights: dict[int, float]):
        for idx, mic in mics.items():
            w = weights.get(idx, 0)
            mic.weight = w
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
                if current_rms >= self.RELEASE_RMS:
                    self._hold_until = now + self.MIN_HOLD
                elif now >= self._hold_until:
                    if loudest != self.dominant_idx and loudest_rms > current_rms * self.SWITCH_RATIO:
                        self.dominant_idx = loudest
                        self._hold_until = now + self.MIN_HOLD
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
    """Dugan automixing: weight_i = rms_i^n / Σ(rms_j^n)"""
    name = "dugan"
    label = "Dugan"
    EXPONENT = 6

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

        return {idx: (energies.get(idx, 0.0) / total if idx in alive else 0.0) for idx in mics}

    def update_ui_state(self, mics, weights):
        super().update_ui_state(mics, weights)
        if not weights:
            return
        max_w = max(weights.values())
        for idx, mic in mics.items():
            w = weights.get(idx, 0)
            mic.is_dominant = (w == max_w and w > 0.3) and not mic.is_dead
            mic.is_muted = (w < 0.1) and not mic.is_dead and not mic.below_gate


class DuganSharpMode(DuganMode):
    name = "dugan_sharp"
    label = "Dugan Sharp"
    EXPONENT = 12


class DuganAlignedMode(DuganMode):
    """Dugan + phase alignment via cross-correlation.

    Before mixing, estimates delay between each mic and a reference mic,
    then shifts signals to align them. Eliminates comb filtering
    when multiple mics pick up the same source.
    """
    name = "dugan_aligned"
    label = "Dugan + Phase Align"
    EXPONENT = 6
    needs_raw_buffers = True

    def __init__(self):
        super().__init__()
        # Ring buffers for cross-correlation (per mic)
        self._ring_buffers: dict[int, collections.deque] = {}
        self._delays: dict[int, int] = {}  # samples delay per mic
        self._ref_idx: int | None = None
        self._last_delay_update: float = 0

    def reset(self):
        self._ring_buffers.clear()
        self._delays.clear()
        self._ref_idx = None
        self._last_delay_update = 0

    def feed_audio(self, dev_idx: int, samples: np.ndarray):
        """Feed raw audio into ring buffer for correlation."""
        if dev_idx not in self._ring_buffers:
            self._ring_buffers[dev_idx] = collections.deque(maxlen=CORRELATION_WINDOW)
        self._ring_buffers[dev_idx].extend(samples.tolist())

    def update_delays(self, mics: dict[int, MicState]):
        """Re-estimate delays between mics using cross-correlation."""
        now = time.monotonic()
        if now - self._last_delay_update < DELAY_UPDATE_INTERVAL:
            return
        self._last_delay_update = now

        # Pick reference mic: loudest alive mic
        alive = {idx: m for idx, m in mics.items()
                 if not m.is_dead and m.smoothed_rms >= NOISE_GATE_RMS}
        if not alive:
            return

        self._ref_idx = max(alive, key=lambda idx: alive[idx].smoothed_rms)

        ref_buf = self._ring_buffers.get(self._ref_idx)
        if ref_buf is None or len(ref_buf) < CORRELATION_WINDOW // 2:
            return

        ref = np.array(ref_buf, dtype=np.float32)

        for idx in mics:
            if idx == self._ref_idx:
                self._delays[idx] = 0
                continue

            buf = self._ring_buffers.get(idx)
            if buf is None or len(buf) < CORRELATION_WINDOW // 2:
                self._delays[idx] = 0
                continue

            other = np.array(buf, dtype=np.float32)

            # Use the shorter of the two
            n = min(len(ref), len(other))
            r = ref[-n:]
            o = other[-n:]

            # Cross-correlation via FFT (fast)
            # Only check delays up to MAX_DELAY_SAMPLES
            corr = np.correlate(
                r[MAX_DELAY_SAMPLES:-MAX_DELAY_SAMPLES] if n > 2 * MAX_DELAY_SAMPLES else r,
                o[:n],
                mode='full' if n <= 2 * MAX_DELAY_SAMPLES else 'same'
            )

            # Simpler: direct small-window correlation
            if n > 2 * MAX_DELAY_SAMPLES:
                best_delay = 0
                best_corr = 0
                for d in range(-MAX_DELAY_SAMPLES, MAX_DELAY_SAMPLES + 1):
                    if d >= 0:
                        c = np.sum(r[d:] * o[:n - d])
                    else:
                        c = np.sum(r[:n + d] * o[-d:])
                    if c > best_corr:
                        best_corr = c
                        best_delay = d
                self._delays[idx] = best_delay
            else:
                self._delays[idx] = 0

        # Update mic state
        for idx, mic in mics.items():
            mic.delay = self._delays.get(idx, 0)

    def get_delays(self) -> dict[int, int]:
        return dict(self._delays)


# ── Mode registry ────────────────────────────────────────────────

MIXING_MODES: dict[str, type[MixMode]] = {
    "mix_all": MixAllMode,
    "dominant": DominantMode,
    "dugan": DuganMode,
    "dugan_sharp": DuganSharpMode,
    "dugan_aligned": DuganAlignedMode,
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
        self._current_mode_name = "dugan_aligned"
        self._mode: MixMode = self._modes[self._current_mode_name]

        self._record_buffers: dict[int, list] = {}  # weighted audio
        self._raw_buffers: dict[int, list] = {}  # raw audio (for phase-aligned modes)
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
            mic.delay = 0
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
            self._raw_buffers[dev.index] = []

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
            self._raw_buffers[idx] = []
        self._notify("recording_started")

    def stop_recording(self) -> str | None:
        if not self.is_recording:
            return None
        self.is_recording = False
        elapsed = time.time() - self._record_start_time

        if self._mode.needs_raw_buffers:
            path = self._save_recording_aligned(elapsed)
        else:
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
                "delay": mic.delay,
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

            # Feed audio to phase-aligned mode's ring buffer
            if isinstance(self._mode, DuganAlignedMode):
                self._mode.feed_audio(dev_idx, mono)
                self._mode.update_delays(self.mics)

            # Compute mixing weights
            weights = self._mode.compute_weights(self.mics)
            self._mode.update_ui_state(self.mics, weights)

            # Recording
            if self.is_recording:
                if self._mode.needs_raw_buffers:
                    # Store raw audio — alignment happens at save time
                    self._raw_buffers[dev_idx].append(mono.copy())
                    # Also store weighted for fallback
                    target = weights.get(dev_idx, 0.0)
                    mic.gain += GAIN_SMOOTHING * (target - mic.gain)
                    self._record_buffers[dev_idx].append(mono * mic.gain)
                else:
                    target = weights.get(dev_idx, 0.0)
                    mic.gain += GAIN_SMOOTHING * (target - mic.gain)
                    self._record_buffers[dev_idx].append(mono * mic.gain)

        return callback

    def _save_recording(self, elapsed: float) -> str | None:
        """Save recording with pre-applied weights (non-aligned modes)."""
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

    def _save_recording_aligned(self, elapsed: float) -> str | None:
        """Save recording with block-by-block phase alignment + Dugan weights."""
        # Get raw audio per mic
        raw = {}
        max_len = 0
        for idx, chunks in self._raw_buffers.items():
            if chunks:
                arr = np.concatenate(chunks)
                raw[idx] = arr
                max_len = max(max_len, len(arr))

        if not raw or max_len == 0:
            return None

        # Pad all to same length
        for idx in raw:
            if len(raw[idx]) < max_len:
                raw[idx] = np.pad(raw[idx], (0, max_len - len(raw[idx])))

        # Process in blocks: align + weight
        block_size = BLOCKSIZE
        mixed = np.zeros(max_len, dtype=np.float32)
        mic_indices = list(raw.keys())

        for start in range(0, max_len - block_size, block_size):
            end = start + block_size
            blocks = {idx: raw[idx][start:end] for idx in mic_indices}

            # Find reference (highest RMS in this block)
            rms_vals = {idx: np.sqrt(np.mean(b ** 2)) for idx, b in blocks.items()}
            ref_idx = max(rms_vals, key=rms_vals.get)
            ref_block = blocks[ref_idx]

            # Compute Dugan weights for this block
            energies = {}
            for idx, r in rms_vals.items():
                if r < NOISE_GATE_RMS:
                    energies[idx] = 0.0
                else:
                    energies[idx] = r ** 6  # Dugan exponent
            total_e = sum(energies.values())

            if total_e < 1e-30:
                weights = {idx: 1.0 / len(mic_indices) for idx in mic_indices}
            else:
                weights = {idx: energies[idx] / total_e for idx in mic_indices}

            # Align each mic to reference and mix
            block_mixed = np.zeros(block_size, dtype=np.float32)
            for idx in mic_indices:
                b = blocks[idx]
                w = weights[idx]

                if idx == ref_idx or w < 0.01:
                    # Reference mic or negligible weight — no alignment needed
                    block_mixed += b * w
                    continue

                # Cross-correlate to find delay
                delay = self._find_delay(ref_block, b)

                # Apply delay shift
                if delay > 0:
                    aligned = np.zeros(block_size, dtype=np.float32)
                    aligned[delay:] = b[:block_size - delay]
                elif delay < 0:
                    aligned = np.zeros(block_size, dtype=np.float32)
                    aligned[:block_size + delay] = b[-delay:]
                else:
                    aligned = b

                block_mixed += aligned * w

            mixed[start:end] = block_mixed

        # Handle remaining samples
        remainder = max_len % block_size
        if remainder > 0:
            start = max_len - remainder
            for idx in mic_indices:
                w = 1.0 / len(mic_indices)
                mixed[start:] += raw[idx][start:] * w

        peak = np.max(np.abs(mixed))
        if peak > 0:
            mixed = mixed / peak * 0.95

        ts = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.recordings_dir, f"aligned_{ts}.wav")
        sf.write(output_path, mixed, SAMPLE_RATE)
        return output_path

    @staticmethod
    def _find_delay(ref: np.ndarray, other: np.ndarray) -> int:
        """Find delay of 'other' relative to 'ref' using cross-correlation.
        Returns positive value if other is delayed (needs shift forward)."""
        n = len(ref)
        max_d = min(MAX_DELAY_SAMPLES, n // 4)

        best_delay = 0
        best_corr = -1e30

        for d in range(-max_d, max_d + 1):
            if d >= 0:
                c = np.dot(ref[d:], other[:n - d])
            else:
                c = np.dot(ref[:n + d], other[-d:])
            if c > best_corr:
                best_corr = c
                best_delay = d

        return best_delay

    def decay_peaks(self):
        for mic in self.mics.values():
            mic.peak = max(mic.peak * 0.95, mic.level)
