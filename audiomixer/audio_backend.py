#!/usr/bin/env python3
"""
Audio backend abstraction.
- Mac/desktop: uses sounddevice (PortAudio)
- RPi/Docker with PulseAudio: uses parec subprocess
"""

import os
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024


@dataclass
class AudioDevice:
    index: int
    name: str
    source_name: str  # PulseAudio source name or sounddevice index


def is_pulseaudio_available():
    """Check if we can reach a PulseAudio server."""
    try:
        r = subprocess.run(
            ["pactl", "info"],
            capture_output=True, text=True, timeout=3,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def discover_devices_pulse():
    """Discover capture sources via PulseAudio."""
    r = subprocess.run(
        ["pactl", "list", "sources", "short"],
        capture_output=True, text=True, timeout=5,
    )
    devices = []
    idx = 0
    for line in r.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and ".monitor" not in parts[1]:
            name = parts[1]
            short = name.split(".")[-1] if "." in name else name
            devices.append(AudioDevice(index=idx, name=f"fifine Mic ({short})", source_name=name))
            idx += 1
    return devices


def discover_devices_sounddevice():
    """Discover input devices via sounddevice/PortAudio."""
    import sounddevice as sd
    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append(AudioDevice(index=i, name=dev["name"], source_name=str(i)))
    return devices


def activate_pulse_sources(devices):
    """Unmute, set volume, and resume PulseAudio sources."""
    for dev in devices:
        subprocess.run(["pactl", "set-source-mute", dev.source_name, "0"],
                       capture_output=True, timeout=3)
        subprocess.run(["pactl", "set-source-volume", dev.source_name, "100%"],
                       capture_output=True, timeout=3)
        subprocess.run(["pactl", "suspend-source", dev.source_name, "1"],
                       capture_output=True, timeout=3)
        time.sleep(0.05)
        subprocess.run(["pactl", "suspend-source", dev.source_name, "0"],
                       capture_output=True, timeout=3)


class PulseAudioStream:
    """Reads audio from a PulseAudio source via parec subprocess."""

    def __init__(self, source_name, callback, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
        self.source_name = source_name
        self.callback = callback
        self.samplerate = samplerate
        self.blocksize = blocksize
        self._proc = None
        self._thread = None
        self._running = False

    def start(self):
        self._running = True
        self._proc = subprocess.Popen(
            [
                "parec",
                f"--device={self.source_name}",
                "--channels=1",
                f"--rate={self.samplerate}",
                "--format=s16le",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        bytes_per_block = self.blocksize * 2  # s16le = 2 bytes per sample
        while self._running and self._proc.poll() is None:
            raw = self._proc.stdout.read(bytes_per_block)
            if not raw:
                break
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            # Reshape to match sounddevice callback format (frames, channels)
            indata = data.reshape(-1, 1)
            self.callback(indata, len(data), None, None)

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def close(self):
        self.stop()


class SoundDeviceStream:
    """Wraps sounddevice.InputStream."""

    def __init__(self, device_index, callback, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
        import sounddevice as sd
        self._stream = sd.InputStream(
            device=device_index,
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=callback,
        )

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()

    def close(self):
        self._stream.close()


class AudioBackend:
    """Unified audio backend that auto-selects PulseAudio or sounddevice."""

    def __init__(self):
        self.use_pulse = is_pulseaudio_available()
        self.devices = []

    def discover(self):
        if self.use_pulse:
            self.devices = discover_devices_pulse()
            if self.devices:
                activate_pulse_sources(self.devices)
        else:
            self.devices = discover_devices_sounddevice()
        return self.devices

    def open_stream(self, device, callback):
        if self.use_pulse:
            return PulseAudioStream(device.source_name, callback)
        else:
            return SoundDeviceStream(int(device.source_name), callback)
