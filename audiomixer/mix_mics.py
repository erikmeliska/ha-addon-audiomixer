#!/usr/bin/env python3
"""
Mix multiple USB microphones into a single WAV file.
Works on Mac (native) and Raspberry Pi (Docker with /dev/snd passthrough).

Usage:
  python mix_mics.py                      # auto-detect USB mics, record 30s
  python mix_mics.py --duration 60        # record 60 seconds
  python mix_mics.py --list               # list available audio devices
  python mix_mics.py --devices 1 3 5 7    # use specific device indices
"""

import argparse
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
BLOCKSIZE = 1024


def list_devices():
    """Print all available audio input devices."""
    print("\nAvailable input devices:\n")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = "  <-- USB" if "usb" in dev["name"].lower() else ""
            print(f"  [{i}] {dev['name']} (inputs: {dev['max_input_channels']}){marker}")
    print()


def find_usb_mics(count=4):
    """Auto-detect USB microphones by name."""
    devices = sd.query_devices()
    usb_indices = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and "usb" in dev["name"].lower():
            usb_indices.append(i)
    if len(usb_indices) < count:
        print(f"Warning: found only {len(usb_indices)} USB mic(s), expected {count}.")
        print("Use --list to see devices, --devices to pick manually.")
        if not usb_indices:
            sys.exit(1)
    return usb_indices[:count]


def record_and_mix(device_indices, duration, output_file):
    """Open one input stream per mic, mix into a single buffer, save to WAV."""
    total_frames = int(SAMPLE_RATE * duration)
    mixed = np.zeros(total_frames, dtype=np.float32)
    locks = [threading.Lock() for _ in device_indices]
    buffers = [np.zeros(total_frames, dtype=np.float32) for _ in device_indices]
    positions = [0] * len(device_indices)
    errors = []

    def make_callback(idx):
        def callback(indata, frames, time_info, status):
            if status:
                errors.append(f"Device {device_indices[idx]}: {status}")
            mono = indata[:, 0] if indata.shape[1] > 1 else indata.flatten()
            with locks[idx]:
                end = min(positions[idx] + len(mono), total_frames)
                count = end - positions[idx]
                buffers[idx][positions[idx]:end] = mono[:count]
                positions[idx] = end
        return callback

    streams = []
    for i, dev_idx in enumerate(device_indices):
        dev_name = sd.query_devices(dev_idx)["name"]
        print(f"  Mic {i+1}: [{dev_idx}] {dev_name}")
        s = sd.InputStream(
            device=dev_idx,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCKSIZE,
            callback=make_callback(i),
        )
        streams.append(s)

    print(f"\nRecording {duration}s from {len(streams)} mic(s)...")
    for s in streams:
        s.start()

    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        print("\nStopped early.")

    for s in streams:
        s.stop()
        s.close()

    if errors:
        print(f"\nAudio warnings: {len(errors)} (first: {errors[0]})")

    # Mix: average all buffers
    for buf in buffers:
        mixed += buf
    mixed /= len(buffers)

    # Normalize to avoid clipping
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak * 0.95

    sf.write(output_file, mixed, SAMPLE_RATE)
    print(f"Saved to {output_file} ({duration}s, {SAMPLE_RATE} Hz, mono)")


def main():
    parser = argparse.ArgumentParser(description="Mix USB microphones into one WAV file")
    parser.add_argument("--list", action="store_true", help="List audio devices and exit")
    parser.add_argument("--devices", type=int, nargs="+", help="Device indices to use (default: auto-detect USB)")
    parser.add_argument("--duration", type=float, default=30, help="Recording duration in seconds (default: 30)")
    parser.add_argument("--output", type=str, default="mixed_output.wav", help="Output WAV file (default: mixed_output.wav)")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    print("Audio device backend:", sd.get_portaudio_version()[1])

    if args.devices:
        device_indices = args.devices
    else:
        print("Auto-detecting USB microphones...")
        device_indices = find_usb_mics(count=4)

    print(f"\nUsing {len(device_indices)} device(s):")
    record_and_mix(device_indices, args.duration, args.output)


if __name__ == "__main__":
    main()
