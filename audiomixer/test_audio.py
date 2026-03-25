#!/usr/bin/env python3
"""Test: diagnose PulseAudio sources and try to unmute/activate them."""
import subprocess
import time
import numpy as np
import soundfile as sf

RATE = 16000
DURATION = 3

def run_cmd(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout + r.stderr

# Full source info
print("=== PulseAudio source details ===")
print(run_cmd(["pactl", "list", "sources"]))

# Get source names (skip monitors)
r = subprocess.run(["pactl", "list", "sources", "short"], capture_output=True, text=True)
sources = []
for line in r.stdout.strip().split("\n"):
    if not line.strip():
        continue
    parts = line.split("\t")
    if len(parts) >= 2 and ".monitor" not in parts[1]:
        sources.append(parts[1])

print(f"\n=== Found {len(sources)} capture sources ===")
for s in sources:
    print(f"  {s}")

# Unmute and set volume on all sources
print("\n=== Unmuting and setting volume ===")
for s in sources:
    print(run_cmd(["pactl", "set-source-mute", s, "0"]))
    print(run_cmd(["pactl", "set-source-volume", s, "100%"]))
    print(f"  {s}: unmuted, volume 100%")

# Suspend/resume to activate
print("\n=== Suspending and resuming sources ===")
for s in sources:
    run_cmd(["pactl", "suspend-source", s, "1"])
    time.sleep(0.1)
    run_cmd(["pactl", "suspend-source", s, "0"])
    print(f"  {s}: resumed")

time.sleep(0.5)

# Now try recording
print(f"\n=== Recording {DURATION}s from each source ===")
for source in sources:
    print(f"\n--- {source} ---")
    proc = subprocess.Popen(
        ["parec", f"--device={source}", "--channels=1",
         f"--rate={RATE}", "--format=s16le"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    chunks = []
    end_time = time.time() + DURATION
    while time.time() < end_time:
        chunk = proc.stdout.read(4096)
        if chunk:
            chunks.append(chunk)

    proc.terminate()
    proc.wait()

    raw = b"".join(chunks)
    if len(raw) < 2:
        print(f"  No data. stderr: {proc.stderr.read().decode()}")
        continue

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    samples = len(data)
    peak = np.max(np.abs(data))
    rms = np.sqrt(np.mean(data ** 2))
    print(f"  Samples: {samples} ({samples/RATE:.2f}s)")
    print(f"  Peak: {peak:.6f}, RMS: {rms:.6f}")

    if peak > 0.001:
        sf.write(f"/tmp/test_{sources.index(source)}.wav", data, RATE)
        print(f"  SUCCESS!")
    else:
        # Check raw bytes for any non-zero
        nonzero = sum(1 for b in raw if b != 0)
        print(f"  Silent. Non-zero bytes: {nonzero}/{len(raw)}")

print("\nDone.")
