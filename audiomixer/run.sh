#!/bin/sh
set -e

echo "=== Audio Mixer Add-on ==="
echo "Recordings dir: ${RECORDINGS_DIR:-/share/audiomixer}"
mkdir -p "${RECORDINGS_DIR:-/share/audiomixer}"

# Activate PulseAudio sources (unmute + set volume)
echo "Activating PulseAudio sources..."
sleep 2  # wait for PA to be ready

for src in $(pactl list sources short 2>/dev/null | grep -v monitor | awk '{print $2}'); do
    pactl set-source-mute "$src" 0 2>/dev/null || true
    pactl set-source-volume "$src" 100% 2>/dev/null || true
    pactl suspend-source "$src" 1 2>/dev/null || true
    sleep 0.1
    pactl suspend-source "$src" 0 2>/dev/null || true
    echo "  Activated: $src"
done

echo ""
echo "Starting Web UI on port ${WEB_PORT:-8099}..."
exec python3 /app/web_app.py
