#!/bin/sh
set -e

echo "=== Audio Mixer Add-on ==="

RECORDINGS_DIR="${RECORDINGS_DIR:-/share/audiomixer}"
WEB_PORT="${WEB_PORT:-8099}"
export RECORDINGS_DIR WEB_PORT

echo "Recordings dir: ${RECORDINGS_DIR}"
mkdir -p "${RECORDINGS_DIR}"

# Wait for PulseAudio to be ready
echo "Waiting for PulseAudio..."
for i in $(seq 1 10); do
    if pactl info >/dev/null 2>&1; then
        echo "PulseAudio connected."
        break
    fi
    sleep 1
done

# Activate PulseAudio sources (unmute + set volume)
echo "Activating PulseAudio sources..."
for src in $(pactl list sources short 2>/dev/null | grep -v monitor | awk '{print $2}'); do
    pactl set-source-mute "$src" 0 2>/dev/null || true
    pactl set-source-volume "$src" 100% 2>/dev/null || true
    pactl suspend-source "$src" 1 2>/dev/null || true
    sleep 0.1
    pactl suspend-source "$src" 0 2>/dev/null || true
    echo "  Activated: $src"
done

echo ""
echo "Starting Web UI on port ${WEB_PORT}..."
exec python3 /app/web_app.py
