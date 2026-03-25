#!/usr/bin/with-contenv bashio
# s6 service script for Audio Mixer

bashio::log.info "Starting Audio Mixer..."

export RECORDINGS_DIR="${RECORDINGS_DIR:-/share/audiomixer}"
export WEB_PORT="${WEB_PORT:-8099}"

mkdir -p "${RECORDINGS_DIR}"

# Activate PulseAudio sources
bashio::log.info "Activating PulseAudio sources..."
sleep 2

for src in $(pactl list sources short 2>/dev/null | grep -v monitor | awk '{print $2}'); do
    pactl set-source-mute "$src" 0 2>/dev/null || true
    pactl set-source-volume "$src" 100% 2>/dev/null || true
    pactl suspend-source "$src" 1 2>/dev/null || true
    sleep 0.1
    pactl suspend-source "$src" 0 2>/dev/null || true
    bashio::log.info "  Activated: $src"
done

bashio::log.info "Starting Web UI on port ${WEB_PORT}..."
exec python3 /app/web_app.py
