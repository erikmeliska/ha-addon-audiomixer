ARG BUILD_FROM
FROM ${BUILD_FROM}

RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy \
    portaudio-dev \
    libsndfile \
    alsa-utils \
    alsa-lib \
    alsa-lib-dev \
    alsa-plugins-pulse \
    pulseaudio-utils \
    gcc \
    python3-dev \
    musl-dev

# PulseAudio client config
RUN mkdir -p /etc/pulse && \
    echo "default-server = unix:/run/pulse/native" > /etc/pulse/client.conf && \
    echo "autospawn = no" >> /etc/pulse/client.conf && \
    echo "daemon-binary = /bin/true" >> /etc/pulse/client.conf

ENV RECORDINGS_DIR=/share/audiomixer
ENV WEB_PORT=8099

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY audio_backend.py mixer.py web_app.py tui.py mix_mics.py deepgram_streamer.py remote_session.py ./
COPY static/ ./static/

COPY run.sh /
RUN chmod +x /run.sh

# Override the base image entrypoint — skip s6-overlay entirely
ENTRYPOINT []
CMD ["/run.sh"]
