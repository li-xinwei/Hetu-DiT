FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/hetudit \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/Hetu-DiT

RUN groupadd --system --gid 10001 hetudit && \
    useradd --system --uid 10001 --gid 10001 --create-home --home-dir /home/hetudit --shell /usr/sbin/nologin hetudit

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=10001:10001 . /workspace/Hetu-DiT

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -e . && \
    mkdir -p /tmp/ray && \
    chown -R 10001:10001 /tmp/ray

USER 10001:10001

EXPOSE 8000

CMD ["python3", "-m", "hetu_dit.entrypoint.api_server", "--host", "0.0.0.0", "--port", "8000"]
