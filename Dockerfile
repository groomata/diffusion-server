FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

WORKDIR /server

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y bash \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libsndfile1-dev \
    python3.10 \
    python3-pip \
    python3.10-venv && \
    rm -rf /var/lib/apt/lists

RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.2.2

ENV PATH $PATH:/root/.local/bin

COPY . .

RUN poetry install

ENTRYPOINT [ "poetry", "run", "uvicorn", "diffusion_backend.app:app", "--port", "8000", "--host", "0.0.0.0" ]
