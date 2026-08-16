# Pipeline + API image.
#
# GPU vs CPU. The roadmap specifies a CUDA base image. CUDA is Linux/NVIDIA
# only, so a hard-coded CUDA base cannot be built or run on the Apple Silicon
# machine this was developed on -- it would be an untested file shipped as if
# it worked. The base image is therefore a build argument: the default is a
# portable CPU image that actually builds and runs here, and a CUDA build is
# one flag away on a GPU host:
#
#   docker build -t traffic-vrag .                       # CPU (default, tested)
#   docker build -t traffic-vrag \                       # CUDA (untested here)
#     --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
#     --build-arg INSTALL_PYTHON=1 .
#
# On a CUDA build you must also install a CUDA-enabled torch wheel; the
# default requirements pull the CPU build. See DOCKER.md.
ARG BASE_IMAGE=python:3.9-slim
FROM ${BASE_IMAGE}

# The CUDA images ship no Python, so it is installed only when asked for.
ARG INSTALL_PYTHON=0
ENV DEBIAN_FRONTEND=noninteractive
RUN if [ "$INSTALL_PYTHON" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip && \
      ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
      ln -sf /usr/bin/pip3 /usr/local/bin/pip && \
      rm -rf /var/lib/apt/lists/*; \
    fi

# OpenCV needs libGL/libglib even in headless builds; git is needed by some
# model downloads.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a source edit does not re-download ~2GB of wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

# Model weights (YOLO ~40MB, Florence-2 ~1GB, OSNet) download on first use.
# Mounting these as volumes in compose keeps them across container rebuilds --
# without that, every rebuild re-downloads over a gigabyte.
ENV HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    YOLO_CONFIG_DIR=/models/ultralytics \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["python", "-m", "src.cli", "serve"]
