# FAIR Universe Weak Lensing ML Challenge
# - Ubuntu 24.04
# - Image architecture: linux/amd64
# - Python: 3.12
# - CUDA base image: 12.6.3
# - PyTorch: 2.13.0+cu126
# - TensorFlow: 2.20.0

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python 3.12 and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3-pip \
        build-essential \
        ca-certificates \
        libgomp1 \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3.12 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip and build tools
RUN python -m pip install --upgrade \
    pip setuptools wheel

# Install PyTorch with CUDA 12.6
RUN python -m pip install \
    torch==2.13.0 \
    torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Install TensorFlow and GPU dependencies
# IMPORTANT: GPU runtime compatibility must be tested
RUN python -m pip install \
    "tensorflow[and-cuda]==2.20.0"

# Scientific computing and data processing
RUN python -m pip install \
    "numpy==2.5.3" \
    pandas scipy pyarrow \
    h5py zarr \
    scikit-learn scikit-image \
    numba joblib tqdm pyyaml

# ML and deep learning utilities
RUN python -m pip install \
    xgboost lightgbm \
    nflows einops timm torchmetrics

# Statistical and cosmological inference
RUN python -m pip install \
    iminuit emcee corner \
    "sbi==0.27.0" \
    astropy

# Visualization and numerical analysis
RUN python -m pip install \
    matplotlib seaborn PyWavelets

# Verify Python version
RUN python --version && \
    python -m pip --version

# Check dependency conflicts
RUN python -m pip check

# Basic import tests (CPU/build-time)
RUN python -c "import numpy, scipy, pandas, sklearn"
RUN python -c "import torch, torchvision"
RUN python -c "import tensorflow as tf; print(tf.__version__)"
RUN python -c "import nflows, sbi, timm, torchmetrics"

# Default command
CMD ["python"]


# To build image from Dockerfile (linux/amd64 for NVIDIA GPUs)
# docker build --platform linux/amd64 -t fair_universe:test .

# To run the image interactively
# docker run --rm -it --platform linux/amd64 fair_universe:test

# To check Python version and package dependencies
# docker run --rm --platform linux/amd64 fair_universe:test python --version
# docker run --rm --platform linux/amd64 fair_universe:test python -m pip check

# To save installed Python package versions
# docker run --rm --platform linux/amd64 fair_universe:test python -m pip freeze > requirements-lock.txt

# To tag the image for Docker Hub
# docker tag fair_universe:test ayaga3540/fair_universe:v1.0

# To log in to Docker Hub
# docker login

# To push the image to Docker Hub
# docker push ayaga3540/fair_universe:v1.0

# To pull the image from Docker Hub
# docker pull --platform linux/amd64 ayaga3540/fair_universe:v1.0