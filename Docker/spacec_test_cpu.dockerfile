# Docker container for efficient iterative testing
# - build with
#   $ docker buildx build -t spacec_test_cpu -f Docker/spacec_test_cpu.dockerfile .
# - run with:
#   $ docker run spacec_test_cpu
#   or if iteratively updating tests:
#   $ docker run -v .:/app/spacec spacec_test_gpu
FROM debian:bullseye-slim

ARG DEBIAN_FRONTEND=noninteractive

# Update the package list and install dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential gcc \
    git \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install Miniforge
RUN apt-get update && \
    apt-get install -y wget && \
    wget "https://github.com/conda-forge/miniforge/releases/download/23.11.0-0/Miniforge3-23.11.0-0-Linux-x86_64.sh" && \
    bash Miniforge3-*.sh -b -p /miniforge && \
    rm Miniforge3-*.sh

# Add Miniforge to PATH
ENV PATH="/miniforge/bin:${PATH}"

# Init mamba
RUN mamba init

# Create a new conda environment with Python 3.10
RUN conda create -n spacec python=3.10 -y

# Install mamba
RUN conda install -n spacec -c conda-forge mamba -y

# Install libxml2
# RUN mamba install -n spacec -c conda-forge libxml2=2.13.5 -y

# Install other dependencies ()
RUN mamba install -n spacec -c conda-forge graphviz libvips openslide -y; \
    mamba install -n spacec -c conda-forge pytest pytest-cov -y

# Install gcc
# RUN apt-get update && \
#     apt-get install -y build-essential gcc && \
#     rm -rf /var/lib/apt/lists/*

# Copy SPACEc development code and install
ADD . /app/spacec
WORKDIR /app/spacec
RUN git init
RUN mamba run -n spacec pip install -e .

# Clean up mamba
RUN mamba clean --all -f -y && \
    rm -rf /root/.cache/pip

# Default command
CMD ["bash", "-c", "source /miniforge/etc/profile.d/conda.sh && conda activate spacec && pytest"]
