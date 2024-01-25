# Base container: CUDA 12.21, cuDNN 8.9.4, Python 3.10, PyTorch 2.1.0
ARG BASE_IMG=nvcr.io/nvidia/pytorch:23.08-py3
FROM ${BASE_IMG}

WORKDIR /workspace

# Install Bayes3d dependencies, including a local genjax install
COPY ./docker/requirements_docker.txt /workspace/requirements.txt
COPY ./genjax  /workspace/genjax
RUN pip install -r /workspace/requirements.txt
RUN pip install -r /workspace/genjax/requirements.txt
RUN pip install /workspace/genjax

# Install JAX (0.4.16) and OpenGL dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN apt-get install -y mesa-common-dev libegl1-mesa-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev

# Cleanup and prepare env variables for graphics
RUN rm -rf /workspace/requirements.txt
RUN rm -rf /workspace/genjax
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,display
