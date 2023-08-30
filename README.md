
![logo](https://github.com/probcomp/bayes3d/assets/66085644/bf4e3d42-2d70-40fa-b980-04bd4e18bf2b)

- **[Documentation](https://probcomp.github.io/bayes3d/bayes3d/)**
- **[Installation](#installation-guide)**
- **[Get Assets](#get-assets)**
- **[Google Cloud Instance Setup](#gcp-setup)**

# Installation Guide
## Setup python environment
```
git clone https://github.com/probcomp/bayes3d.git
conda create -n bayes3d python=3.9
conda activate bayes3d
pip install poetry
poetry install
```
## Install JAX
Check your CUDA version:
```
nvcc --version
```

For CUDA 11.x run:
```
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
For CUDA 12.x run:
```
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
## Test
Run `python demo.py` to test installation.

## Common issues

If you encounter any of the following:
```
fatal error: EGL/egl.h: No such file or directory
    #include <EGL/egl.h>

fatal error: GL/glu.h: No such file or directory
    #include <GL/glu.h>
```
run:
```
sudo apt-get install mesa-common-dev libegl1-mesa-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
```

If you encounter:
```
[F glutil.cpp:338] eglInitialize() failed
Aborted (core dumped)
```
Reinstall NVIDIA drivers.
```
sudo apt-get install nvidia-driver-XXX
```
You can check the right version of the NVIDIA drivers by running `nvidia-smi`.


# Get Assets

Download model and data assets:
```
bash scripts/download.sh
```

# GCP Setup
Start new VM instance (see [link](https://cloud.google.com/compute/docs/instances/create-start-instance))

From the VM instances page, searched for public image `c2-deeplearning-pytorch-2-0-gpu-v20230807-debian-11-py310`

SSH into instance and when prompted, install the NVIDIA drivers.

Follow [installation guide](#installation-guide).