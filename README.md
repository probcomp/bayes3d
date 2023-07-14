# Bayes3D

[![](https://img.shields.io/badge/docs-stable-blue.svg?style=flat-square)](https://probcomp.github.io/bayes3d/bayes3d/)


## Setup Instructions

```
git clone https://github.com/probcomp/bayes3d.git
python3 -m venv venv
pip install -r requirements.txt
pip install -e assets/genjax
```

Install JAX:
```
pip install --upgrade pip

# CUDA 12 installation
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# CUDA 11 installation
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Download assets:
```
bash scripts/download.sh
```

## GCP Setup Instructions

Get images `tf-2-12-gpu-debian-11-py310`, on NVIDIA V100

sudo /opt/deeplearning/install-driver.sh


# Helpful commands

```
# Check CuDNN version
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```
