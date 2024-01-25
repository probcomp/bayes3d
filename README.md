![logo][logo]

<div align="center">

[![License][license]][license-url]
![Stability: Experimental][experimental-badge]

</div>

- **[Documentation](https://probcomp.github.io/bayes3d/)**
- **[Installation](#installation-guide)**
- **[Get Assets](#get-assets)**
- **[Google Cloud Instance Setup](#gcp-setup)**

# Installation Guide

Setup Python environment:
```
git clone https://github.com/probcomp/bayes3d.git
cd bayes3d
conda create -n bayes3d python=3.9
conda activate bayes3d
pip install -r requirements.txt
pip install -e .
```

Install GenJAX (optional):
```
pip install git+https://github.com/probcomp/genjax.git@v0.1.0
```

Install JAX and Torch:
```
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch torchvision --upgrade --index-url https://download.pytorch.org/whl/cu118
```

Download model and data assets:
```
bash download.sh
```


## Test
Run `python demo.py` to test installation setup.


## Common issues

Error:
```
fatal error: EGL/egl.h: No such file or directory
    #include <EGL/egl.h>

fatal error: GL/glu.h: No such file or directory
    #include <GL/glu.h>
```
Run:
```
sudo apt-get install mesa-common-dev libegl1-mesa-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
```

Error:
```
[F glutil.cpp:338] eglInitialize() failed
Aborted (core dumped)
```
Reinstall NVIDIA drivers with `sudo apt-get install nvidia-driver-XXX`. Check version of driver using `nvidia-smi`.

Error:
```
ImportError: libcupti.so.11.7: cannot open shared object file: No such file or directory
```
Run:
```
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

Error:
```
raise RuntimeError("Ninja is required to load C++ extensions")
```
Run:
```
sudo apt-get update
sudo apt-get install ninja-build
```

I did somethi!

To check your CUDA version:
```
nvcc --version
```


# GCP Setup
- Start new VM instance (see [link](https://cloud.google.com/compute/docs/instances/create-start-instance)). Select GPU - NVIDIA V100 and Machine Type 8vCPU 4 Core 30GB.

-From the VM instances page, searched for public image `c2-deeplearning-pytorch-2-0-gpu-v20230925-debian-11-py310`. Increase storage to 1000GB.

- Note that public image names get updated frequently, so it is possible you may not find the one mentioned above. To find the latest public image, go to the [public list](https://cloud.google.com/compute/docs/images#console), and look for an image as close to the one above (Debian 11, CUDA 11.8, Python 3.10, Pytorch 2.0 etc.).

- SSH into instance and when prompted, install the NVIDIA drivers.

- Follow [installation guide](#installation-guide).

## License

Distributed under the [Apache 2.0](LICENSE) license. See [LICENSE](LICENSE).

[experimental-badge]: https://img.shields.io/badge/stability-experimental-orange.svg
[license-url]: LICENSE
[license]: https://img.shields.io/badge/License-Apache_2.0-brightgreen.svg
[logo]: https://github.com/probcomp/bayes3d/assets/66085644/bf4e3d42-2d70-40fa-b980-04bd4e18bf2b
