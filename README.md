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

Set up a fresh Python environment:

```bash
conda create -n bayes3d python=3.9
conda activate bayes3d
```

Install compatible versions JAX and Torch:

```bash
pip install --upgrade torch==2.2.0 torchvision==0.17.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade jax[cuda11_local]==0.4.20 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Bayes3D is built on top of GenJAX, which is currently hosted in a private Python
package repository. To configure your machine to access GenJAX:

- [File an issue](https://github.com/probcomp/bayes3d/issues/new) asking @sritchie to give you access.
- [Install the Google Cloud command line tools](https://cloud.google.com/sdk/docs/install).
- Follow the instructions on the [installation page](https://cloud.google.com/sdk/docs/install)
- run `gcloud auth application-default login` as described [in this
  guide](https://cloud.google.com/sdk/docs/initializing).

Then run the following command to configure `pip` to use these new gcloud
commands:

```bash
pip install keyring keyrings.google-artifactregistry-auth
```

Finally, install Bayes3D:

```bash
pip install bayes3d --extra-index-url https://us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple/
```

Download model and data assets:

```bash
wget -q -O - https://raw.githubusercontent.com/probcomp/bayes3d/main/download.sh | bash
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

To check your CUDA version:
```
nvcc --version
```


# GCP Setup

- Start new VM instance (see
  [link](https://cloud.google.com/compute/docs/instances/create-start-instance)).
  Select GPU - NVIDIA V100 and Machine Type 8vCPU 4 Core 30GB.

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
