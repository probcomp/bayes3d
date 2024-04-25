![logo][logo]

<div align="center">

[![License][license]][license-url]
![Stability: Experimental][experimental-badge]

</div>

- **[Documentation](https://probcomp.github.io/bayes3d/)**
- **[Installation](#installation-guide)**
- **[Google Cloud Setup](#running-on-a-google-cloud-instance)**

# Installation Guide

These instructions require a GPU. To run on a cloud instance, follow [Google Cloud Setup](#running-on-a-google-cloud-instance) and then return here.

First, set up a fresh Python environment:

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
- run `gcloud init` as described [in this
  guide](https://cloud.google.com/sdk/docs/initializing) and configure the tool
  with the `probcomp-caliban` project ID.

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
# Running on a Google Cloud Instance

Bayes3D has high compute/GPU requirements, so working in a Cloud VM is a great option. After you've set up a [Google Cloud Platform]([url](https://cloud.google.com)) account, you can follow these instructions to get up and running.

## Configuring the `gcloud` CLI

Install the [Google Cloud command line
tools](https://cloud.google.com/sdk/docs/install).

- Follow the instructions on the [installation page](https://cloud.google.com/sdk/docs/install)
- run `gcloud init` as described [in this guide](https://cloud.google.com/sdk/docs/initializing) and configure the tool with the ID of a Cloud project.

## Launching a GPU-Configured VM

To launch a Cloud VM, run the following commands at your terminal:

```bash
export ZONE="us-west1-b"
export INSTANCE_NAME="bayes3d-template"

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family="common-gpu-debian-11-py310" \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=300GB \
  --machine-type n1-standard-8 \
  --accelerator="type=nvidia-tesla-v100,count=1" \
  --metadata="install-nvidia-driver=True" \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

- Make sure to customize `INSTANCE_NAME`, these are shared across the project / region.
- You may need to increase `--boot-disk-size`, but don't go lower.

Of course you can customize anything you like, but don't change the
`--image-project` or `--image-family` arguments.

## Accessing the VM

After a few minutes your VM will be available for access via SSH. You can reach
a terminal in a few different ways:

- Locate your image on the [Cloud
  Console](https://console.cloud.google.com/compute/instances) and click the
  "SSH" button
- Log in via the `gcloud` command line tool with the following command:

```bash
# These environment variables were set in the code block above:
gcloud compute ssh --zone $ZONE $INSTANCE_NAME
```

- Configure your `ssh` credentials so the normal `ssh` command works by running

```bash
gcloud compute config-ssh
```

Then you should be able to log in like:

```bash
ssh $INSTANCE_ID.$ZONE.$PROJECT_ID
```

The `gcloud compute config-ssh` command needs to be re-run after instances have been stopped/started, as they are often assigned new IP addresses.

## Port-forwarding from a VM

Configure your ssh credentials:

```bash
gcloud compute config-ssh
```

Then `ssh` into the instance using this command:

```bash
ssh $INSTANCE_ID.$ZONE.$PROJECT_ID -L <local_port>:localhost:<remote_port>
```

For example, to forward port 8888 on the VM to my local port 8888:

```
ssh my-image.us-west1-b.probcomp-caliban -L 8888:localhost:8888
```

---
## License

Distributed under the [Apache 2.0](LICENSE) license. See [LICENSE](LICENSE).

[experimental-badge]: https://img.shields.io/badge/stability-experimental-orange.svg
[license-url]: LICENSE
[license]: https://img.shields.io/badge/License-Apache_2.0-brightgreen.svg
[logo]: https://github.com/probcomp/bayes3d/assets/66085644/bf4e3d42-2d70-40fa-b980-04bd4e18bf2b
