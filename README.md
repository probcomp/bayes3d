# Bayes3D


## Setup

Setup virtualenv and install dependencies:
```
conda create -n bayes3d python=3.8
conda activate bayes3d
pip install -r requirements.txt
python setup.py develop
```

To get assets:
```
bash download.sh
```

Add this to `~/.bashrc`
```
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Test

To test successful setup run:
```
python demo.py
```
and view `demo.gif`

![](assets/demo.gif)

## CosyPose Setup

```
git submodule update --init --recursive
cd jax3dp3/cosypose_baseline
bash cosypose_setup.sh
```

To test setup, run `test/test_cosypose.py`


## Instance Setup

### Installing Cuda 11.7

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```


### Installing cuDNN
In order to download cuDNN libraries, you need to go to https://developer.nvidia.com/cudnn and click on the Download cuDNN button. The webpage will ask you to login into the NVIDIA developer account. After logging in and accepting their terms and conditions, you should click on the following three links:

sudo apt-get install libglu-dev

sudo apt-get remove libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev

## MuJoCo (Physics Rendering) Setup
To use the MuJoCo physics engine + renderer extension, `pip install dm_control` (already specified in `requirements.txt`). For headless rendering, install the OpenGL Extension Wrapper
```
sudo apt-get install libglew2.1 
```
This is for Ubuntu 20.04 LTS. Check https://pkgs.org/search/?q=libglew for other OSes
