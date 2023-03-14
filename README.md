# Jax3DP3


## Setup

Setup virtualenv and install dependencies:
```
conda create -n jax python=3.8
conda activate jax
pip install -r requirements.txt

# Installs the wheel compatible with Cuda >= 11.4 and cudnn >= 8.2
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Installs the wheel compatible with Cuda >= 11.1 and cudnn >= 8.0.5
pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

python setup.py develop
```

Using a *Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04) 20230309* instance of AWS comes with CUDA 11.7 and the proper CuDNN version. 

Use these commands to identify Cuda and CuDNN versions:
```
nvcc --version
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

Add the following to `~/.bashrc`
```
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Install additional libraries and restart to apply changes
```
sudo apt-get install libegl1-mesa-dev
sudo apt-get install libglu1-mesa-dev
```

## Test

To test successful setup run:
```
python test/test.py
```
and view `test.gif`

## Assets

Get additional model and data assets by running
```
pip install -r requirements.txt
bash download.sh
```



```
sudo apt install libeigen3-dev
sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
