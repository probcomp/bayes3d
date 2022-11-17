# Jax3DP3

Setup virtualenv and install dependencies:
```
conda create -n jax python=3.8
conda activate jax
pip install -r requirements.txt
python setup.py develop
```

Install `jax` and `jaxlib`
```
# Installs the wheel compatible with Cuda >= 11.4 and cudnn >= 8.2
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Installs the wheel compatible with Cuda >= 11.1 and cudnn >= 8.0.5
pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Use these commands to identify Cuda and CuDNN versions:
```
nvcc --version
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

To test successful setup run:
```
python test/test.py
```
and view `out.gif`
