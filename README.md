
![logo](https://github.com/probcomp/bayes3d/assets/66085644/bf4e3d42-2d70-40fa-b980-04bd4e18bf2b)

**[Documentation](https://probcomp.github.io/bayes3d/bayes3d/)**

## Installation
### Python Environment
```
git clone https://github.com/probcomp/bayes3d.git
conda create -n bayes3d python=3.9
conda activate bayes3d
pip install poetry
poetry install
```
### JAX Installation
```
pip install --upgrade pip

# CUDA 12 installation
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# CUDA 11 installation
pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
### Download Assets
```
bash scripts/download.sh
```
