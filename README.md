
![logo](https://github.com/probcomp/bayes3d/assets/66085644/e79f1773-b56d-4329-b418-fe3005fe216b)

**[Documentation](https://probcomp.github.io/bayes3d/bayes3d/)**

## Installation
### Python Environment
```
git clone https://github.com/probcomp/bayes3d.git
conda create -n bayes3d python=3.9
conda activate bayes3d
pip install -r requirements.txt
pip install -e assets/genjax
pip install -e .
```
### JAX Installation
```
pip install --upgrade pip

# CUDA 12 installation
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# CUDA 11 installation
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
### Download Assets
```
bash scripts/download.sh
```
