![Untitled-2](https://github.com/probcomp/bayes3d/assets/66085644/b99496a7-8efd-42a4-9550-fad0f8ff596d)

**[Documentation](https://probcomp.github.io/bayes3d/bayes3d/)**

# Installation

```
git clone https://github.com/probcomp/bayes3d.git
conda create -n bayes3d python=3.9
conda activate bayes3d
pip install -r requirements.txt
pip install -e assets/genjax
pip install -e .
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
