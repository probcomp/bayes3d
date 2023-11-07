# Installing Bayes3D

### Fetch repository and setup Python environment.
```
git clone https://github.com/probcomp/bayes3d.git
cd bayes3d
conda create -n bayes3d python=3.9
conda activate bayes3d
pip install -r requirements.txt
pip install -e .
```

### Install GenJAX (optional)
```
pip install git+https://github.com/probcomp/genjax.git
```

### Install JAX
```
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Install Torch
```
pip install torch torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu118
```