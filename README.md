Requires python version greater than 3.7

```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Select jax install based on cuda and cudnn versions:
```
pip install --upgrade pip

# Installs the wheel compatible with Cuda >= 11.4 and cudnn >= 8.2
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Installs the wheel compatible with Cuda >= 11.1 and cudnn >= 8.0.5
pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Run:
```
python test.py
```

If it produces `out.gif` then installation is successful.