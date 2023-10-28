import bayes3d as b
import numpy as np
import jax.numpy as jnp
import jax
import pickle
import os

def save_metadata(metadata, name):
    if os.path.exists(f'{name}.pkl'):
        check = input(f"{name}.pkl already exists, do you want to overwrite? (y/n)")
        if 'n' in check.lower():
            raise FileExistsError(f"{name}.pkl already exists")
    with open(f'{name}.pkl', 'wb') as file:
        pickle.dump(metadata, file)

def load_metadata(name):
    if '.pkl' != name[-4:]:
        name += '.pkl' 
    with open(name, 'rb') as file:
        return pickle.load(file)