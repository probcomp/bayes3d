import bayes3d as b
import jax.numpy as jnp
import numpy as np
import jax
import os
import matplotlib.pyplot as plt
import matplotlib
import trimesh
from tqdm import tqdm
from collections import namedtuple
import cv2


# MASTER_SCALE = 0.2 #1 - this needs to be downscaled
MASTER_SCALE = 0.15

