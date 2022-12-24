import torch
import torch.nn.functional
import functorch
import numpy as np

h,w, = 120,160
obs_xyz = torch.zeros((h,w,3), device='cuda')
rendered_xyz = torch.zeros((h,w,3), device='cuda')

filter_size = 3
r  = 0.1
outlier_prob = 0.5
obs_mask = obs_xyz[:,:,2] > 0.0
rendered_mask = rendered_xyz[:,:,2] > 0.0

rendered_xyz_padded = torch.nn.functional.pad(rendered_xyz,  (0,0, filter_size, filter_size, filter_size, filter_size), mode='constant',value=-100.0)
jj, ii = torch.meshgrid(torch.arange(obs_xyz.shape[1]), torch.arange(obs_xyz.shape[0]))

def count_ii_jj(idx, data_xyz, model_xyz, h, w, r):
    i = idx // w
    j = idx % h
    t = data_xyz[i,j, :3] - model_xyz[i:i + 2*filter_size + 1, j:j+2*filter_size + 1, :3]
    distance = torch.linalg.norm(t, axis=-1)
    return torch.sum(distance <= r)

idxs = torch.arange(h*w,dtype=torch.int32,device='cuda')
f = np.vectorize(lambda x: count_ii_jj(x, obs_xyz, rendered_xyz, h, w, r))
f(idxs)
from IPython import embed; embed()


from IPython import embed; embed()

indices = jnp.stack([ii,jj],axis=-1)
counts = count_ii_jj(indices, obs_xyz, rendered_xyz_padded, r, filter_size)
num_latent_points = rendered_mask.sum()
probs = outlier_prob  +  jnp.nan_to_num((1.0 - outlier_prob) / num_latent_points  * counts * 1.0 / (4/3 * jnp.pi * r**3))
log_probs = jnp.log(probs)