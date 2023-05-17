import bayes3d as b
import jax.numpy as jnp
import jax

H,W = 150,150
intrinsics = b.Intrinsics(
    height=H,
    width=W,
    fx=200.0, fy=200.0,
    cx=W/2.0, cy=H/2.0,
    near=0.001, far=2.0
)
renderer = b.Renderer(intrinsics)



observed_xyz = j.t3d.unproject_depth(observed_depth, intrinsics)
rendered_xyz = j.t3d.unproject_depth(rendered_depth, intrinsics)
rendered_seg = jnp.ones((H,W))

r_array = jnp.linspace(0.01, 0.1,100)

variance = 0.1
outlier_prob = 0.1
outlier_volume = 1.0
filter_size = 3

print(j.threedp3_likelihood_jit(
    observed_xyz, rendered_xyz, 0.01, 0.01, 100.0, 1
))


num_mixture_components = observed_xyz.shape[0] * observed_xyz.shape[1]

rendered_xyz_padded = jax.lax.pad(rendered_xyz,  -100.0, ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,)))
jj, ii = jnp.meshgrid(jnp.arange(observed_xyz.shape[1]), jnp.arange(observed_xyz.shape[0]))
indices = jnp.stack([ii,jj],axis=-1)

ij = indices[100,50]


latent_points = jax.lax.dynamic_slice(rendered_xyz_padded, (ij[0], ij[1], 0), (2*filter_size + 1, 2*filter_size + 1, 3))

xyz = observed_xyz[ij[0], ij[1], :3]
distance = jnp.linalg.norm(xyz)
unit_vec = xyz / distance

mixture_centers = latent_points @ unit_vec
distances_to_ray = jnp.linalg.norm(latent_points - mixture_centers[...,None] * unit_vec, axis=-1)

weights = jax.scipy.stats.norm.pdf(
    distances_to_ray,
    loc=0.0,
    scale=0.01
)
print(weights)
weights = weights / weights.sum()
print(weights)


probability = jax.scipy.special.logsumexp(
    jax.scipy.stats.norm.logpdf(
        distance - mixture_centers,
        loc=0.0,
        scale=jnp.sqrt(variance)
    ).sum(-1) - jnp.log(num_mixture_components),
    b = weights
)


print(j.threedp3_likelihood_multi_r_jit(
    observed_xyz, rendered_xyz, rendered_seg, jnp.array([0.6, 0.6]), 0.2, 0.1, 3
))


from IPython import embed; embed()
