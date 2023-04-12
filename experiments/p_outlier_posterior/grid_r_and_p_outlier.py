import jax3dp3 as j
import os
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np

intrinsics = j.Intrinsics(
    height=150,
    width=150,
    fx=200.0, fy=200.0,
    cx=75.0, cy=75.0,
    near=1.0, far=1000.0
)

plane = j.mesh.make_cuboid_mesh([1000.0, 1000.0, 0.001])
wall_pose = j.t3d.transform_from_pos(jnp.array([0.0, 0.0, 800.0]))

model_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv/models")




renderer = j.Renderer(intrinsics)
renderer.add_mesh(plane)
model_names = j.ycb_loader.MODEL_NAMES
for IDX in range(len(model_names)):
    mesh_path_ply = os.path.join(model_dir,"obj_" + "{}".format(IDX+1).rjust(6, '0') + ".ply")
    mesh = j.mesh.load_mesh(mesh_path_ply)
    renderer.add_mesh(mesh)

object_pose = j.distributions.gaussian_vmf_sample(
    jax.random.PRNGKey(2),
    j.t3d.transform_from_pos(
        jnp.array([0.0, 0.0, 300.0])
    ),
    0.01, 0.1
)

GT_ID = 4
observed_image = renderer.render_multiobject(
    jnp.array([object_pose, wall_pose]),
    [GT_ID, 0]
)

likelihood_outlier_parallel_jit = jax.jit(
    jax.vmap(jax.vmap(j.threedp3_likelihood, in_axes=(None, None, None, 0, None)), in_axes=(None, None, 0, None, None))
)


j.vstack_images(
    [
        j.get_depth_image(observed_image[:,:,2], max=intrinsics.far),
        j.get_depth_image(rendered[:,:,2], max=intrinsics.far)
    ]
).save("data.png")



delta = j.distributions.gaussian_vmf(jax.random.PRNGKey(6), 0.01, 10000.0)
rendered = renderer.render_multiobject(
    jnp.linalg.inv(delta) @ jnp.array([object_pose, wall_pose]),
    [GT_ID, 0]
)
OUTLIER_PROBS = jnp.linspace(0.01, 0.1, 200)
R = jnp.linspace(7.0, 10.0, 100)
OUTLIER_VOLUME = 1000000000.0


delta = j.distributions.gaussian_vmf(jax.random.PRNGKey(6), 0.001, 1000000.0)
rendered = renderer.render_multiobject(
    jnp.linalg.inv(delta) @ jnp.array([object_pose, wall_pose]),
    [GT_ID+1, 0]
)
OUTLIER_PROBS = jnp.linspace(0.01, 0.1, 200)
R = jnp.linspace(1.0, 10.0, 100)
OUTLIER_VOLUME = 1000000000.0


j.meshcat.show_cloud("1", observed_image[...,:3].reshape(-1,3) / 100.0)
j.meshcat.show_cloud("2", rendered[...,:3].reshape(-1,3) / 100.0, color=j.RED)


p = likelihood_outlier_parallel_jit(observed_image, rendered, R, OUTLIER_PROBS, OUTLIER_VOLUME)
norm_p = j.utils.normalize_log_scores(p)

ii,jj = jnp.unravel_index(norm_p.argmax(), norm_p.shape)
best_r, best_outlier_prob = (R[ii], OUTLIER_PROBS[jj])
print(R[ii], OUTLIER_PROBS[jj])


plt.clf()
plt.matshow(norm_p)
plt.xlabel("outlier prob")
plt.ylabel("R")
plt.yticks([0,len(R)-1],[str(np.round(R[0].item(),6)), str(np.round(R[-1].item(),6)) ])
plt.xticks([0,len(OUTLIER_PROBS)-1],[str(np.round(OUTLIER_PROBS[0].item(),6)),str(np.round(OUTLIER_PROBS[-1].item(),6))])
# ([str(OUTLIER_PROBS[0].item())] + ['' for _ in range(len(OUTLIER_PROBS)-2)] + [str(OUTLIER_PROBS[-1].item())])
plt.colorbar()
plt.tight_layout()
plt.savefig("1.png")



outliers = (
    (j.gaussian_mixture_image(observed_image, rendered, best_r) * (1.0 - best_outlier_prob)) 
        <
    (best_outlier_prob / OUTLIER_VOLUME)
)
j.get_depth_image(outliers).save("outliers.png")











from IPython import embed; embed()




