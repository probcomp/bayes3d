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
j.get_depth_image(observed_image[:,:,2],max=intrinsics.far).save("gt.png")


object_pose_noisy = j.distributions.gaussian_vmf_sample(
    jax.random.PRNGKey(5),
    object_pose,
    3.0, 800.0
)

rendered_1 = renderer.render_multiobject(
    jnp.array([object_pose_noisy, wall_pose]),
    [GT_ID, 0]
)
rendered_2 = renderer.render_multiobject(
    jnp.array([object_pose_noisy, wall_pose]),
    [GT_ID+1, 0]
)

object_pose_noisier = j.distributions.gaussian_vmf_sample(
    jax.random.PRNGKey(9),
    object_pose,
    3.0, 400.0
)

rendered_3 = renderer.render_multiobject(
    jnp.array([object_pose_noisier, wall_pose]),
    [GT_ID, 0]
)

likelihood_outlier_parallel_jit = jax.jit(
    jax.vmap(j.threedp3_likelihood, in_axes=(None, None, None, 0, None))
)
OUTLIER_PROBS = jnp.linspace(0.0001, 0.5, 200)
R = 20.0
OUTLIER_VOLUME = 10000.0

def make_outlier_posterior_graph(x,y):
    plt.clf()
    color = np.array([229, 107, 111])/255.0
    plt.plot(x, y, label="Matched", color=color)
    plt.fill_between(x, y, color=color, alpha=0.5)
    plt.xlim((0.0, 0.25))
    plt.ylim((0.0, 1.1))
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4],fontsize=15)
    plt.xlabel("Outlier Probability",fontsize=20)
    plt.yticks([ 0.2, 0.4, 0.6, 0.8, 1.0],fontsize=15)
    plt.ylabel("Probability",fontsize=20)
    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    return im


middle_width = 50
mean_outlier_prob = 0.01
var_outlier_prob = 0.05
prior_on_outlier_prob = jnp.array([
     jax.scipy.stats.multivariate_normal.logpdf(O, mean_outlier_prob, var_outlier_prob) for O in OUTLIER_PROBS
])


point_cloud = observed_image[observed_image[:,:,2] < 799.0,:3].reshape(-1,3)
noise = jax.vmap(
    lambda key: jax.random.multivariate_normal(
        key, jnp.zeros(3), jnp.eye(3) * 0.2
    )
)(
    jax.random.split(jax.random.PRNGKey(3), point_cloud.shape[0])
)
point_cloud_noisy = noise + point_cloud
img_noisy = j.render_point_cloud(point_cloud_noisy, intrinsics)
j.get_depth_image(img_noisy[:,:,2],max=intrinsics.far).save("noisy.png")


rendered_images = [rendered_1, rendered_2, rendered_3]
viz_panels = []
for (idx,rendered) in enumerate(rendered_images):

    scores_matched = likelihood_outlier_parallel_jit(observed_image, rendered, R, OUTLIER_PROBS, OUTLIER_VOLUME)

    viz_outlier_panels = []

    probs = j.gaussian_mixture_image(observed_image, rendered, R)
    for o in OUTLIER_PROBS:
        mask = (probs * (1.0 - o) / (probs.shape[0] * probs.shape[1]) ) < (o / OUTLIER_VOLUME)
        viz_outlier_panels.append(j.get_depth_image(mask))
    j.hstack_images(viz_outlier_panels).save(f"{idx}.png")

    normalized_scores_matched = j.utils.normalize_log_scores(scores_matched)


    outlier_viz = make_outlier_posterior_graph(OUTLIER_PROBS, normalized_scores_matched)

    height_factor = outlier_viz.size[1] / observed_image.shape[0]

    gt_viz = j.scale_image(j.get_depth_image(observed_image[...,2],max=800.0), height_factor)
    rendered_1_viz = j.scale_image(j.get_depth_image(rendered[:,:,2],max=800.0), height_factor)

    panel = j.multi_panel(
        [gt_viz, rendered_1_viz, outlier_viz],
        labels=["Observed Image", "Latent Image", "Posterior on Outlier Probability"],
        label_fontsize=40,
        middle_width = middle_width
    )
    viz_panels.append(panel)

j.vstack_images(viz_panels).save("posterior_on_outlier_prob.png")




from IPython import embed; embed()




