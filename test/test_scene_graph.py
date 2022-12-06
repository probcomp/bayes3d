from jax3dp3.scene_graph import get_contact_planes, get_contact_transform, relative_pose_from_contact
import jax.numpy as jnp
import jax
import jax3dp3.transforms_3d as t3d
from jax3dp3.shape import get_rectangular_prism_shape
from jax3dp3.rendering import render_sphere, render_planes_multiobject,render_planes_multiobject_rays
from jax3dp3.viz import save_depth_image, get_depth_image
from jax3dp3.camera import camera_rays_from_params
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.utils import make_centered_grid_enumeration_2d_points
import jax3dp3.viz
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
from PIL import Image
from jax3dp3.batched_scorer import batched_scorer_parallel
import time
import matplotlib

box_dims = jnp.array([
    [30.0, 0.1, 20.0],
    [1.0, 1.0, 1.0],
    [3.0, 4.0, 0.2],
    [6.0, 6.0, 0.2],
    [10.0, 3.0, 0.1]
])
contact_planes = jax.vmap(get_contact_planes)(box_dims)

camera_pose = t3d.transform_from_rot_and_pos(
    t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]),-jnp.pi/6)[:3,:3],
    jnp.array([0.0, -15.0, -30.0])
)

poses = jnp.array([
    jnp.linalg.inv(camera_pose),
    jnp.eye(4),
    jnp.eye(4),
    jnp.eye(4),
    jnp.eye(4),
])

edges = jnp.array([
    [-1,0],
    [0,1],
    [0,2],
    [0,3],
    [0,4],
])

contact_params = jnp.array([
    [0.0, 0.0, jnp.pi/4],
    [-8.0, 0.0, jnp.pi/4],
    [-2.0, 4.0, jnp.pi/2],
    [-8.0, -1.0, jnp.pi/2],
    [7.0, -4.0, jnp.pi/2],
])


def iter(edge, contact_params, poses):
    i, j = edge
    x,y,ang = contact_params
    rel_pose = relative_pose_from_contact(box_dims[i], box_dims[j], 1, 0, (x,y,ang))
    return (
        poses[i].dot(rel_pose) * (i != -1)
        +
        poses[j] * (i == -1)
    )

iter_parallel = jax.vmap(iter, in_axes=(0, 0, None))

def get_poses(edges, contact_params, poses):
    def _f(poses, _):
        new_poses = iter_parallel(edges, contact_params, poses)
        return (new_poses, new_poses)
    return jax.lax.scan(_f, poses, jnp.ones(edges.shape[0]))[0]

f = jax.jit(get_poses)
poses = f(edges, contact_params, poses)

shape_planes, shape_dims = jax.vmap(get_rectangular_prism_shape)(box_dims)

min,max = 1.0, 40.0

h, w, fx, fy, cx, cy = (
    200,
    400,
    400.0,
    400.0,
    200.0,
    100.0,
)
rays = camera_rays_from_params(h,w, fx,fy,cx,cy)
gt_image = render_planes_multiobject_rays(poses, shape_planes, shape_dims, rays)
print('gt_image.shape ',gt_image.shape)
save_depth_image(gt_image[:,:,2], "multiobject.png", min=min,max=max)

key = jax.random.PRNGKey(3)
def scorer(contact_params, obs):
    obj_poses = get_poses(edges, contact_params, poses)
    rendered_image = render_planes_multiobject_rays(obj_poses, shape_planes, shape_dims, rays)
    weight = threedp3_likelihood(obs, rendered_image, 0.1, 0.01)
    return weight

scorer_parallel = jax.vmap(lambda x: scorer(x, gt_image))
scorer_parallel_jit = jax.jit(scorer_parallel)

grid = make_centered_grid_enumeration_2d_points(-15.0, 15.0, -10.0, 10.0, 50, 50)

contact_params_sweep = jnp.stack([contact_params for _ in range(grid.shape[0])])
contact_params_sweep = contact_params_sweep.at[:,1,:2].set(grid)
batched_scorer_jit = jax.jit(lambda x: batched_scorer_parallel(scorer_parallel, 10, x))

batched_scorer_jit(contact_params_sweep)


start = time.time()
x = batched_scorer_jit(contact_params_sweep)
end = time.time()
print ("Time elapsed:", end - start)


best_params = contact_params_sweep[jnp.argsort(x)[-1]]
obj_poses = get_poses(edges, best_params, poses)
best_image = render_planes_multiobject_rays(obj_poses, shape_planes, shape_dims, rays)
print('best_image.shape ',best_image.shape)
save_depth_image(best_image[:,:,2], "best_image.png", max=30.0)

p = jnp.exp(x - logsumexp(x))
plt.clf()
plt.plot(p)
plt.savefig("distribution.png")


idxs = jax.random.categorical(jax.random.PRNGKey(3),x,shape=(1000,))
idxs = jnp.argsort(-x)[:500]
object_idxs = jnp.array([0,1])
idx = idxs[0]

def make_image(contact_params):
    obj_poses = get_poses(edges, contact_params, poses)
    image = render_planes_multiobject_rays(obj_poses[object_idxs], shape_planes[object_idxs], shape_dims[object_idxs], rays)
    return image

make_image_jit = jax.jit(make_image)

images = []

gt_image_viz = get_depth_image(gt_image[:,:,2],  min=min,max=max)
for idx in idxs:
    d = make_image_jit(contact_params_sweep[idx])
    images.append(
       Image.blend(get_depth_image(d[:,:,2],  min=min, max=max), gt_image_viz, 0.5)
    )

images[0].save("test.png")
images[0].save(
    fp="posterior.gif",
    format="GIF",
    append_images=images,
    save_all=True,
    duration=100,
    loop=0,
)

jax3dp3.viz.viz_graph(len(edges), edges, "test.png")

from IPython import embed; embed()
