import jax.numpy as jnp
import jax3dp3 as j
import trimesh

table_dims = jnp.array([27.0, 0.1, 20.0])

model_names = ["bunny", "sphere", "pyramid", "cube"]
intrinsics = j.Intrinsics(
    200,
    400,
    400.0,
    400.0,
    200.0,
    100.0,
    0.0,
    10.0
)
meshes = []



box_dims = jnp.array([
    [27.0, 0.1, 20.0],
    [1.0, 1.0, 1.0],
    [3.0, 4.0, 0.2],
    [6.0, 6.0, 0.2],
    [10.0, 3.0, 0.1]
])

camera_pose = t3d.transform_from_rot_and_pos(
    t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]),-jnp.pi/6)[:3,:3],
    jnp.array([0.0, -18.0, -33.0])
)

absolute_poses = jnp.array([
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

contact_params = jnp.array(
    [
        [0.0, 0.0, jnp.pi/4],
        [-8.0, -2.0, jnp.pi/4],
        [-2.0, 2.0, jnp.pi/2],
        [-8.0, -4.0, jnp.pi/2],
        [7.0, -7.0, jnp.pi/2],
    ]
)

face_params = jnp.array(
    [
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
    ]
)

poses = get_poses(
    absolute_poses, box_dims, edges, contact_params, face_params
)

shape_planes, shape_dims = jax.vmap(get_rectangular_prism_shape)(box_dims)

min,max = 1.0, 45.0
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

from IPython import embed; embed()


key = jax.random.PRNGKey(3)
def scorer(contact_params, obs):
    obj_poses = get_poses(poses, box_dims, edges, contact_params, face_params)
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
       Image.blend(get_depth_image(d[:,:,2],  min=min, max=max), gt_image_viz, 0.4)
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

jax3dp3.viz.viz_graph(len(edges), edges, "test.png", node_names=["table", "obj_1", "obj_2","obj_3","obj_4"])

from IPython import embed; embed()
