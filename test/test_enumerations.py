import numpy as np 
import jax
import jax.numpy as jnp
import jax3dp3.transforms_3d as t3d
import jax3dp3 as j
import os
from collections import namedtuple

Intrinsics = namedtuple('Intrinsics', ['height', 'width', 'fx', 'fy', 'cx', 'cy', 'near', 'far'])


## setup intrinsics and renderer
intrinsics = Intrinsics(height=144, width=192, fx=320, fy=320, cx=96.0, cy=72.0, near=0.001, far=50.0)
renderer = j.Renderer(intrinsics, num_layers=25)

## load models
model_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv/models")
model_names = ["obj_" + f"{str(idx+1).rjust(6, '0')}.ply" for idx in range(14)]
mesh_paths = []
for name in model_names:
    mesh_path = os.path.join(model_dir,name)
    mesh_paths.append(mesh_path)
    model_scaling_factor = 1.0/1000.0
    renderer.add_mesh_from_file(
        mesh_path,
        scaling_factor=model_scaling_factor
    )

# render at gt
GT_IDX = 13
gt_pose = t3d.transform_from_pos(jnp.array([0,0,0.75])) @ t3d.transform_from_rot(t3d.rotation_from_axis_angle(jnp.array([1,0,0]), np.pi/2))
rendered = renderer.render_single_object(gt_pose, GT_IDX)  
viz = j.viz.get_depth_image(rendered[:,:,2], min=jnp.min(rendered), max=5.0)
viz = j.viz.resize_image(viz, intrinsics.height, intrinsics.width)
viz.save(f"gt_render.png")

fib_pts = 15
planar_pts = 15
sphere_angle_range = jnp.pi/2
min_rot_angle = 0
max_rot_angle = jnp.pi*2


def get_new_poses(fib_pts, planar_pts, min_rot_angle=0, max_rot_angle=2*jnp.pi, sphere_angle_range=jnp.pi):
    rot_props = j.enumerations.make_rotation_grid_enumeration(fib_pts, planar_pts, min_rot_angle, max_rot_angle, sphere_angle_range=sphere_angle_range)
    new_poses = jnp.einsum("ij,ajk->aik", gt_pose, rot_props)
    return new_poses

new_poses = get_new_poses(fib_pts, planar_pts, min_rot_angle=min_rot_angle, max_rot_angle=max_rot_angle, sphere_angle_range=sphere_angle_range)

depth_viz = []
for new_pose in new_poses:
    new_rendered = renderer.render_single_object(new_pose, GT_IDX)  
    viz = j.viz.get_depth_image(new_rendered[:,:,2], min=jnp.min(new_rendered), max=5.0)
    viz = j.viz.resize_image(viz, intrinsics.height, intrinsics.width)
    depth_viz.append(viz)


j.hvstack_images(depth_viz, planar_pts, fib_pts).save("dataset4.png")


from IPython import embed; embed()
