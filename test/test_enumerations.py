import jax.numpy as jnp
import jax3dp3

fib_pts = 50
planar_pts = 50
sphere_angle_range = 1e-10 

pose = jnp.eye(4)
rot_props = jax3dp3.enumerations.make_rotation_grid_enumeration(fib_pts, planar_pts, sphere_angle_range)

# check that more proposed angles are close to jnp.eye(4) as sphere_angle_range -> 0
close_to_identity = [jnp.all(jnp.isclose(rot[:3,:3], jnp.eye(3), atol=0.2)) for rot in rot_props]
print(f"phi range={sphere_angle_range}\n{sum(close_to_identity)} of {len(rot_props)} is close to identity transformation")


new_poses = jnp.einsum("ij,ajk->aik", pose, rot_props)
# TODO visualize with matplotlib


from IPython import embed; embed()
# TODO test visually 