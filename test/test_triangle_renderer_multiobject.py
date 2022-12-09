import os

from jax3dp3.viz import save_depth_image
import jax.numpy as jnp
import jax3dp3.camera
import jax3dp3.utils
import jax3dp3.triangle_renderer
import jax3dp3.transforms_3d as t3d
from jax3dp3.triangle_renderer import ray_triangle_vmap
import jax
import time
import trimesh

height, width = 200, 200
cx = (width-1)/2
cy = (height-1)/2
fx = 200.0
fy = 200.0
rays = jax3dp3.camera.camera_rays_from_params(height, width, fx, fy, cx, cy)

mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj"))
trimesh_shape = mesh.vertices[mesh.faces][:100]

trimesh_shape = jnp.concatenate([trimesh_shape, trimesh_shape])
trimesh_shape_h = jnp.concatenate([trimesh_shape, jnp.ones((*trimesh_shape.shape[:2],1))], axis=-1)

print('trimesh_shapes.shape:');print(trimesh_shapes.shape)
poses = jnp.stack([
    t3d.transform_from_pos(jnp.array([1.0, 1.0, 5.0])),
    t3d.transform_from_pos(jnp.array([-1.0, -1.0, 5.0]))
])
idxs = jnp.zeros((trimesh_shapes.shape[0],),dtype=jnp.int8)
print('idxs.shape:');print(idxs.shape)

@functools.partial(
    jnp.vectorize,
    signature='(3)->(4)',
    excluded=(1,2,3,),
)
def ray_triangle_vmap(ray, vertices, index, poses):
    dir = ray
    orig = jnp.zeros(3)

    shifted_vertices = jnp.einsum("ik,abk->abi", pose, trimesh_shape_h)[:,:,:3]

    v0,v1,v2 = vertices[0], vertices[1], vertices[2]

    v0v1 = v1 - v0; 
    v0v2 = v2 - v0; 

    N = jnp.cross(v0v1, v0v2)
    area2 = jnp.linalg.norm(N)

    NdotRayDirection = N.dot(dir); 
    bad1 = ~(jnp.abs(NdotRayDirection) < 1e-20)


    d = -N.dot(v0); 
 
    t = -(N.dot(orig) + d) / NdotRayDirection; 
    bad2 =  ~(t < 0)
 
    P = orig + t * dir; 

    edge0 = v1 - v0; 
    vp0 = P - v0; 
    C = jnp.cross(edge0, vp0)

    bad3 = ~(N.dot(C) < 0) 
    
    edge1 = v2 - v1; 
    vp1 = P - v1; 
    C = jnp.cross(edge1, vp1); 
    bad4 = ~(N.dot(C) < 0) 
 
    edge2 = v0 - v2; 
    vp2 = P - v2; 
    C = jnp.cross(edge2, vp2); 
    bad5 = ~(N.dot(C) < 0) 

    valid = bad1 * bad2 * bad3 * bad4 * bad5 * jnp.all(~jnp.isnan(P))
    return jnp.nan_to_num(P) * valid + 10000.0 * (1 - valid)




points = jax.vmap(ray_triangle_vmap, in_axes=(None, 0), out_axes=-2)(
    rays,
    all_vertices
)


from IPython import embed; embed()

img = jax3dp3.triangle_renderer.render_triangles(pose, trimesh_shape, rays)
save_depth_image(img[:,:,2], "triangle.png", max=6.0)


def f(a):
    return jnp.sum(a)

f_jit = jax.jit(f)
f_jit(jnp.ones((4,3)))

from IPython import embed; embed()

