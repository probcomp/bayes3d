import functools
import jax
import jax.numpy as jnp

@functools.partial(
    jnp.vectorize,
    signature='(2,3)->(4)',
    excluded=(1,),
)
def ray_triangle_vmap(ray, vertices):
    dir = ray[0]
    orig = ray[1]
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


def render_triangles(pose, trimesh_shape, rays):
    trimesh_shape_h = jnp.concatenate([trimesh_shape, jnp.ones((*trimesh_shape.shape[:2],1))], axis=-1)
    all_vertices = jnp.einsum("ik,abk->abi", pose, trimesh_shape_h)[:,:,:3]

    points = jax.vmap(ray_triangle_vmap, in_axes=(None, 0), out_axes=-2)(
        rays,
        all_vertices
    )
    idxs = jnp.argmin(points[:,:,:,2],axis=-1)
    points_final = points[jnp.arange(points.shape[0])[:, None], jnp.arange(points.shape[1])[None, :], idxs]
    points_final_final = points_final * (points_final[:,:,2] < 9000.0)[:,:,None]
    return points_final_final