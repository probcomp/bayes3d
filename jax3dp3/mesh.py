import trimesh
import numpy as np
import jax3dp3.transforms_3d as t3d
import jax3dp3
import jax.numpy as jnp
from itertools import product

def center_mesh(mesh):
    _, pose = jax3dp3.bbox.axis_aligned_bounding_box(mesh.vertices)
    shift = np.array(pose[:3,3])
    mesh.vertices = mesh.vertices - shift
    return mesh

def make_table_mesh(
    table_width,
    table_length,
    table_height,
    table_thickness,
    table_leg_width
):
        
    table_face = trimesh.creation.box(
        np.array([table_width, table_length, table_height]),
        np.array(t3d.transform_from_pos(jnp.array([0.0, 0.0, table_height/2.0 - table_thickness/2.])))
    )
    table_leg_height = table_height-table_thickness
    leg_dims =  np.array([table_leg_width, table_leg_width, table_leg_height])
    leg_center = np.array([table_width, table_length])/2. - table_leg_width/2.0*np.ones(2)
    leg_xys = [np.multiply(leg_center, np.array(signs))
                for signs in product([-1, +1], repeat=len(leg_center))]
    table_legs = [
        trimesh.creation.box(
            leg_dims,
            np.array(t3d.transform_from_pos(np.array([x, y, table_leg_height/2. - table_height/2.0])))
        )
        for (x,y) in leg_xys
    ]
    table = trimesh.util.concatenate([table_face] + table_legs)
    return table