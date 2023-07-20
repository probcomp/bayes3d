import jax.numpy as jnp
import bayes3d as b
import numpy as np
import bayes3d.utils.ycb_loader
import trimesh
import jax
import os
from tqdm import tqdm


def test_ycb_loading():
    bop_ycb_dir = os.path.join(b.utils.get_assets_dir(), "bop/ycbv")
    rgbd, gt_ids, gt_poses, masks = b.utils.ycb_loader.get_test_img('52', '1', bop_ycb_dir)

    b.setup_renderer(rgbd.intrinsics, num_layers=1)

    model_dir =os.path.join(b.utils.get_assets_dir(), "bop/ycbv/models")
    for idx in range(1,22):
        b.RENDERER.add_mesh_from_file(os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply"),scaling_factor=1.0/1000.0)

    reconstruction_depth = b.RENDERER.render(gt_poses, gt_ids)[:,:,2]
    match_fraction = (jnp.abs(rgbd.depth - reconstruction_depth) < 0.05).mean()
    assert match_fraction > 0.2

bop_ycb_dir = os.path.join(b.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = b.utils.ycb_loader.get_test_img('55', '22', bop_ycb_dir)
poses = jnp.concatenate([jnp.eye(4)[None,...], rgbd.camera_pose @ gt_poses],axis=0)
ids = jnp.concatenate([jnp.array([21]), gt_ids],axis=0)


b.setup_renderer(rgbd.intrinsics, num_layers=1)

model_dir =os.path.join(b.utils.get_assets_dir(), "bop/ycbv/models")
for idx in range(1,22):
    b.RENDERER.add_mesh_from_file(os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply"),scaling_factor=1.0/1000.0)

b.RENDERER.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(), "sample_objs/cube.obj"), scaling_factor=1.0/1000000000.0)





scene_graph = b.scene_graph.SceneGraph(
    root_poses=poses,
    box_dimensions=b.RENDERER.model_box_dims[ids],
    parents=jnp.full(poses.shape[0], -1),
    contact_params=jnp.zeros((poses.shape[0],3)),
    face_parent=jnp.zeros(poses.shape[0], dtype=jnp.int32),
    face_child=jnp.zeros(poses.shape[0], dtype=jnp.int32),
)
assert jnp.isclose(scene_graph.get_poses(), poses).all()

def get_slack(scene_graph, parent_object_index, child_object_index, face_parent, face_child):
    parent_pose = scene_graph.get_poses()[parent_object_index]
    child_pose = scene_graph.get_poses()[child_object_index]
    dims_parent = scene_graph.box_dimensions[parent_object_index]
    dims_child = scene_graph.box_dimensions[child_object_index]
    parent_contact_plane = parent_pose @ b.scene_graph.get_contact_planes(dims_parent)[face_parent]
    child_contact_plane = child_pose @ b.scene_graph.get_contact_planes(dims_child)[face_child]

    contact_params, slack = b.scene_graph.closest_approximate_contact_params(parent_contact_plane, child_contact_plane)
    return jnp.array([parent_object_index, child_object_index, face_parent, face_child]), contact_params, slack

add_edge_scene_graph = jax.jit(b.scene_graph.add_edge_scene_graph)



N = poses.shape[0]
b.setup_visualizer()

get_slack_vmap = jax.jit(b.utils.multivmap(get_slack, (False, False, False, True, True)))

edges = [(0,1),(0,2),(0,3),(0,4),(0,6),(2,5)]
for i,j in edges:
    settings, contact_params, slacks = get_slack_vmap(scene_graph, i,j, jnp.arange(6), jnp.arange(6))
    settings = settings.reshape(-1,settings.shape[-1])
    contact_params = contact_params.reshape(-1,contact_params.shape[-1])
    error = jnp.abs(slacks - jnp.eye(4)).sum([-1,-2]).reshape(-1)
    indices = jnp.argsort(error.reshape(-1))

    parent_object_index, child_object_index, face_parent, face_child = settings[indices[0]]
    scene_graph = add_edge_scene_graph(scene_graph,parent_object_index, child_object_index, face_parent, face_child, contact_params[indices[0]])

node_names = np.array([*b.utils.ycb_loader.MODEL_NAMES, "table"])
scene_graph.visualize("graph.png", node_names=list(map(str,enumerate(node_names[ids]))))

b.clear()
for i,p in enumerate(scene_graph.get_poses()):
    b.show_trimesh(f"pose_{i}", b.RENDERER.meshes[ids[i]])
    b.set_pose(f"pose_{i}", p)

from IPython import embed; embed()


