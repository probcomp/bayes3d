import jax.numpy as jnp
import jax
import bayes3d.transforms_3d as t3d
from collections import namedtuple
 
class SceneGraph(namedtuple('SceneGraph', ['root_poses', 'box_dimensions', 'parents', 'contact_params', 'face_parent', 'face_child'])):
    """Scene graph data structure.
    
    Args:
        root_poses: Array of root poses. Shape (N,4,4).
        box_dimensions: Array of bounding box dimensions. Shape (N,3).
        parents: Array of parent indices. Shape (N,).
        contact_params: (N,3) array of contact parameters.
        face_parent: (N,) array of parent face indices.
        face_child: (N,) array of child face indices.
    """

    def get_poses(self):
        return poses_from_scene_graph(
            self.root_poses,
            self.box_dimensions,
            self.parents,
            self.contact_params,
            self.face_parent,
            self.face_child,
        )
    
    def visualize(self, filename, node_names=None, colors=None):
        import graphviz
        import matplotlib
        import distinctipy

        scene_graph = self
        num_nodes = len(scene_graph.root_poses)

        if node_names is None:
            node_names = [f"node_{i}" for i in range(len(scene_graph.root_poses))]
        if colors is None:
            colors = distinctipy.get_colors(num_nodes, pastel_factor=0.7)

        g_out = graphviz.Digraph()
        g_out.attr("node", style="filled")

        for i in range(num_nodes):
            g_out.node(f"{i}", node_names[i], fillcolor=matplotlib.colors.to_hex(colors[i]))


        edges = []
        edge_label = []
        for i,parent in enumerate(scene_graph.parents):
            if parent == -1:
                continue
            edges.append((parent, i))
            contact_string = f"contact:\n" + " ".join([f"{x:.2f}" for x in scene_graph.contact_params[i]])
            contact_string += f"\nfaces\n{scene_graph.face_parent[i].item()} --- {scene_graph.face_child[i].item()}" 
            edge_label.append(contact_string)

        for ((i,j),label) in zip(edges, edge_label):
            if i==-1:
                continue
            g_out.edge(f"{i}",f"{j}", label=label)

        max_width_px = 2000
        max_height_px = 2000
        dpi = 200

        g_out.attr("graph",
                    # See https://graphviz.gitlab.io/_pages/doc/info/attrs.html#a:size
                    size="{},{}!".format(max_width_px / dpi, max_height_px / dpi),
                    dpi=f"{dpi}")
        filename_prefix, filetype = filename.split(".")
        g_out.render(filename_prefix, format=filetype)

def create_floating_scene_graph(scene_graph):
    """Create a new scene graph with the same structure, but with all objects floating.

    Returns:
        A new scene graph with the same structure, but with all objects floating.
    """
    return SceneGraph(
        root_poses=scene_graph.get_poses(),
        box_dimensions=scene_graph.box_dimensions,
        parents=jnp.full(scene_graph.parents.shape, -1),
        contact_params=jnp.zeros(scene_graph.contact_params.shape),
        face_parent=jnp.zeros(scene_graph.face_parent.shape, dtype=jnp.int32),
        face_child=jnp.zeros(scene_graph.face_child.shape, dtype=jnp.int32),
    )


def add_edge_scene_graph(scene_graph, parent, child, face_parent, face_child, contact_params):
    print(parent, child, face_parent, face_child)
    N = scene_graph.get_poses().shape[0]
    sg_parents = jnp.array(scene_graph.parents)
    sg_parents = sg_parents.at[child].set(parent)
    sg_contact_params = jnp.array(scene_graph.contact_params)
    sg_contact_params = sg_contact_params.at[child].set(contact_params)
    sg_face_parent = jnp.array(scene_graph.face_parent)
    sg_face_parent = sg_face_parent.at[child].set(face_parent)
    sg_face_child = jnp.array(scene_graph.face_child)
    sg_face_child = sg_face_child.at[child].set(face_child)
    return SceneGraph(
        root_poses=scene_graph.get_poses(),
        box_dimensions=scene_graph.box_dimensions,
        parents=sg_parents,
        contact_params=sg_contact_params,
        face_parent=sg_face_parent,
        face_child=sg_face_child,
    )

def get_contact_planes(dimensions):
    return jnp.stack(
        [
            # bottom
            t3d.transform_from_pos(jnp.array([0.0, dimensions[1]/2.0, 0.0])).dot(t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/2)),
            # top
            t3d.transform_from_pos(jnp.array([0.0, -dimensions[1]/2.0, 0.0])).dot(t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi/2)),
            # back
            t3d.transform_from_pos(jnp.array([0.0, 0.0, dimensions[2]/2.0])).dot(t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), 0.0)),
            # front
            t3d.transform_from_pos(jnp.array([0.0, 0.0, -dimensions[2]/2.0])).dot(t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi)),
            # left
            t3d.transform_from_pos(jnp.array([-dimensions[0]/2.0, 0.0, 0.0])).dot(t3d.transform_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), -jnp.pi/2)),
            # right
            t3d.transform_from_pos(jnp.array([dimensions[0]/2.0, 0.0, 0.0])).dot(t3d.transform_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), jnp.pi/2)),
        ]
    )

def contact_params_to_pose(contact_params):
    x,y,angle = contact_params
    return t3d.transform_from_pos(jnp.array([x,y, 0.0])).dot(
        t3d.transform_from_axis_angle(jnp.array([1.0, 1.0, 0.0]), jnp.pi).dot(
            t3d.transform_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), angle)
        )
    )

def relative_pose_from_edge(
    contact_params,
    face_child, dims_child,
):
    x,y,angle = contact_params
    contact_transform = (
        t3d.transform_from_pos(jnp.array([x,y, 0.0])).dot(
            t3d.transform_from_axis_angle(jnp.array([1.0, 1.0, 0.0]), jnp.pi).dot(
                t3d.transform_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), angle)
            )
        )
    )
    child_plane = get_contact_planes(dims_child)[face_child]
    return contact_transform.dot(jnp.linalg.inv(child_plane))

relative_pose_from_edge_jit = jax.jit(relative_pose_from_edge)
relative_pose_from_edge_parallel_jit = jax.jit(
    jax.vmap(
        relative_pose_from_edge,
        in_axes=(0, 0, 0),
    )
)

def iter(root_poses, box_dimensions, parent, child, contact_params, face_parent, face_child):
    parent_plane = get_contact_planes(box_dimensions[parent])[face_parent]
    relative = parent_plane.dot(
        relative_pose_from_edge(contact_params, face_child, box_dimensions[child])
    )
    return (
        root_poses[parent].dot(relative) * (parent != -1)
        +
        root_poses[child] * (parent == -1)
    )

def poses_from_scene_graph(root_poses, box_dimensions, parents, contact_params, face_parent, face_child):
    def _f(poses, _):
        new_poses = jax.vmap(iter, in_axes=(None, None, 0, 0, 0, 0, 0))(poses, box_dimensions, parents, jnp.arange(parents.shape[0]), contact_params, face_parent, face_child)
        return (new_poses, new_poses)
    return jax.lax.scan(_f, root_poses, jnp.ones(parents.shape[0]))[0]
poses_from_scene_graph_jit = jax.jit(poses_from_scene_graph)

def closest_approximate_contact_params(parent_contact_plane, child_contact_plane):
    contact_pose = t3d.inverse_pose(parent_contact_plane) @ child_contact_plane
    (x, y, _) = contact_pose[:3,3]
    pose_ = (
        t3d.inverse_pose(t3d.transform_from_axis_angle(jnp.array([1.0, 1.0, 0.0]), jnp.pi)) @ 
        contact_pose
    )
    quaternion = t3d.rotation_matrix_to_quaternion(pose_[:3,:3])
    angle = 2 * jnp.arctan2(quaternion[3], quaternion[0])
    inferred_contact_params = jnp.array([x,y,angle])
    slack = t3d.inverse_pose(contact_params_to_pose(inferred_contact_params)) @ contact_pose
    return inferred_contact_params, slack