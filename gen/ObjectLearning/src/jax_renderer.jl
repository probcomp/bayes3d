Intrinsics = @NamedTuple begin
    height::Int64
    width::Int64
    fx::Float64
    fy::Float64
    cx::Float64
    cy::Float64
    near::Float64
    far::Float64
end

struct JAXRenderer
    pyobj::PyObject
end

struct DepthImage
    arr::Array{Float32,3}
end

make_renderer(intrinsics::Intrinsics) = JAXRenderer(j.Renderer(intrinsics))
    
add_mesh!(renderer::JAXRenderer, mesh::Mesh) = renderer.pyobj.add_mesh(mesh.pyobj)

render_object_at_pose(renderer::JAXRenderer, reg_idx::Int, p::Pose) =
    np.array(renderer.pyobj.render_single_object(jnp.array(homcoords(p)), reg_idx))

function image_to_cloud(img::DepthImage)
    idxs = findall(img.arr[:, :, 3] .> 0);
    hcat([img.arr[idx.I..., 1:3] for idx in idxs]...)
end

function add_scene_meshes!(renderer::JAXRenderer, scene::TableTopSceneGraph)
    add_mesh!(renderer, scene.table_node.mesh)
    for node in scene.object_nodes
        add_mesh!(renderer, node.mesh)
    end
end

function render_scene(scene::TableTopSceneGraph, cam_pose::Pose, cam_intrinsics::Intrinsics)
    renderer = make_renderer(cam_intrinsics)
    add_scene_meshes!(renderer, scene)
    absolute_poses = jnp.array([homcoords(scene.table_node.pose),
                                fill(homcoords(IDENTITY_POSE), length(scene.object_nodes))...])
    edges = jnp.array([[-1,0], [[0,i] for i=1:num_objs(scene)]...])
    contact_params = jnp.array([[0.0, 0.0, 0.0],
                               [[n.contact_params.child_relative_pose...] for n in scene.object_nodes]...])
    face_parents = jnp.array([Int(top),[Int(n.contact_params.parent_plane) for n in scene.object_nodes]...])
    face_childs = jnp.array([Int(bottom),[Int(n.contact_params.child_plane) for n in scene.object_nodes]...])
    poses = j.scene_graph.absolute_poses_from_scene_graph_jit(
        absolute_poses, renderer.pyobj.model_box_dims, edges, contact_params, face_parents, face_childs)
    img = np.array(renderer.pyobj.render_multiobject(
        jnp.matmul(jnp.linalg.inv(homcoords(cam_pose)), poses),
        0:length(scene.object_nodes)))
    pydecref(renderer.pyobj)
    DepthImage(img)
end

render_scene_to_cloud(scene::TableTopSceneGraph, cam_pose::Pose, cam_intrinsics::Intrinsics) =
    image_to_cloud(render_scene(scene, cam_pose, cam_intrinsics))
    
