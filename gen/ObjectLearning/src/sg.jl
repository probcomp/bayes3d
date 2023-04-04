@enum ContactPlane bottom top back front left right

struct ContactParams
    parent_plane::ContactPlane
    child_plane::ContactPlane
    child_relative_pose::Tuple{Float64,Float64,Float64} #Δx, Δy, θ 
end

abstract type SceneGraphNode end

struct ChildNode <: SceneGraphNode
    label::Symbol
    mesh::Mesh
    contact_params::ContactParams
end

struct FloatingNode <: SceneGraphNode
    label::Symbol
    mesh::Mesh
    pose::Pose
end

struct TableTopSceneGraph 
    table_node::FloatingNode
    object_nodes::Vector{ChildNode}
end

num_objs(scene::TableTopSceneGraph) = length(scene.object_nodes)

TableTopSceneGraph(table_node) = TableTopSceneGraph(table_node, ChildNode[])

add_child_node!(sg::TableTopSceneGraph, node::ChildNode) = 
    push!(sg.object_nodes, node)