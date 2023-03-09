@enum ContactPlane top bot left right front back

struct ContactParams
    parent_plane::ContactPlane
    child_plane::ContactPlane
    child_relative_pose::Pose 
end

abstract type SceneGraphNode end

struct ChildNode <: SceneGraphNode
    label::Symbol
    shape_params::ShapeModelParams
    parent::SceneGraphNode
    contact_params::ContactParams
end

struct FloatingNode <: SceneGraphNode
    label::Symbol
    shape_params::ShapeModelParams
    pose::Pose
    children::Vector{ChildNode}
end

FloatingNode(label::Symbol, shape_params::ShapeModelParams, pose::Pose) =
    FloatingNode(label, shape_params, pose, ChildNode[])

"""
A scene graph is a forest of rooted trees. All contacts are flush. All contacts
are parameterized w.r.t. bounding boxes.
"""
struct SceneGraph 
    source_nodes::Vector{FloatingNode}
end

SceneGraph() = SceneGraph(FloatingNode[])

add_floating_node!(sg::SceneGraph, node::SceneGraphNode) =
    push!(sg.source_nodes, node)

add_child_node!(sg::SceneGraph, node::ChildNode) = 
    push!(node.parent.children, node)
