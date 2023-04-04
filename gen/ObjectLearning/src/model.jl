@gen (static) function sample_contact_params(cluster_center)
    cplane ~ uniform_contact_plane()
    relpose ~ relative_pose_prior(cluster_center, Δx, Δy)
    return cplane, relpose
end

function make_scene(shape_paramss, contact_params)
    scene = SceneGraph()
    table_node = FloatingNode(:table, TABLE, TABLE_POSE)
    add_floating_node!(scene, table_node)
    for (i, (shape_params, (cplane, relpose))) in enumerate(zip(shape_paramss, contact_params))
        contact_params = ContactParams(top, cplane, relpose)
        node = ChildNode(Symbol(:obj_, i), shape_params, table_node, contact_params)
        add_child_node!(scene, node)
    end
    scene
end

crp_init(m) = foldl(update, 1:m; init=empty_crp_state)

@gen (static) function model(cluster_centers)
    m, n = length(KNOWN_SHAPES), length(cluster_centers)
    crp_θ ~ exponential(1/5)
    shape_assignments ~ cond_crp(crp_init(m), m+n, crp_θ)
    shape_params ~ shapes_prior(n, shape_assignments)
    contact_params ~ Map(sample_contact_params)(cluster_centers)
    scene = make_scene(shape_params, contact_params)
    cam_pose ~ cam_pose_prior()
    p_outlier ~ beta(P_OUTLIER_SHAPE, P_OUTLIER_SCALE)
    noise ~ inv_gamma(NOISE_SHAPE, NOISE_SCALE) # variance of Gaussian noise
    obs ~ stoch_renderer(scene, cam_pose, p_outlier, noise) # XXX prox
end

@load_generated_functions()
