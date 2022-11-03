
current_pose_estimate = ...


images_over_time = []

# Extension is to do this with particles


schedule = [(1.0, 10), (0.1, 10), ]
for params in schedule:
    width, num_points = params
    deltas = make_centered_grid_enumeration_3d_points(width, width, width, num_points, num_points, num_points)
    proposals = jnp.einsum("ij,ajk->aik", current_pose_estimate, deltas)
    weights = scorer_parallel(proposals)
    current_pose_estimate = proposals[jnp.argmax(weights)]
    images_over_time.append(render_from_pose(current_pose_estimate))

    deltas = rotation_gridding(width, width, width, num_points, num_points, num_points)
    proposals = jnp.einsum("ij,ajk->aik", current_pose_estimate, deltas)
    weights = scorer_parallel(proposals)
    current_pose_estimate = proposals[jnp.argmax(weights)]


