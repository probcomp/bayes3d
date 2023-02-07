def get_occluded_occupied_free_masks(grid, camera_pose, depth_image, fx,fy,cx,cy, tolerance = 0.01):
    grid_in_cam_frame = t3d.apply_transform(grid, t3d.inverse_pose(camera_pose))
    pixels = jax3dp3.project_cloud_to_pixels(grid_in_cam_frame, fx,fy,cx,cy).astype(jnp.int32)
    real_depth_vals = depth_image[pixels[:,1],pixels[:,0]]
    projected_depth_vals = grid_in_cam_frame[:,2]
    occupied = jnp.abs(real_depth_vals - projected_depth_vals) < tolerance
    occluded = real_depth_vals < projected_depth_vals
    occluded = occluded * (1.0 - occupied)
    free = (1.0 - occluded) * (1.0 - occupied)
    return occupied > 0, occluded > 0, free > 0