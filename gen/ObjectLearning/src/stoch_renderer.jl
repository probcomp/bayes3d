struct StochRenderer <: Distribution{DepthImage} end
stoch_renderer = StochRenderer()


function random(::StochRenderer, scene::TableTopSceneGraph, cam_pose::Pose,
                cam_intrinsics::Intrinsics, p_outlier::Float64, noise::Float64)
    full_cloud = render_scene_to_cloud(scene, cam_pose, cam_intrinsics)
    obs_cloud = Array{Float32, 2}(undef, 3, OBS_CLOUD_SIZE)
    outlier_idxs = findall(rand(OBS_CLOUD_SIZE) .< p_outlier)
    inlier_idxs = filter(x -> !in(x, outlier_idxs), 1:OBS_CLOUD_SIZE)
    obs_cloud[:, inlier_idxs] = full_cloud[:, rand(1:size(full_cloud, 2), length(inlier_idxs))]
    obs_cloud[:, outlier_idxs] = rand(3, length(outlier_idxs)) .* [OUTLIER_ΔX, OUTLIER_ΔY, OUTLIER_ΔZ]
    jax_normal = jax.vmap(key->jax.random.multivariate_normal(key, jnp.zeros(3), jnp.eye(3)*noise))
    noise_cloud = np.array(jax_normal(jax.random.split(jax.random.PRNGKey(3), length(inlier_idxs))))
    obs_cloud[:, inlier_idxs] += noise_cloud'
    DepthImage(np.array(j.render_point_cloud(obs_cloud', cam_intrinsics)))
end

# XXX not parallel friendly
# XXX unsound, implement different sound versions
function logpdf(::StochRenderer, depth_img::DepthImage, scene::TableTopSceneGraph,
                cam_pose::Pose, cam_intrinsics::Intrinsics, p_outlier::Float64, noise::Float64)
    full_cloud = render_scene_to_cloud(renderer, scene, cam_pose, cam_intrinsics)
    obs_cloud = image_to_cloud(depth_img)
    j.threedp3_likelihood_jit(full, rendered_img, [noise], p_outlier, OUTLIER_VOLUME)
end