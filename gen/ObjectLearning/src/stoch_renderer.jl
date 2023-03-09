struct DepthImage end
depth_img = DepthImage()

# XXX stubbed
# XXX add prox, use good proposal for pseudomarg
struct StochRenderer <: Distribution{DepthImage} end
stoch_render = StochRenderer()

random(::StochRenderer, scene, cam_pose, p_outlier, noise) = depth_img
logpdf(::StochRenderer, _, scene, cam_pose, p_outlier, noise) = 0.0
