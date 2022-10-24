from .likelihood import neural_descriptor_likelihood
from .rendering import render_planes

def make_scoring_function(shape, h, w, fx_fy, cx_cy, r, outlier_prob):
    def scorer(pose, gt_image):
        rendered_image = render_planes(pose, shape, h, w, fx_fy, cx_cy)
        weight = neural_descriptor_likelihood(gt_image, rendered_image, r, outlier_prob)
        return weight
    return scorer
