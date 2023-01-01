import numpy as np

def classify(gt_image, scorer_parallel_jit, render, poses_to_score, h,w, object_indices):
    all_scores = []
    for idx in object_indices:
        images = render(poses_to_score, h,w,idx)
        weights = scorer_parallel_jit(images, gt_image)
        all_scores.append(weights.max())
    return object_indices[np.argmax(all_scores)]
