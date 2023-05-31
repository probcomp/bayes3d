class Config:
    # Inference metadata
    scene = None
    receptacle_name = "bowl"
    num_steps = "auto"
    start_t = 0

    # Rendering
    width = 300
    height = 300
    fov = 90

    # Enumerative tracking
    iterations_per_step = 1
    num_past_poses = 3
    grid_n = 7
    grid_deltas = [0.15, 0.1, 0.05]

    # Prior parameters
    gravity_shift_prior = 0.1
    occlusion_threshold = 10


class swap_config(Config):
    scene = "swap"
    receptacle_name = "mug"
    start_t = 11


class gravity_config(Config):
    scene = "gravity"
    receptacle_name = "bowl"
    start_t = 36
    num_steps = 50


CONFIG_MAP = {c.scene: c for c in {swap_config, gravity_config}}
