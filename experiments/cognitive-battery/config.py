class Config:
    # Inference metadata
    scene = None
    label_key = "final_location"
    receptacle_name = "bowl"
    num_steps = "auto"
    start_t = 0

    # Rendering
    width = 300
    height = 300
    fov = 90

    # Enumerative tracking
    iterations_per_step = 1
    num_past_poses = 5
    grid_n = 7
    grid_deltas = [0.15, 0.1, 0.05]

    # Prior parameters
    gravity_shift_prior = 0.1
    occlusion_threshold = 10


class swap_config(Config):
    scene = "swap"
    receptacle_name = "mug"
    label_key = "final_object_location"
    start_t = 11
    table_y = 0.5


class gravity_config(Config):
    scene = "gravity"
    receptacle_name = "bowl"
    start_t = 36
    num_steps = 35
    table_y = 0.5


class rotation_config(Config):
    scene = "rotation"
    receptacle_name = "cup"
    start_t = 6
    table_y = 0.5

    label_key = "final_label"


class addition_config(Config):
    scene = "addition"
    receptacle_name = "plate"
    start_t = 6
    table_y = 0.5

    label_key = "final_label"


class shape_config(Config):
    scene = "shape"
    receptacle_name = "plate"
    start_t = 6
    table_y = 0.5

    label_key = "final_label"


class relative_config(Config):
    scene = "relative"
    receptacle_name = "plate"
    start_t = 6
    table_y = 0.5

    label_key = "final_label"


CONFIG_MAP = {c.scene: c for c in Config.__subclasses__()}
