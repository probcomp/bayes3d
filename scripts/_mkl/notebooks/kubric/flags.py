import kubric as kb
from argparse import Namespace, ArgumentParser

# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--objects_split", choices=["train", "test"],
                    default="train")
# Configuration for the objects of the scene
parser.add_argument("--min_num_objects", type=int, default=3,
                    help="minimum number of objects")
parser.add_argument("--max_num_objects", type=int, default=10,
                    help="maximum number of objects")
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--backgrounds_split", choices=["train", "test"],
                    default="train")

parser.add_argument("--camera", choices=["fixed_random", "linear_movement"],
                    default="fixed_random")
parser.add_argument("--max_camera_movement", type=float, default=4.0)


# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
parser.add_argument("--gso_assets", type=str,
                    default="gs://kubric-public/assets/GSO/GSO.json")
parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.set_defaults(save_state=False, frame_end=24, frame_rate=12,
                    resolution=256)
FLAGS = parser.parse_args()


print(vars(FLAGS).keys())
print(type(FLAGS))
print(1)


k = set(['objects_split', 'min_num_objects', 'max_num_objects', 'floor_friction', 'floor_restitution', 'backgrounds_split', 'camera', 'max_camera_movement', 'kubasic_assets', 'hdri_assets', 'gso_assets', 'save_state', 'frame_end', 'frame_rate', 'resolution'])
k_ = set(['frame_rate', 'step_rate', 'frame_start', 'frame_end', 'logging_level', 'seed', 'resolution', 'scratch_dir', 'job_dir', 'objects_split', 'min_num_objects', 'max_num_objects', 'floor_friction', 'floor_restitution', 'backgrounds_split', 'camera', 'max_camera_movement', 'kubasic_assets', 'hdri_assets', 'gso_assets', 'save_state'])

print(len(k), len(k_))
print(k == k_)