import jax3dp3 as j

class PhysicsState(object):
    def __init__(self):
        self.counter = 0
        self.intrinsics = j.Intrinsics(
            height=300,
            width=300,
            fx=200.0, fy=200.0,
            cx=150.0, cy=150.0,
            near=0.001, far=50.0
        )
        self.renderer = j.Renderer(self.intrinsics)
    
    def update(self, args):
        self.counter += 2
        return self.counter

    def final_prediction(self, args):
        self.counter += 1
        return self.counter

def load_mcs_scene_data(scene_path):
    cache_dir = os.path.join(j.utils.get_assets_dir(), "mcs_cache")
    scene_name = scene_path.split("/")[-1]
    
    cache_filename = os.path.join(cache_dir, f"{scene_name}.npz")
    if os.path.exists(cache_filename):
        images = np.load(cache_filename,allow_pickle=True)["arr_0"]
    else:
        controller = mcs.create_controller(
            os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons",  "config_level2.ini")
        )

        scene_data = mcs.load_scene_json_file(os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", scene_name +".json"))

        step_metadata = controller.start_scene(scene_data)
        image = j.RGBD.construct_from_step_metadata(step_metadata)

        step_metadatas = [step_metadata]
        for _ in tqdm(range(200)):
            step_metadata = controller.step("Pass")
            if step_metadata is None:
                break
            step_metadatas.append(step_metadata)

        all_images = []
        for i in tqdm(range(len(step_metadatas))):
            all_images.append(j.RGBD.construct_from_step_metadata(step_metadatas[i]))

        images = all_images
        np.savez(cache_filename, images)

    return images