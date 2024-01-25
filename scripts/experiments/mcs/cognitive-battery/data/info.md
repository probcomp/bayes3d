* `meshes` contains the four meshes visible (table, apple, occluders, door) in the the rotaiton experiment generated using Blender and their corresponding material (for 3DMax, if needed).
* `videos` contains 50 videos with resolution `640 x 480` and fov `90`. Each subdirectory `videos/i` contains:
	* `experiment_video.mp4`: the generated video, with a framerate `30 fps`.
	* `experiment_stats.yaml`: contains information about the experiment and camera intrinsics (e.g. width, height, fov).
	* `frames/`: contains each RGB frame JPEG images.
	* `depths/`: contains each depth frame as loadable numpy arrays.
