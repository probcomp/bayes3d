import bayes3d as b
from tqdm import tqdm
import numpy as np
import zmq
import pickle5
import pickle
import subprocess
import zlib



class MCS_Observation:
    def __init__(self, rgb, depth, intrinsics, segmentation):
        """RGBD Image
        
        Args:
            rgb (np.array): RGB image
            depth (np.array): Depth image
            intrinsics (b.camera.Intrinsics): Camera intrinsics
            segmentation (np.array): Segmentation image
        """
        self.rgb = rgb
        self.depth = depth
        self.intrinsics = intrinsics
        self.segmentation  = segmentation


class PhysicsServer():
    def __init__(self):
        self.observations = []
        self.intrinsics = None
        self.scene_ID = None

    def reset(self):
        self.observations = []
        self.intrinsics = None
        self.scene_ID = None

    def get_obs_from_step_metadata(self,rgb,depth,seg):
        colors, seg_final_flat = np.unique(seg.reshape(-1,3), axis=0, return_inverse=True)
        seg_final = seg_final_flat.reshape(seg.shape[:2])
        observation = MCS_Observation(rgb, depth, self.intrinsics, seg_final)
        self.observations.append(observation)

    def process_message(self, message):
        (request_type, args) = message
        if request_type == "reset":
            self.reset()
            (h,w,fx,fy,cx,cy,near,far) = args # (400,600, 514.2991467983065,514.2991467983065,300.0,200.0,0.009999999776482582,150.0)
            intrinsics = b.Intrinsics(
                h,w,fx,fy,cx,cy,near,far
            )
            self.intrinsics = intrinsics
            return True
        elif request_type == "update":
            rgb, depth, seg = args
            self.get_obs_from_step_metadata(rgb,depth,seg)
        elif request_type == "get_info":
            print("Observations loaded")
            self.scene_ID = np.random.randint(79384572398)
            np.savez(f"{self.scene_ID}.npz", self.observations)
            print("Observations saved as NPZ")
            command = ["python", "run_mcs_eval7.py", str(self.scene_ID)]
            try:
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                # The "stdout" attribute of the result object contains the command's standard output
                print("Command Output:")
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                # If the command returns a non-zero exit code, it will raise a CalledProcessError
                print(f"Command failed with exit code {e.returncode}")
                print("Error Output:")
                print(e.stderr)
            
            with open(f"final_result_{self.scene_ID}.pkl", "rb") as file:
                final_result = pickle.load(file)

            return final_result["rating"], final_result["score"], final_result["report"]
        else:
            print("I HAVE NO IDEA WHAT REQUEST YOU'RE MAKING!")

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5432")
physics_server = PhysicsServer()

while True:
    #  Wait for next request from client
    print("Waiting for request...")
    message = pickle5.loads(zlib.decompress(socket.recv()))
    response = physics_server.process_message(message)
    print(f"Sent response {response}...")
    socket.send(zlib.compress(pickle5.dumps(response)))