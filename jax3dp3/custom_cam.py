import kubric as kb
import numpy as np 
from kubric.kubric_typing import ArrayLike

#class to extend the kb.PerspectiveCamera class to add the ability to set the camera pose and intrinsics
class KubCamera(kb.PerspectiveCamera):
    def __init__(self, 
                 f_x = None, f_y = None, p_x = None, p_y = None, near = None, far = None,
                 focal_length: float = 50.0,
                 sensor_width: float = 36.0,
                 position = (0., 0., 0.),
                 quaternion = None, up="Y", front="-Z", look_at=None, euler=None, **kwargs):
        super().__init__(focal_length=focal_length, sensor_width=sensor_width, position=position,
                     quaternion=quaternion, up=up, front=front, look_at=look_at, euler=euler,
                     **kwargs)
        self.f_x = focal_length
        self.f_y = focal_length
        self.width, self.height = 1., 1.
        self.p_x = self.width / 2
        self.p_y = self.height / 2

        
    @property
    def intrinsics(self):
        return np.array([[self.f_x, 0., self.p_x],
                        [0., self.f_y, self.p_y],
                        [0., 0., 1.]])
    
    #set camera pose with position and quaternion
    def update_camera_pose(self, position, quaternion):
        self.position = position
        self.quaternion = quaternion
        return self.position, self.quaternion
    
    # check for z_to_depth 
    def z_to_depth(self, z: ArrayLike) -> np.ndarray:
        raise NotImplementedError("z_to_depth not implemented for custom camera")
    
    def sanity_check(self):
        print(dir(self))