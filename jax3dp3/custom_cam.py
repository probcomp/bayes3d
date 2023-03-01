import kubric as kb
import numpy as np 

#class to extend the kb.PerspectiveCamera class to add the ability to set the camera pose and intrinsics
class KubCamera(kb.PerspectiveCamera):
    def __init__(self, 
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
    #set intrinsics
    def update_intrinsic_values(self, width, height, f_x, f_y, p_x, p_y, near, far):
        self.width = width
        self.height = height
        self.f_x = float(f_x)
        self.f_y = float(f_y)
        self.p_x = float(p_x)
        self.p_y = float(p_y)
        self.near = float(near)
        self.far = float(far)
        return np.array([[f_x, 0., p_x],
                        [0., f_y, p_y],
                        [0., 0., 1.]])
    
    #set camera sensor properties with default intrinsic calculations 
    def update_camera_settings(self, focal_length, sensor_width):
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.f_x = self.focal_length/self.sensor_width*self.width
        self.f_y = self.focal_length/self.sensor_width*self.height
        self.p_x = self.width / 2
        self.p_y = self.height / 2
        return np.array([[self.f_x, 0., self.p_x],
                        [0., self.f_y, self.p_y],
                        [0., 0., 1.]])
    
    #set camera pose with position and quaternion
    def update_camera_pose(self, position, quaternion):
        self.position = position
        self.quaternion = quaternion
        return self.position, self.quaternion
    
    def sanity_check(self):
        print(dir(self))