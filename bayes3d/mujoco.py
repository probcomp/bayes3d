import os
import PIL
import imageio
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
# For headless rendering, use egl. Otherwise, use GLFW or OSMesa
# os.environ["MUJOCO_GL"] = 'egl'

# PyMJCF imports
from dm_control import mujoco
from dm_control import mjcf
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# basic rendering functionality
class MJFC_Physics_Renderer:
    """
    TODO: currently keyframe is set after each object is loaded, not before
    TODO: function to extract poses from physics.data.geom()
    TODO: function to add or extract meshes (not sure if the latter is possible)
    TODO: multi object keyframe setting is still buggy
    """
    def __init__(self, name, width = '640', height = '480'):
        self.model = mjcf.RootElement(model = name)
        self.model.visual.__getattr__('global').offwidth = str(width)
        self.model.visual.__getattr__('global').offheight = str(height)
        self.model.option.integrator = "RK4"
        camera = self.model.worldbody.add('camera', name="camera", pos="0 -0.5 0.1", euler=[85, 0, 0])
        self.entities = {"camera" : camera}
        self.body_to_id = {}

    def add_light(self, pos, **kwargs):
        """
        Add light to a scene
        """
        light = self.model.worldbody.add('light', name = 'light', pos=pos)
        self.entities['light'] = light

    def add_floor(self, size, **kwargs):
        """
        Add floor to a scene, it is currently set to a chequered floor
        TODO: Set texture/grid/material of floor as inputs
        """
        chequered = self.model.asset.add('texture', type='2d', builtin='checker', width=300, height=300, rgb1=[.1, .2, .3], rgb2=[.2, .3, .4])
        grid = self.model.asset.add('material', name='grid', texture=chequered, texrepeat=[8, 8], reflectance=0)
        floor = self.model.worldbody.add('geom', name = "floor", type='plane', size=size, material=grid, **kwargs)
        self.entities['floor'] = floor

    def add_camera(self, camera_id, **kwargs):
        """
        additional camera, with settings of pos, quat/euler
        """
        camera = self.model.worldbody.add('camera', name=camera_id, **kwargs)
        self.entities[camera_id] = camera        

    def add_object(self, obj_type, name, **kwargs):
        """
        Add object to scene in the form of a body 
        ("possible kwargs include pos, size, euler, rgba, solref, friction etc.")
        """
        assert(name not in ['light', 'floor'] and name not in self.entities)
        body = self.model.worldbody.add('body', name = name + 'body')
        body.add('freejoint')
        body.add('geom', type=obj_type, name=name, **kwargs)
        self.entities[name] = body
        self.body_to_id[name] = len(self.body_to_id) # 0-indexed
        

    def set_object_pose(self, name, qpos, keyframe = 0):
        """
        sets the pose of an object at a specific keyframe
        in the formal [x, y, z, quaternion --> 4 numbers]
        TODO: Accept 4x4 poses too
        """
        assert(len(qpos) == 7) # xyz and quat
        kf_name = 'keyframe_' + str(keyframe)
        body_id = self.body_to_id[name]
        if kf_name in self.entities and self.entities[kf_name].qpos is not None:
            current_qpos = self.entities[kf_name].qpos
        else:
            current_qpos = [0,0,0,1,0,0,0] * len(self.body_to_id)

        current_qpos[body_id*7 : body_id*7 + 7] = qpos

        if kf_name in self.entities:
            self.entities[kf_name].qpos = current_qpos
        else:
            keyframe = self.model.keyframe.add('key', name=kf_name, time= str(keyframe), qpos=current_qpos)
            self.entities[kf_name] = keyframe

    def set_object_velocity(self, name, qvel, keyframe = 0):
        """
        sets the velocity of an object at a specific keyframe
        in the formal [x, y, z, x-axis, y-axis, z-axis]
        """
        assert(len(qvel) == 6) # xyz and quat
        kf_name = 'keyframe_' + str(keyframe)
        body_id = self.body_to_id[name]
        if kf_name in self.entities and self.entities[kf_name].qvel is not None:
            current_qvel = self.entities[kf_name].qvel
        else:
            current_qvel = [0,0,0,0,0,0] * len(self.body_to_id)

        current_qvel[body_id*6 : body_id*6 + 6] = qvel
        
        if kf_name in self.entities:
            self.entities[kf_name].qvel = current_qvel
        else:
            keyframe = self.model.keyframe.add('key', name=kf_name, time=str(keyframe), qvel=current_qvel)
            self.entities[kf_name] = keyframe

    def set_camera_pose(self, pos, camera_id = 'camera', **kwargs):
        """
        Set the camera pose in terms of xyz translation and quaternions
        TODO: Convert 4x4 homogeneous pose to xyz + quat
        TODO: Convert xyz axes rotation to quat
        """
        self.entities[camera_id].pos = pos
        for attr in kwargs:
            self.entities[camera_id].attr = kwargs[attr]

    def get_camera_pose(self):
        pass # TODO

    def set_physical_attribute(self, name, **kwargs):
        for attr in kwargs:
            self.entities[name].attr = kwargs[attr]
    
    #   Then some basic functions to return a scene object of some kind (object thrown into scene, set num_obejcts)
    def render_image(self, render_type = 'rgb', savepath = None):
        if render_type == 'depth':
            depth = True
        else:
            depth = False
        if render_type == 'depth':
            seg = True
        else:
            seg = False
        physics = mjcf.Physics.from_mjcf_model(self.model)
        pixels =  PIL.Image.fromarray(physics.render(camera_id='camera', segmentation=seg, depth=depth))
        if savepath is not None:
            pixels.save(savepath)
        else:
            return pixels

    def render_video(self, duration, render_type = 'rgb', display = True, 
        savepath = None, fps = 30, start_time = 0, camera_id = 'camera'):
        """
        Simulate and display video.
        TODO: More efficient starting duration for video
        """
        frames = []
        physics = mjcf.Physics.from_mjcf_model(self.model)

        if render_type == 'depth':
            depth = True
        else:
            depth = False
        if render_type == 'segmentation':
            seg = True
        else:
            seg = False

        if 'keyframe_0' in self.entities:
            physics.reset(0)  # Reset to keyframe 0 if it exists(load a saved state).

        # go to start time
        while physics.data.time < start_time:
            physics.step()
            print("init")
        # start rendering frames from camera_id
        while physics.data.time < duration:
            physics.step()
            if len(frames) < (physics.data.time) * fps:
                pixels = physics.render(camera_id='camera', segmentation=seg, depth=depth)
                frames.append(pixels)

        if savepath is not None:
            imageio.mimwrite(savepath, frames, fps=fps)

        if display: # in python notebook
            return self.display_video(frames, fps)

    def display_video(self, frames, fps=30):
        """
        This function is taken from dm_control's tutorial
        """
        height, width, _ = frames[0].shape
        dpi = 70
        orig_backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
        fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
        matplotlib.use(orig_backend)  # Switch back to the original backend.
        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.set_position([0, 0, 1, 1])
        im = ax.imshow(frames[0])
        def update(frame):
            im.set_data(frame)
            return [im]
        interval = 1000/fps
        anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                    interval=interval, blit=True, repeat=False)
        return HTML(anim.to_html5_video())

    def get_physics(self):
        return mjcf.Physics.from_mjcf_model(self.model)

    def get_xml(self):
        return self.model.to_xml_string()
