import open3d as o3d
import numpy as np

def create_sphere_geom(center, radius):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.translate(center)
    return mesh

def create_cube_geom(center, half_extents):
    mesh = o3d.geometry.TriangleMesh.create_box(width=half_extents[0]*2, height=half_extents[1]*2, depth=half_extents[2]*2)
    mesh.translate(np.array(center)-np.array(half_extents))
    return mesh

def draw_scene_offscreen(sphere_data, wall_data, sphere_color=[0, 0, 1], wall_color=[1, 0, 0]):
    sphere = create_sphere_geom(sphere_data['positions'], sphere_data['radius'])
    sphere.paint_uniform_color(sphere_color)
    wall = create_cube_geom(wall_data['positions'], wall_data['half_extents'])
    wall.paint_uniform_color(wall_color)

    renderer = o3d.visualization.rendering.OffscreenRenderer(960, 720)
    scene = renderer.scene
    scene.add_geometry("sphere", sphere, sphere_color)
    scene.add_geometry("wall", wall, wall_color)

    # Set up camera
    center = (np.asarray(sphere_data['positions']) + np.asarray(wall_data['positions'])) / 2.0
    eye = center + [3, 3, 3]
    up = [0, 1, 0]
    fov = 60  # vertical field of view in degrees
    renderer.setup_camera(fov, center, eye, up)

    # Render to an image
    image = renderer.render_to_image()

    # Convert image to numpy array
    image = np.asarray(image)
    return image

# Example usage:
sphere_data = {"positions": [0, 0, 0], "radius": 1}
wall_data = {"positions": [3, 0, 0], "half_extents": [0.5, 1, 1]}
image = draw_scene_offscreen(sphere_data, wall_data)