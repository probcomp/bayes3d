import open3d as o3d
import open3d.visualization.rendering as rendering

# Initialize the renderer and scene
renderer = o3d.visualization.rendering.OffscreenRenderer(640, 480)
# renderer.scene = rendering.Open3DScene(renderer)

# # Create a MaterialRecord object
# mat = rendering.MaterialRecord()
# mat.base_color = [1, 0, 0, 1]  # Base color (red in this case)
# mat.shader = "defaultUnlit"

# Create a sphere and add it to the scene
sphere = o3d.geometry.TriangleMesh.create_sphere(1.0)
sphere.compute_vertex_normals()
renderer.scene.add_geometry("sphere", sphere)

# # Fix for material_ids
# m = o3d.visualization.rendering.MaterialRecord()
# sphere.triangle_material_ids = o3d.utility.IntVector([0]*len(sphere.triangles))

# Add the sphere to the scene
# renderer.scene.add_geometry("sphere", sphere, mat)

# # Render the scene
# renderer.scene.scene.set_background([1, 1, 1, 1])
# renderer.scene.scene.set_sun_light([0, 0, -1], [1, 1, 1], 1000000)
# renderer.scene.scene.enable_sun_light(True)
renderer.render_to_image()

# Save the image
img = renderer.get_image()
img.write_image("sphere.png")
