import pybullet as p
import pybullet_data
import pickle

# Initialize the PyBullet physics simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set up the simulation environment
p.setGravity(0, 0, 0) # Zero gravity 
plane_id = p.loadURDF("plane.urdf")

# Create the wall
wall_half_extents = [0.5, 1, 1]
wall_position = [2, 0, 1]
wall_orientation = p.getQuaternionFromEuler([0, 0, 0])
wall_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_half_extents)
p.createMultiBody(0, wall_id, basePosition=wall_position, baseOrientation=wall_orientation)

# Create the sphere
sphere_radius = 0.5
sphere_mass = 1
sphere_position = [-2, 0, 1]
sphere_start_velocity = [5, 0, 0]
sphere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)
p.createMultiBody(sphere_mass, sphere_id, basePosition=sphere_position)
p.resetBaseVelocity(sphere_id, sphere_start_velocity)

sphere_loc = []
wall_loc = []

# Step through the simulation
for _ in range(200):
    p.stepSimulation()
    # record positions of spheres
    sphere_position = p.getBasePositionAndOrientation(sphere_id)[0]
    wall_position = p.getBasePositionAndOrientation(wall_id)[0]
    sphere_loc.append(sphere_position)
    wall_loc.append(wall_position)

# Serialize the sphere locations
with open('sph_loc.pkl', 'wb') as f:
    pickle.dump(sphere_loc, f)
with open('wall_loc.pkl', 'wb') as f:
    pickle.dump(wall_loc, f)

# Get the final position of the sphere
sphere_final_position, _ = p.getBasePositionAndOrientation(sphere_id)
print("Final position:", sphere_final_position)

# Disconnect from the simulation
p.disconnect()
