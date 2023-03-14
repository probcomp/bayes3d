

object_type ~ categorical(NUM_OBJECT_TYPES)
pose ~ uniform_6DOF_pose()

depth_image = render(object_type, pose)
noisy_depth_image ~ sensor_model(depth_image)
