import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)


# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_record_to_file('object_detection.bag')
align = rs.align(rs.stream.color)

# Start streaming
pipeline.start(config)

e1 = cv2.getTickCount()

rgb_images = []
depth_images = []
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()).copy()
        color_image = np.asanyarray(color_frame.get_data()).copy()
        rgb_images.append(color_image)
        depth_images.append(depth_image)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
        e2 = cv2.getTickCount()
        t = (e2 - e1) / cv2.getTickFrequency()
        if t>30: # change it to record what length of video you are interested in
            print("Done!")
            break

finally:
    np.savez("data.npz",rgb_images=rgb_images, depth_images=depth_images)
    # Stop streaming
    pipeline.stop()