import open3d as o3d
import open3d.visualization.rendering as rendering

intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, 525.0, 525.0, 319.5, 239.5)
print(intrinsics)