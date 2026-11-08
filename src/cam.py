from PIL import Image
import numpy as np
import mujoco as mj
import open3d as o3d
import struct
from spatialmath import SE3, SO3
import math


def get_camera_rgb(
    m: mj.MjModel, d: mj.MjData, renderer: mj.Renderer, camera_name: str = "cam_exterior"
) -> np.ndarray:
    """Render and return the RGB frame as a uint8 numpy array (H, W, 3)."""
    renderer.update_scene(d, camera=camera_name)
    return renderer.render()


def get_dual_camera_rgb(
    m: mj.MjModel, d: mj.MjData, renderer: mj.Renderer
) -> dict[str, np.ndarray]:
    """Capture synchronized RGB frames from both RealSense D415 cameras."""
    return {
        "cam_exterior": get_camera_rgb(m, d, renderer, "cam_exterior"),
        "cam_wrist": get_camera_rgb(m, d, renderer, "cam_wrist"),
    }


def get_rgb(
    m: mj.MjModel, d: mj.MjData, renderer: mj.Renderer, camera_name: str = "cam_exterior"
):
    rgb_img = get_camera_rgb(m, d, renderer, camera_name=camera_name)
    im = Image.fromarray(rgb_img)
    im.save("rgb_img.jpeg")

