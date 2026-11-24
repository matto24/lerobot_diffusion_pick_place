import mujoco as mj
import numpy as np


def get_camera_rgb(
    m: mj.MjModel, d: mj.MjData, renderer: mj.Renderer, camera_name: str = "cam_exterior"
) -> np.ndarray:
    """Render and return the RGB frame as a uint8 numpy array (H, W, 3)."""
    renderer.update_scene(d, camera=camera_name)
    return renderer.render()


def get_dual_camera_rgb(
    m: mj.MjModel, d: mj.MjData, renderer: mj.Renderer
) -> dict[str, np.ndarray]:
    """Capture synchronized RGB frames from the exterior and wrist cameras."""
    return {
        "cam_exterior": get_camera_rgb(m, d, renderer, "cam_exterior"),
        "cam_wrist": get_camera_rgb(m, d, renderer, "cam_wrist"),
    }
