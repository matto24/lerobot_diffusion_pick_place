import mujoco
import numpy as np

from src.paths import PROJECT_ROOT
from src.cam import get_dual_camera_rgb
from src.robot import ur_set_qpos, hande_ctrl_qpos, ur_ctrl_qpos, ur_get_qpos
from scripts.box_movement_example import program
from pick_place_lerobot.lerobot_dataset import SUBSAMPLE_STEPS


def generate_single_episode(episode_id: int) -> dict | None:
    """Runs a single episode headless and returns the recorded frames."""
    # Each worker loads its own independent MuJoCo model and renderer
    m = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "assets/scenes/scene.xml"))
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, 480, 640)

    # 1. Reset to home pose
    target_pos = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
    ur_set_qpos(d, target_pos)
    hande_ctrl_qpos(d, 0)
    mujoco.mj_forward(m, d)

    # 2. Plan path (RRT*)
    try:
        cmd_queue = program(d, m)
    except Exception as e:
        print(f"Episode {episode_id} planning failed, skipping...")
        return None

    # 3. Step physics and collect frames in a list
    episode_frames = []
    step_count = 0

    while len(cmd_queue) > 0:
        cmd_element, cmd_queue = cmd_queue[0], cmd_queue[1:]
        desired_cmd, gripper_value = cmd_element

        if isinstance(desired_cmd, np.ndarray):
            ur_ctrl_qpos(d, desired_cmd)
        if gripper_value is not None:
            hande_ctrl_qpos(d, gripper_value)

        mujoco.mj_step(m, d)

        if step_count % SUBSAMPLE_STEPS == 0:
            frames = get_dual_camera_rgb(m, d, renderer)
            current_q = ur_get_qpos(d, m)
            current_gripper = d.ctrl[6]

            episode_frames.append(
                {
                    "observation.images.cam_exterior": frames["cam_exterior"],
                    "observation.images.cam_wrist": frames["cam_wrist"],
                    "observation.state": np.append(current_q, current_gripper).astype(
                        np.float32
                    ),
                    "action": np.append(
                        desired_cmd,
                        gripper_value if gripper_value is not None else current_gripper,
                    ).astype(np.float32),
                }
            )
        step_count += 1

    return {"episode_id": episode_id, "frames": episode_frames}
