import mujoco
import numpy as np

from src.paths import PROJECT_ROOT
from src.cam import get_dual_camera_rgb
from src.robot import ur_set_qpos, hande_ctrl_qpos, ur_ctrl_qpos, ur_get_qpos
from scripts.box_movement_example import program
from pick_place_lerobot.lerobot_dataset import IMAGE_HEIGHT, IMAGE_WIDTH, SUBSAMPLE_STEPS

# hande_fingers_actuator maps ctrl 0..255 onto the "split" tendon length 0..0.025 m,
# so this scale puts the measured opening in the same units as the gripper command.
GRIPPER_LENGTH_TO_CTRL = 255.0 / 0.025


def measured_gripper(m, d) -> float:
    """Actual finger opening in command units (0 open .. 255 closed)."""
    tendon_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, "split")
    return float(d.ten_length[tendon_id] * GRIPPER_LENGTH_TO_CTRL)


def frame_targets(d, window) -> tuple[np.ndarray, float]:
    """Joint and gripper targets in effect at the end of the next control window.

    One policy action is held for SUBSAMPLE_STEPS physics steps, so the action
    recorded for a frame is where the expert is heading by the next frame.
    """
    q_target = d.ctrl[0:6].copy()
    gripper_target = float(d.ctrl[6])
    for desired_cmd, gripper_value in window:
        if isinstance(desired_cmd, np.ndarray):
            q_target = desired_cmd
        if gripper_value is not None:
            gripper_target = float(gripper_value)
    return q_target, gripper_target


def generate_single_episode(episode_id: int) -> dict | None:
    """Runs a single episode headless and returns the recorded frames."""
    # Each worker loads its own independent MuJoCo model and renderer
    m = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "assets/scenes/scene.xml"))
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, IMAGE_HEIGHT, IMAGE_WIDTH)

    # 1. Reset to home pose
    target_pos = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
    ur_set_qpos(d, target_pos)
    hande_ctrl_qpos(d, 0)
    mujoco.mj_forward(m, d)

    # 2. Plan path (RRT*)
    try:
        cmd_queue = program(d, m)
    except Exception as e:
        print(f"Episode {episode_id} planning failed ({e}), skipping...")
        return None

    # 3. Step physics and collect frames in a list.
    # Each frame observes the scene *before* acting, then pairs it with the
    # target the expert reaches over the next SUBSAMPLE_STEPS physics steps,
    # matching deployment: observe -> act -> hold action for one frame.
    episode_frames = []
    for step_count, (desired_cmd, gripper_value) in enumerate(cmd_queue):
        if step_count % SUBSAMPLE_STEPS == 0:
            frames = get_dual_camera_rgb(m, d, renderer)
            current_q = ur_get_qpos(d, m)
            q_target, gripper_target = frame_targets(
                d, cmd_queue[step_count : step_count + SUBSAMPLE_STEPS]
            )

            episode_frames.append(
                {
                    "observation.images.cam_exterior": frames["cam_exterior"],
                    "observation.images.cam_wrist": frames["cam_wrist"],
                    "observation.state": np.append(
                        current_q, measured_gripper(m, d)
                    ).astype(np.float32),
                    "action": np.append(q_target, gripper_target).astype(np.float32),
                }
            )

        if isinstance(desired_cmd, np.ndarray):
            ur_ctrl_qpos(d, desired_cmd)
        if gripper_value is not None:
            hande_ctrl_qpos(d, gripper_value)

        mujoco.mj_step(m, d)

    return {"episode_id": episode_id, "frames": episode_frames}
