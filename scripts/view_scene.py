import mujoco
import numpy as np
import mujoco.viewer
import mujoco as mj

from src.robot import ur_set_qpos, hande_ctrl_qpos
from src.paths import STANDARD_SCENE_PATH

if __name__ == "__main__":
    # Initialize OpenGL context first
    mj.GLContext(max_width=1280, max_height=720)  # Adjust size as needed

    model_path = STANDARD_SCENE_PATH

    time_step = 0.002  # Defined in scene.xml

    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(model=m, data=d) as viewer:
        target_pos = np.array(
            [0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]
        )  # UR home position
        ur_set_qpos(data=d, q_desired=target_pos)
        hande_ctrl_qpos(data=d, gripper_value=0)  # Open gripper
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
            # time.sleep(0.01)
