from lerobot.datasets import LeRobotDataset

FPS = 25  # Policy control frequency (Hz)
SIM_DT = 0.002  # MuJoCo physics timestep (500 Hz)
SUBSAMPLE_STEPS = int(1.0 / (FPS * SIM_DT))  # Record every 20 physics steps

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
    "gripper",
]
# names[2] must be "channels": LeRobot uses it to convert HWC -> CHW for policies.
IMAGE_NAMES = ["height", "width", "channels"]


def create_dataset(
    repo_id: str = "local/ur5e_pick_place", root: str | None = None
) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        root=root,
        robot_type="ur5e_hande",
        features={
            "observation.images.cam_exterior": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": IMAGE_NAMES,
            },
            "observation.images.cam_wrist": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": IMAGE_NAMES,
            },
            # Measured joint positions (rad) + measured finger opening (0 open .. 255 closed),
            # read before the frame's action is applied.
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": JOINT_NAMES,
            },
            # Absolute joint targets (rad) + gripper command (0 / 255), to be held for
            # one frame (SUBSAMPLE_STEPS physics steps).
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": JOINT_NAMES,
            },
        },
    )
