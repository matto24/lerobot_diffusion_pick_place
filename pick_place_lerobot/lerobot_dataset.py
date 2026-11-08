from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 25  # Policy control frequency (Hz)
SIM_DT = 0.002  # MuJoCo physics timestep (500 Hz)
SUBSAMPLE_STEPS = int(1.0 / (FPS * SIM_DT))  # Record every 20 physics steps


def create_dataset(repo_id: str = "local/ur5e_pick_place") -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features={
            "observation.images.cam_exterior": {
                "dtype": "image",
                "shape": (480, 640, 3),
            },
            "observation.images.cam_wrist": {"dtype": "image", "shape": (480, 640, 3)},
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
            },  # 6 joints + 1 gripper
            "action": {
                "dtype": "float32",
                "shape": (7,),
            },  # 6 target joints + 1 target gripper
        },
    )
