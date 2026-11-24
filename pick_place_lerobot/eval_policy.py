"""Evaluate a trained Diffusion Policy on the UR5e pick-and-place task in MuJoCo.

Each episode starts like a recorded expert episode: home pose, box at a random
position and yaw, green drop target at a random position. The policy then runs
closed loop at 25 Hz. An episode succeeds when the box rests on the table inside
the green drop disc with the gripper released, held for SUCCESS_HOLD_FRAMES.

Usage:
    # Headless, prints the success rate (any platform):
    uv run python pick_place_lerobot/eval_policy.py --num-episodes 20 --headless

    # With the MuJoCo viewer (macOS needs mjpython for the viewer):
    uv run mjpython pick_place_lerobot/eval_policy.py --num-episodes 5

    # From a local checkpoint instead of the Hub:
    uv run python pick_place_lerobot/eval_policy.py --pretrained-path /path/to/checkpoints/last/pretrained_model
"""

import argparse
import contextlib
import random
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from pick_place_lerobot.expert_episode import measured_gripper
from pick_place_lerobot.lerobot_dataset import (
    FPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SUBSAMPLE_STEPS,
)
from scripts.box_movement_example import get_random_drop_frame, random_box_pos
from src.cam import get_dual_camera_rgb
from src.paths import PROJECT_ROOT
from src.robot import hande_ctrl_qpos, ur_ctrl_qpos, ur_get_qpos, ur_set_qpos

HOME_Q = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])

# Success: box centre within the green disc (radius 0.06 m in scene.xml), resting
# on the table, gripper commanded open, for half a second in a row.
DROP_RADIUS = 0.06
SUCCESS_HOLD_FRAMES = FPS // 2


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy in MuJoCo.")
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default="MattoTT/diffusion_ur5e_pick_place_v3",
        help="HuggingFace repo ID or local path to a pretrained_model folder.",
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=700,
        help="Max policy steps (25 Hz) per episode. Expert demos are 450-500 steps.",
    )
    parser.add_argument(
        "--seed", type=int, default=1000, help="Episode i uses seed + i."
    )
    parser.add_argument(
        "--headless", action="store_true", help="No viewer, run as fast as possible."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    return parser.parse_args()


def to_policy_image(frame: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 render -> (3, H, W) float in [0, 1], like the training data.

    Frames are rendered at the recording size (IMAGE_HEIGHT x IMAGE_WIDTH), so no resize.
    """
    return torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0


def reset_episode(m, d, seed: int) -> np.ndarray:
    """Reset like generate_single_episode does. Returns the drop target xy."""
    random.seed(seed)
    np.random.seed(seed)
    mujoco.mj_resetData(m, d)
    ur_set_qpos(d, HOME_Q)
    hande_ctrl_qpos(d, 0)
    mujoco.mj_forward(m, d)
    box_xy = random_box_pos(m, d)
    drop_frame = get_random_drop_frame(m, d, box_xy=box_xy)
    return np.array(drop_frame.t[:2])


def box_placed(m, d, target_xy: np.ndarray, gripper_cmd: float) -> tuple[bool, float]:
    box_pos = d.body("box").xpos
    dist = float(np.linalg.norm(box_pos[:2] - target_xy))
    resting = box_pos[2] < m.geom("box").size[2] + 0.005
    released = gripper_cmd < 128
    return dist < DROP_RADIUS and resting and released, dist


def run_episode(
    args, m, d, renderer, policy, preprocessor, postprocessor, seed, viewer
):
    target_xy = reset_episode(m, d, seed)
    policy.reset()
    held = 0
    gripper_target = 0.0
    prev_q = ur_get_qpos(d, m).copy()

    for step_idx in range(args.max_steps):
        if viewer is not None and not viewer.is_running():
            raise KeyboardInterrupt

        frames = get_dual_camera_rgb(m, d, renderer)
        state = np.append(ur_get_qpos(d, m), measured_gripper(m, d)).astype(np.float32)
        obs = {
            "observation.images.cam_exterior": to_policy_image(
                frames["cam_exterior"]
            ).unsqueeze(0),
            "observation.images.cam_wrist": to_policy_image(
                frames["cam_wrist"]
            ).unsqueeze(0),
            "observation.state": torch.from_numpy(state).unsqueeze(0),
        }

        with torch.inference_mode():
            action = policy.select_action(preprocessor(obs))
            action = postprocessor(action)

        action_np = action.squeeze(0).cpu().numpy()
        q_target, gripper_target = action_np[:6], float(action_np[6])

        # The action is where the arm should be at the end of this frame. Ramp
        # the target there from the previous action over the frame's physics
        # steps, like the expert's per-step commands. Holding it constant
        # instead makes the position servo jump every 40 ms, and that shaking
        # slides the box out of the gripper during the carry.
        for i in range(SUBSAMPLE_STEPS):
            ur_ctrl_qpos(d, prev_q + (q_target - prev_q) * (i + 1) / SUBSAMPLE_STEPS)
            hande_ctrl_qpos(d, gripper_target)
            mujoco.mj_step(m, d)
        prev_q = q_target
        if viewer is not None:
            viewer.sync()
            time.sleep(1.0 / FPS)

        ok, dist = box_placed(m, d, target_xy, gripper_target)
        held = held + 1 if ok else 0
        if held >= SUCCESS_HOLD_FRAMES:
            return True, step_idx + 1, dist

    return False, args.max_steps, box_placed(m, d, target_xy, gripper_target)[1]


def main():
    args = parse_args()
    print(f"Loading policy from {args.pretrained_path} on {args.device}...")

    policy = DiffusionPolicy.from_pretrained(args.pretrained_path)
    policy.config.device = args.device
    policy.to(args.device)
    policy.eval()

    # Normalization lives in these pipelines, not in the policy: the preprocessor
    # normalizes state/images with the training stats, the postprocessor maps the
    # policy output back to radians and gripper 0..255.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    m = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "assets/scenes/scene.xml"))
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, IMAGE_HEIGHT, IMAGE_WIDTH)

    viewer_ctx = (
        contextlib.nullcontext()
        if args.headless
        else mujoco.viewer.launch_passive(m, d)
    )
    results = []
    try:
        with viewer_ctx as viewer:
            for ep in range(args.num_episodes):
                seed = args.seed + ep
                t0 = time.time()
                ok, steps, dist = run_episode(
                    args,
                    m,
                    d,
                    renderer,
                    policy,
                    preprocessor,
                    postprocessor,
                    seed,
                    viewer,
                )
                results.append(ok)
                print(
                    f"Episode {ep + 1}/{args.num_episodes} (seed {seed}): "
                    f"{'SUCCESS' if ok else 'fail'}  steps={steps}  "
                    f"box-to-target={dist * 100:.1f} cm  ({time.time() - t0:.0f} s)"
                )
    except KeyboardInterrupt:
        print("Stopped early.")

    if results:
        print(
            f"\nSuccess rate: {sum(results)}/{len(results)} = {100 * np.mean(results):.0f}%"
        )


if __name__ == "__main__":
    main()
