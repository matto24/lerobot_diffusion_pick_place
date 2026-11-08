# Vision-Guided Robotic Grasping in MuJoCo

Estimating the 6-DoF pose of an object from a point cloud and using it to plan and
execute a robotic pick-and-place with a simulated **UR5e + Robotiq Hand-E** in
[MuJoCo](https://mujoco.org/). This project combines **computer-vision pose estimation**
(point-cloud registration) with **robotics motion planning** (RRT-Connect and
trapezoidal trajectory generation).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Demo](media/clip_rrtconnect-ezgif.com-video-to-gif-converter.gif)

---

## What it does

A duck is dropped at a random position and orientation on a table. The robot must
find it and pick it up using only simulated sensor data:

1. **Perceive** — render a depth image from a wrist/overhead camera and back-project it
   into a 3D point cloud of the scene.
2. **Estimate pose** — register a known model of the object against the scene point cloud
   to recover its full 6-DoF pose (position + orientation).
3. **Plan the grasp** — from the estimated pose, compute an approach pose and a grasp
   pose along the object's long axis, then solve inverse kinematics for each.
4. **Execute** — drive the arm through the approach → grasp → lift → place sequence with
   a smooth trapezoidal (parabolic-blend) joint trajectory and close the gripper.

## Pipeline

```
 depth render ──► point cloud ──► preprocess ──► RANSAC (global) ──► ICP (local) ──► 6-DoF pose
   cam.py           cam.py         do_pe.py         do_pe.py           do_pe.py
                                                                          │
                                     grasp/approach pose  ◄───────────────┘
                                          project/cv_demo.py
                                                   │
                          IK + trapezoidal trajectory ──► execute on UR5e
                             robot.py / project/cv_demo.py
```

### Computer vision — pose estimation (`do_pe.py`, `cam.py`)
- **Point-cloud capture** (`cam.py`): `get_pointcloud()` renders a depth image and
  back-projects it using the camera intrinsics/extrinsics (`get_camera_pose_cv()`).
- **Preprocessing** (`do_pe.py`): spatial (pass-through) crop, RANSAC plane removal for
  the table, and statistical outlier removal.
- **Global alignment**: RANSAC over FPFH shape-feature correspondences to find a coarse
  pose (`RANSAC()`).
- **Local refinement**: point-to-point ICP to refine the alignment (`ICP()`).
- Entry point: `do_pe.do_pose_estimation(scene_cloud, object_cloud)` → 4×4 transform.

### Robotics — planning & control (`robot.py`, `project/`)
- `robot.py` — the `UR5robot` class (`move_l`, `move_j`, `move_j_via`, `set_gripper`,
  `get_current_q`) plus MuJoCo helpers and a collision/limit check (`is_q_valid`).
- `project/rrt_planning.py` — **OMPL RRT-Connect** planning in joint space with a MuJoCo
  state validity checker (used for obstacle-aware motion; see `scene_obstacles.xml`).
- `project/point_to_point.py` — trapezoidal (parabolic-blend) point-to-point trajectories.
- `project/cv_demo.py` — the full integrated demo: perception → pose estimation → grasp
  planning → trajectory execution. The final pick-and-place uses trapezoidal interpolation
  (rewritten from an earlier RRT-based version for smoother, faster motion).

### Results

| Executed joint trajectory | Trapezoidal velocity profile |
| --- | --- |
| ![trajectory](media/position_trajectory.png) | ![profile](media/trajectory_profile_parabolic.png) |

Recorded trajectories for both planners (RRT vs. trapezoidal, across several objects)
live in [`recordings/`](recordings/).

## Project structure

```
├── main.py                 # Entry point — launches the MuJoCo viewer and runs a program()
├── robot.py                # UR5robot control, IK, MuJoCo helpers, validity checks
├── cam.py                  # Camera: RGB / depth / point-cloud rendering
├── do_pe.py                # Pose estimation: preprocessing + RANSAC + ICP
├── project/                # ← this project
│   ├── cv_demo.py          # Full vision-guided grasp pipeline (the main demo)
│   ├── rrt_planning.py     # OMPL RRT-Connect path planning
│   ├── point_to_point.py   # Trapezoidal point-to-point trajectories
│   └── test_pcd.py         # Standalone pose-estimation test on a saved point cloud
├── scene*.xml              # MuJoCo scenes (scene_project_cv.xml is the grasp demo)
├── assets/                 # Robot, gripper, table and object (duck) models
├── recordings/             # Recorded joint trajectories (RRT + trapezoidal)
├── media/                  # Demo gif + result plots
├── docs/                   # Course reference material
└── exercises/              # University coursework — see below
```

> **`exercises/`** contains the course assignments completed alongside this project
> (kinematics, Jacobians, DMPs, trajectory generation, etc.). They are kept for
> reference and are **not** part of the project deliverable.

## Prerequisites

- **Python 3.10** (pinned in `.python-version`).
- **OMPL (Open Motion Planning Library)** — required for the RRT planner. It is provided
  here as the `ompl` Python wheel via the lockfile; if you install it yourself note it may
  need a system-level build. See the [OMPL install guide](https://ompl.kavrakilab.org/installation.html).
- [`uv`](https://docs.astral.sh/uv/) for dependency management.

## Setup

```bash
uv sync
```

## Running

The simulation is launched via `main.py`. On macOS, MuJoCo's passive viewer requires
`mjpython`:

```bash
uv run mjpython main.py
```

By default `main.py` loads `scene_project_cv.xml` and runs `project/cv_demo.py`. To run a
different program, change the import near the top of `main.py`, e.g.:

```python
# from project.point_to_point import program   # trapezoidal point-to-point
# from project.rrt_planning   import program    # RRT-Connect planning
from project.cv_demo import program             # full vision-guided grasp (default)
```

and pick the matching scene via `model_path` (e.g. `scene_obstacles.xml` for the RRT demo).

## License

MIT
