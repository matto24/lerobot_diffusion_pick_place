import numpy as np
import spatialmath as sm
import numpy as np
import matplotlib.pyplot as plt
import random

from ompl import base as ob
from ompl import geometric as og
from src.rrt_planning import plan

from src.robot import *

import numpy as np
import roboticstoolbox as rtb
from scipy.optimize import minimize
from spatialmath import SE3, SO3
import mujoco


# The scene uses the Robotiq Hand-E (assets/UR5eGripper/ur5e_hande.xml), a
# parallel gripper: the fingers slide straight in, so the box is held
# square. The Hand-E opens to 50 mm, hence the 40 mm boxes.

# Steps to hold the gripper command before moving on (0.002 s each)
GRIPPER_WAIT = 500
# Gripper command for a closed grasp (0 = open, 255 = fully closed)
GRIP_CLOSE = 255
# How far below the box centre the TCP goes when grasping (m). The Hand-E
# TCP is 11 mm above the fingertips, so at the centre of a 40 mm box the
# fingertips clear the table by ~9 mm.
GRASP_DEPTH = 0.0


def random_box_pos(m, d):
    selected_pos = []
    for i in range(1, 7):
        STOP = False
        while not STOP:
            rand_x = random.uniform(0.45 + 0.25, 0.75 - 0.25)
            rand_y = random.uniform(-0.55 + 0.25, 0.55 - 0.25)
            if len(selected_pos) == 0:
                STOP = True
                break

            SHOULD_STOP = True
            for pos in selected_pos:
                dist = np.linalg.norm(np.array([rand_x, rand_y]) - np.array(pos))
                if dist <= 0.15:
                    SHOULD_STOP = False
                    break

            if SHOULD_STOP:
                STOP = True

        z = m.geom(f"box{i}").size[2]  # half height: box resting on the table
        d.joint(f"box{i}").qpos[0:3] = [rand_x, rand_y, z + 0.01]
        print(f"box{i}: ", rand_x, rand_y)
        selected_pos.append([rand_x, rand_y])

    mujoco.mj_step(m, d)


def linear_path_is_continuous(robot, q_start, T0, T1, samples=10, max_step=0.3):
    """
    True if move_l from q_start along T0 -> T1 stays in one configuration.
    Near a joint limit, IK has to switch to another configuration part way,
    which shows up as a large jump between consecutive solutions.
    """
    q_prev = q_start
    for T in rtb.ctraj(T0=T0, T1=T1, t=samples):
        sol = robot.robot_ur5.ik_LM(Tep=T, q0=q_prev)
        q, success = sol.q, sol.success
        if not success or np.abs(q - q_prev).max() > max_step:
            return False
        q_prev = q
    return True


def solve_ik(robot, m, d, frame, q_near, follow=None, tries=50):
    """
    Distinct collision-free IK solutions for frame, closest to q_near first.

    A single ik_LM call can converge to any of the UR's 8 configurations,
    e.g. elbow-down with the arm in the table, and may fail outright. Retry
    from perturbed seeds, keep only converged, collision-free solutions
    within the planner's [-pi, pi] bounds. If follow is given, also require
    that the straight move from frame to follow (the descent to the object)
    works from that solution. Returns an empty list if there are none.
    """
    sols = []
    for k in range(tries):
        seed = q_near if k == 0 else q_near + np.random.uniform(-1, 1, 6)
        sol = robot.robot_ur5.ik_LM(Tep=frame, q0=seed)
        q, success = sol.q, sol.success
        if not success:
            continue
        q = (q + np.pi) % (2 * np.pi) - np.pi  # same pose, inside planner bounds
        if any(np.allclose(q, s, atol=0.05) for s in sols):
            continue  # same configuration as one we already have
        if not is_q_valid(d=d, m=m, q=q):
            continue
        if follow is not None and not linear_path_is_continuous(
            robot, q, frame, follow
        ):
            continue
        sols.append(q)
    return sorted(sols, key=lambda q: np.linalg.norm(q - q_near))


def smooth_timing(path):
    """
    Re-time a planned path so the arm speeds up and slows down smoothly.

    The planner's waypoints are evenly spaced, and main.py executes one per
    step, so the arm would go from standing still to full speed in a single
    step (and stop just as abruptly). That jolt makes a held box slide in
    the gripper. Same path and the same number of steps, but the progress
    along it follows a quintic (zero velocity and acceleration at both ends),
    like move_l's ctraj.
    """
    path = np.asarray(path)
    n = len(path)
    t = np.linspace(0, 1, n)
    s = (10 * t**3 - 15 * t**4 + 6 * t**5) * (n - 1)  # waypoint index, 0..n-1
    idx = np.arange(n)
    return [
        np.array([np.interp(si, idx, path[:, j]) for j in range(path.shape[1])])
        for si in s
    ]


def move_to_point(robot, m, d, start_q, obj_frame, follow=None, max_goals=3):
    """
    Plan a collision-free path to obj_frame and queue it. Returns the goal q,
    or None if no valid IK solution or path exists (nothing is queued then).
    follow: the pose move_l goes to next, see solve_ik.

    The closest IK solution in joint space is not always the easiest to
    reach: it can be a contorted pose that RRT* cannot find a way into
    within its time limit. If planning fails, try the next-closest solution.
    """
    goals = solve_ik(robot, m, d, obj_frame, start_q, follow=follow)
    if not goals:
        print("No collision-free IK solution found")
        return None

    for obj_desired_q in goals[:max_goals]:
        trajectory = plan(d, m, start_q, obj_desired_q)
        if trajectory is not None:
            for step in smooth_timing(trajectory):
                robot.queue.append((step, None))
            return obj_desired_q
        print("No collision free trajectory found, trying another IK solution")
    return None


def program(d, m):
    # Randomize box positions
    random_box_pos(m, d)

    # Define our robot object
    robot = UR5robot(data=d, model=m)

    red_boxes = ["box1", "box2", "box3"]
    green_boxes = ["box4", "box5"]
    blue_boxes = ["box6"]

    # EXERCISE: Create/implement a program that stacks the blocks on the dropzone
    # - Localize the blocks and define the grasping frames
    # - Path plan the pick and place trajectory for each block for stacking

    i = 0
    start_q = robot.get_current_q()
    for box in red_boxes:
        i += 1
        print("box nr: ", i)
        obj_frame = get_mjobj_frame(model=m, data=d, obj_name=box) * sm.SE3.Rx(-np.pi)
        # tool z points down here, so +z moves the TCP down into the box
        obj_frame = obj_frame * sm.SE3(0, 0, GRASP_DEPTH)
        # create an intermediate point 10cm above the object
        approach_frame = obj_frame * sm.SE3(0, 0, -0.1)

        # The moves below assume the robot reached the approach pose. If it
        # did not, stop here: queueing them anyway makes the arm jump.
        q = move_to_point(robot, m, d, start_q, approach_frame, follow=obj_frame)
        if q is None:
            print(f"Skipping {box}: cannot reach it")
            continue
        start_q = q
        robot.move_l(approach_frame, obj_frame, start_q, total_time=2)
        # hold 1 s (500 steps) so the fingers fully close before lifting
        robot.set_gripper(value=GRIP_CLOSE, t=GRIPPER_WAIT)
        start_q = robot.queue[-1][0]
        robot.move_l(obj_frame, approach_frame, start_q, total_time=2)
        start_q = robot.queue[-1][0]

        obj_frame = get_mjobj_frame(
            model=m, data=d, obj_name=f"drop_point_box{i}"
        ) * sm.SE3.Rx(-np.pi)
        # same offset as the grasp, so the box is put down at table height
        obj_frame = obj_frame * sm.SE3(0, 0, GRASP_DEPTH)
        approach_frame = obj_frame * sm.SE3(0, 0, -0.1)
        q = move_to_point(robot, m, d, start_q, approach_frame, follow=obj_frame)
        if q is None:
            # holding the box, so there is no safe way to continue
            print(f"Stopping: cannot reach drop_point_box{i} with {box}")
            break
        start_q = q
        robot.move_l(approach_frame, obj_frame, start_q, total_time=2)
        start_q = robot.queue[-1][0]
        # hold 1 s so the fingers fully open; after only 0.2 s the box
        # can still be stuck between the pads and lift with the gripper
        robot.set_gripper(value=0, t=GRIPPER_WAIT)
        robot.move_l(obj_frame, approach_frame, start_q, total_time=2)
        start_q = robot.queue[-1][0]

    return robot.queue
