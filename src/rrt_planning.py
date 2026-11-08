import numpy as np
import spatialmath as sm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import csv

from ompl import base as ob
from ompl import geometric as og

from src.robot import *

matplotlib.use("agg")


class StateValidator:
    # This is mujoco specific, so I have implemented this for you
    def __init__(self, d, m, num_joint):
        self.d = d
        self.m = m
        self.num_joint = num_joint

    def __call__(self, state):
        # print("isStateValid - state: ", state)
        q_pose = [state[i] for i in range(self.num_joint)]
        return is_q_valid(d=self.d, m=self.m, q=q_pose)


def plan(d, m, start_q, goal_q):
    num_joint = 6

    space = ob.RealVectorStateSpace(
        num_joint
    )  # Create a joint-space vector instead of a 2D space as in the example
    bounds = ob.RealVectorBounds(num_joint)
    for i in range(num_joint):
        bounds.setLow(i, -np.pi)
        bounds.setHigh(i, np.pi)

    space.setBounds(bounds)

    ss = og.SimpleSetup(space)

    # Create a state validity checker object
    validator = StateValidator(d, m, num_joint)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(validator))

    # Set start and goal states
    start = ob.State(space)
    goal = ob.State(space)

    for i in range(num_joint):
        start[i] = start_q[i]
        goal[i] = goal_q[i]

    ss.setStartAndGoalStates(start, goal)

    # ===== EXPLICITLY SET PLANNER =====
    planner = og.RRTstar(ss.getSpaceInformation())

    ss.setPlanner(planner)
    solved = ss.solve(20.0)
    # solve() is also truthy for an APPROXIMATE solution: a path that ends
    # somewhere near, but not at, the goal. Following it leaves the robot
    # off-target, so only accept exact solutions.
    if solved and ss.haveExactSolutionPath():
        # Extract solution path
        path = ss.getSolutionPath()
        path.interpolate(2000)

        # Return the q-poses to solution_trajectory
        solution_trajectory = []
        for i in range(len(path.getStates())):
            state = path.getState(i)
            q_pose = np.array([state[i] for i in range(num_joint)])
            solution_trajectory.append(q_pose)

        return solution_trajectory


def move_to_point(robot, m, d, start_q, end_q):
    # Define grasping frames for object: box
    obj_frame = get_mjobj_frame(model=m, data=d, obj_name=end_q)  # Get body frame
    obj_frame = obj_frame * sm.SE3.Rx(-np.pi)
    obj_desired_q = robot.robot_ur5.ik_LM(Tep=obj_frame, q0=start_q)[0]
    trajectory = plan(d, m, start_q, obj_desired_q)
    if trajectory is None:
        print("No collision free trajectory found")
    else:
        for step in trajectory:
            robot.queue.append((step, None))

    return obj_desired_q


def program(d, m):
    # Define our robot object
    robot = UR5robot(data=d, model=m)
    dk = []

    replay = True
    if replay:
        # get q-poses from csv file
        with open("recordings/box_qpose_RRT.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                q_pose = np.array(row[:6], dtype=float)
                gripper_val = None if row[6] == "" else float(row[6])
                dk.append(robot.robot_ur5.fkine(q_pose))
                robot.queue.append((q_pose, gripper_val))

            positions = np.array([Q[:3, 3] for Q in dk])  # x, y, z positions
            fig = plt.figure()
            plt.plot(positions[:, 0], label="x")
            plt.plot(positions[:, 1], label="y")
            plt.plot(positions[:, 2], label="z")
            plt.xlabel("Step")
            plt.legend()
            plt.savefig("DK_box_rrt.svg", format="svg")

        return robot.queue

    start_q = robot.get_current_q()
    # BOX
    # start_q = move_to_point(robot, m, d, start_q, "pickup_point_box")
    # robot.set_gripper(value=200, t=1000)
    # start_q = move_to_point(robot, m, d, start_q, "drop_point_box")
    # robot.set_gripper(value=0, t=1000)

    # Cylinder
    # start_q = move_to_point(robot, m, d, start_q, "approach_point_cylinder")
    # start_q = move_to_point(robot, m, d, start_q, "pickup_point_cylinder")
    # robot.set_gripper(value=250, t=1000)
    # start_q = move_to_point(robot, m, d, start_q, "pre_drop_cylinder")
    # start_q = move_to_point(robot, m, d, start_q, "drop_point_cylinder")
    # robot.set_gripper(value=0, t=1000)

    # TBLOCK
    # start_q = move_to_point(robot, m, d, start_q, "pickup_point_tblock")
    # robot.set_gripper(value=200, t=1000)
    # start_q = move_to_point(robot, m, d, start_q, "drop_point_tblock")
    # robot.set_gripper(value=0, t=1000)

    # save q_pose to csv file
    # with open("recordings/tblock_qpose.csv", "w") as f:
    #     writer = csv.writer(f)
    #
    #     for q_pose, gripper_val in robot.queue:
    #         row = list(q_pose) + [gripper_val]
    #         writer.writerow(row)
    #
    # print("DONE")

    return robot.queue
