import numpy as np
import spatialmath as sm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import csv

from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

ou.setLogLevel(ou.LogLevel.LOG_ERROR)

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
    obj_desired_q = robot.robot_ur5.ik_LM(Tep=obj_frame, q0=start_q).q
    trajectory = plan(d, m, start_q, obj_desired_q)
    if trajectory is None:
        print("No collision free trajectory found")
    else:
        for step in trajectory:
            robot.queue.append((step, None))

    return obj_desired_q
