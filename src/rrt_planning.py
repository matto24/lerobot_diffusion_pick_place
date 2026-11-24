"""Collision-free joint-space planning for the UR5e with OMPL RRT*."""

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

from src.robot import is_q_valid

ou.setLogLevel(ou.LogLevel.LOG_ERROR)


class StateValidator:
    """OMPL validity check: a joint configuration is valid if the arm collides with nothing."""

    def __init__(self, d, m, num_joint):
        self.d = d
        self.m = m
        self.num_joint = num_joint

    def __call__(self, state):
        q_pose = [state[i] for i in range(self.num_joint)]
        return is_q_valid(d=self.d, m=self.m, q=q_pose)


def plan(d, m, start_q, goal_q):
    num_joint = 6

    # Joint space, each joint limited to [-pi, pi]
    space = ob.RealVectorStateSpace(num_joint)
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

