# UR5e + Robotiq Hand-E (MJCF)

`ur5e_hande.xml` is the robot used in every scene: a UR5e arm with a Robotiq Hand-E
parallel gripper and a wrist camera.

- The UR5e model and meshes come from
  [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e),
  which derives them from the ROS-Industrial URDF. They are BSD-3-Clause licensed; see
  [`LICENSE_UR5E`](LICENSE_UR5E).
- The Hand-E meshes (`hande.stl`, `coupler.stl`, `finger_*-v3.0.stl`) model the
  [Robotiq Hand-E](https://robotiq.com/products/hand-e-adaptive-robot-gripper) gripper.
