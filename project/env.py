from __future__ import annotations

import math
from math import pi
from random import choice, uniform
from typing import Any, Dict, Optional, Tuple

import numpy as np
from panda_gym.envs.core import RobotTaskEnv, Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet


class SortTask(Task):
    def __init__(
        self,
        sim: PyBullet,
        robot: Panda,
    ):
        super().__init__(sim)
        self.robot = robot

        self.sim.create_plane(z_offset=-0.4)

        # Set up our goal
        self.__goal_position = np.array([-0.25, 0.00, 0.01])
        self.sim.create_box(
            body_name="goal",
            half_extents=[0.05, 0.1, 0.01],
            mass=0.0,
            ghost=False,
            position=self.__goal_position,
            rgba_color=[0.0, 1.0, 0.0, 0.4],
        )

        # Set up our blocker bar
        self.sim.create_box(
            body_name="blocker",
            half_extents=[0.01, 0.3, 0.005],
            mass=0.0,
            ghost=False,
            position=np.array([-0.2, 0.0, 0.01]),
            rgba_color=[0.0, 0.0, 0.0, 0.8],
        )

        # region_limits are the corners that we'll use for coordinates to
        # normalize positions by
        self.region_limits = [(-0.4, 0.4), (-0.4, 0.4), (0.01, 0.2)]

        # Track our target id
        self.target: Optional[Cube] = None

        # Create our table
        self.sim.create_table(length=0.8, width=0.8, height=0.4, x_offset=-0.3)

        # Set the limit the position of our object (if the object
        # goes beyond this limit, we will assume failure)
        self.object_limits = ((-0.06, 0.06), (-0.2, 0.2))

        self.score = 0.0
        self.reset()

    def reset_target_object(self):
        """
        Place the cube randomly somewhere in the starting zone
        """
        self.delete_cube()

        name = "target"

        # Generate our random position
        x = uniform(self.object_limits[0][0], self.object_limits[0][1])
        y = uniform(self.object_limits[1][0], self.object_limits[1][1])
        z = 0.01
        position = np.array([x, y, z])

        size = 0.03

        self.sim.create_box(
            body_name=name,
            half_extents=np.array([size / 2, size / 2, size / 2]),
            mass=0.25,
            ghost=False,
            position=position,
            rgba_color=np.array([0.984, 0.494, 0.945, 0.8]),
        )

        id = self.sim._bodies_idx[name]

        self.target = Cube(
            id=id,
            name=name,
            position=position,
            size=size,
        )
        self.goal = {}

    def delete_cube(self):
        if self.target:
            self.sim.physics_client.removeBody(self.target.id)
            self.target = None

    def reset(self):
        self.delete_cube()
        self.reset_target_object()
        self.score = 0.0

    def step(self) -> np.array:
        """
        step will determine if any objects collided and need to be removed,
        and adjust the score as expected.

        Returns an observation of the current state
        """
        if self.target:
            floor_id = self.sim._bodies_idx["plane"]
            if self.check_collision(floor_id, self.target.id):
                self.delete_cube()
                self.score -= 1.0
            goal_id = self.sim._bodies_idx["goal"]
        if self.target:
            if self.check_collision(goal_id, self.target.id):
                self.delete_cube()
                self.score += 1.0
                self.finish = True

        return self.get_obs()

    def check_collision(self, object1: str, object2: str) -> bool:
        """
        check_collision will check if either object1 or object2 are colliding
        and returns a boolean to that effect.
        """
        contacts = self.sim.physics_client.getContactPoints(object1, object2)
        return contacts is not None and len(contacts) > 0

    def get_cube_pose(self) -> np.array:
        """
        Get the pose of the cube
        """
        if self.target == None:
            return np.zeros(12, dtype=np.float32)

        object_position = self.sim.get_base_position(self.target.name)
        object_rotation = self.sim.get_base_rotation(self.target.name)
        object_velocity = self.sim.get_base_velocity(self.target.name)
        object_angular_velocity = self.sim.get_base_angular_velocity(self.target.name)
        observation = np.concatenate(
            [
                object_position,
                object_rotation,
                object_velocity,
                object_angular_velocity,
            ]
        )
        return observation.astype(np.float32)

    def get_obs(self) -> np.array:
        """
        Builds a state vector of the current state of the
        environment for the agent
        """
        observation = np.zeros(18, dtype=np.float32)

        # Robot state is in order:
        # end effector position (3)
        # end effector velocity (3)
        # fingers width (1)
        robot_state = self.robot.get_obs().astype(np.float32)

        # Convert our robot EE position into a normalized coordinate
        # with our region limits
        for i in range(3):
            coordinate = (robot_state[i] - self.region_limits[i][0]) / (
                self.region_limits[i][1] - self.region_limits[i][0]
            )
            if coordinate < 0:
                coordinate = 0
            if coordinate > 1:
                coordinate = 1
            observation[i] = coordinate

        observation[3:7] = robot_state[3:7]

        # Get the cube state;
        # cube position (3)
        # cube orientation (3)
        # cube velocity (3)
        # cube angular velocity (3)
        cube_state = self.get_cube_pose()

        # Convert our cube position into a normalized coordinate
        # with our region limits
        for i in range(3):
            coordinate = (cube_state[i] - self.region_limits[i][0]) / (
                self.region_limits[i][1] - self.region_limits[i][0]
            )
            if coordinate < 0:
                coordinate = 0
            if coordinate > 1:
                coordinate = 1
            observation[i + 7] = coordinate

        observation[9:18] = cube_state[3:12]

        # return np.concatenate([robot_state, cube_state])
        return observation

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any] = ...,
    ) -> np.ndarray:
        return np.array([self.score], dtype="float32")

    def is_success(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any] = ...,
    ) -> np.ndarray:
        return np.array(self.target is None and self.score != 0.0, dtype="bool")

    def get_achieved_goal(self) -> np.ndarray:
        return np.array(self.target is None and self.score != 0.0, dtype="bool")


class SortEnv(RobotTaskEnv):
    """
    Sort task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "human".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
    """

    def __init__(
        self,
        render_mode: str = "human",
        renderer: str = "OpenGL",
        render_width: int = 720,
        render_height: int = 480,
    ):
        sim = PyBullet(
            render_mode=render_mode,
            background_color=np.array([200, 200, 200]),
            renderer=renderer,
        )
        robot = Panda(
            sim,
            block_gripper=False,
            base_position=np.array([-0.6, 0.0, 0.0]),
            control_type="ee",
        )
        self.task = SortTask(sim, robot)
        super().__init__(
            robot=robot,
            task=self.task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=None,
            render_distance=0.9,
            render_yaw=45,
            render_pitch=-30,
            render_roll=0.0,
        )
        self.sim.place_visualizer(
            target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30
        )

        # Gripper closed maintained so we have constant force on
        # the gripper
        self.__gripper_closed = False

        self.reset()

    def reset(self) -> Tuple[np.array, Dict[str, Any]]:
        """
        Reset the environment.

        Returns:
            Tuple[np.array, Dict[str, Any]]: Observation and info.
        """
        self.__gripper_closed = False

        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        observation = self._get_obs()
        return observation, None

    def _get_obs(self) -> Dict[str, np.ndarray]:
        observation = self.task.step().astype(np.float32)
        achieved_goal = self.task.get_achieved_goal().astype(np.float32)

        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
        }

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action (Action): Action being performed

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Observation, reward, done, info.
        """
        score_prior = self.task.score

        # Track the gripper state
        if action.is_a(Action.OPEN_GRIPPER):
            self.__gripper_closed = False
        elif action.is_a(Action.CLOSE_GRIPPER):
            self.__gripper_closed = True

        action_output = action.to_panda_action()

        # Apply the gripper force if we are in "closed" state
        if self.__gripper_closed:
            action_output[3] = -0.5

        self.robot.set_action(action_output)
        self.sim.step()

        obs = self._get_obs()
        score_after = self.task.score
        terminated = self.task.is_success(obs["achieved_goal"], self.task.get_goal())
        info = {"is_success": terminated}
        reward = score_after - score_prior

        return obs, reward, terminated, False, info


class Action:
    """
    Action is a class handling the discretization of actions
    for the robot arm
    """

    NORTH = 0
    NORTHEAST = 1
    EAST = 2
    SOUTHEAST = 3
    SOUTH = 4
    SOUTHWEST = 5
    WEST = 6
    NORTHWEST = 7
    UP = 8
    DOWN = 9
    OPEN_GRIPPER = 10
    CLOSE_GRIPPER = 11

    def __init__(self, action: int):
        self.action = action

    @classmethod
    def Random(cls) -> Action:
        return Action(choice(range(12)))

    @classmethod
    def FromOneHot(cls, action: np.ndarray) -> Action:
        return Action(np.argmax(action))

    def one_hot(self) -> np.ndarray:
        action = np.zeros(12, dtype=np.float32)
        action[self.action] = 1.0
        return action

    def to_panda_action(self) -> np.ndarray:
        """
        Converts our action to a panda "ee" controlled
        robot action - a set of forces upon each
        """
        move = 0.5
        gripper = 0.5

        movement = np.zeros(4)
        # Note that the movements below are relative to the
        # application directions, and not the actual end effector
        # x y z frame
        # movement = [NORTH, EAST, SOUTH, WEST, RAISE, GRIPPER]

        if self.action == Action.NORTH:
            movement[0] = move
        if self.action == Action.NORTHEAST:
            movement[0] = move
            movement[1] = move
        if self.action == Action.EAST:
            movement[1] = move
        if self.action == Action.SOUTHEAST:
            movement[0] = move
            movement[1] = -move
        if self.action == Action.SOUTH:
            movement[0] = -move
        if self.action == Action.SOUTHWEST:
            movement[0] = -move
            movement[1] = -move
        if self.action == Action.WEST:
            movement[1] = -move
        if self.action == Action.NORTHWEST:
            movement[0] = move
            movement[1] = -move
        if self.action == Action.UP:
            movement[2] = move / 2
        if self.action == Action.DOWN:
            movement[2] = -move / 2
        if self.action == Action.OPEN_GRIPPER:
            movement[3] = gripper
        if self.action == Action.CLOSE_GRIPPER:
            movement[3] = -gripper

        return movement

    def is_a(self, action: int) -> bool:
        return self.action == action

    @classmethod
    def GetString(cls, action: int) -> str:
        if action == Action.NORTH:
            return "NORTH"
        if action == Action.NORTHEAST:
            return "NORTHEAST"
        if action == Action.EAST:
            return "EAST"
        if action == Action.SOUTHEAST:
            return "SOUTHEAST"
        if action == Action.SOUTH:
            return "SOUTH"
        if action == Action.SOUTHWEST:
            return "SOUTHWEST"
        if action == Action.WEST:
            return "WEST"
        if action == Action.NORTHWEST:
            return "NORTHWEST"
        if action == Action.UP:
            return "UP"
        if action == Action.DOWN:
            return "DOWN"
        if action == Action.OPEN_GRIPPER:
            return "OPEN_GRIPPER"
        if action == Action.CLOSE_GRIPPER:
            return "CLOSE_GRIPPER"

    def __str__(self):
        return f"{Action.GetString(self.action)}"


class Cube:
    """
    Cube tracks the lifecycle of a target object (not a goal)
    """

    def __init__(
        self,
        id: str,
        name: str,
        position: np.array,
        size: float,
        color: Tuple[int, int, int] = (255, 255, 255),
    ):
        self.id = id
        self.name = name
        self.position = position
        self.size = size
        self.color = color
