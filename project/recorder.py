import os
import numpy as np
from pickle import dump, load
from panda_gym.envs.core import RobotTaskEnv, Task
from project.env import Action
from typing import List, Tuple

from PIL import Image


class Recorder:
    """
    Recorder is meant to handle recording the current state, returned observation,
    and possibly the image for a given environment and step. It is agnostic to
    the control methods being used. Its goal is to handle file management and
    loading for replay
    """

    def __init__(self, target_directory: str, save_images: bool = False, fps: int = 24):
        self.__target_directory = target_directory

        self.__observations: List[np.ndarray] = []
        self.__actions: List[np.ndarray] = []

        self.__save_images = save_images
        if self.__save_images:
            self.__images: List[np.ndarray] = []
            self.__image_directory = os.path.join(self.__target_directory, "images")

        self.__fps = fps

    def record(
        self, step: int, observation: np.ndarray, action: Action, env: RobotTaskEnv
    ) -> None:
        """
        Records into memory the current state, observation, and action
        """
        # Convert the action to a one hot vector for easier storage
        action_vector = action.one_hot()

        self.__observations.append(observation)
        self.__actions.append(action_vector)

        if self.__save_images:
            old_render_mode = env.sim.render_mode
            env.sim.render_mode = "rgb_array"
            x = env.render()
            img = Image.fromarray(x)
            env.sim.render_mode = old_render_mode
            self.__images.append(img)

    def save(self, success: bool) -> None:
        """
        Saves the recorded data to disk
        """
        # Ensure the target directories exist
        os.makedirs(self.__target_directory, exist_ok=True)
        if self.__save_images:
            os.makedirs(self.__image_directory, exist_ok=True)

        with open(os.path.join(self.__target_directory, "observations.pkl"), "wb") as f:
            dump(self.__observations, f)

        with open(os.path.join(self.__target_directory, "actions.pkl"), "wb") as f:
            dump(self.__actions, f)

        # If the mission is a success, make a blank file called success
        if success:
            with open(os.path.join(self.__target_directory, "success"), "w") as f:
                pass

        if self.__save_images:
            for i, image in enumerate(self.__images):
                image.save(os.path.join(self.__image_directory, f"{i}.png"))

            frame_delay = 1.0 / self.__fps

            Image.new("RGB", self.__images[0].size).save(
                fp=os.path.join(self.__target_directory, "playback.gif"),
                format="GIF",
                append_images=self.__images,
                save_all=True,
                duration=frame_delay * len(self.__images),
                loop=0,
            )

    def Load(cls, filepath: str) -> Tuple[List[np.ndarray], List[Action]]:
        """
        Load from the disk the observations and actions as would have been
        recorded by an instantiated Recorder
        """
        with open(os.path.join(filepath, "observations.pkl"), "rb") as f:
            observations = load(f)

        with open(os.path.join(filepath, "actions.pkl"), "rb") as f:
            actions_onehot = load(f)

        actions: List[Action] = []
        for action_onehot in actions_onehot:
            actions.append(Action.FromOneHot(action_onehot))

        return observations, actions
