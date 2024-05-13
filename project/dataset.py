from __future__ import annotations

import os
from pickle import load
from random import choice
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from project.env import Action


class RobotDataset(Dataset):

    def __init__(
        self,
        directory: str,
        transforms=None,
        gamma: float = 0.95,
        success_only: bool = False,
        episodes: Optional[
            Tuple[int, List[np.ndarray], List[Action], List[float]]
        ] = None,
    ):
        """
        episodes is a tuple of the episode ID, observations, actions, and rewards
        for the episode. This is used to quickly load a dataset from memory manually
        instead of loading from disk. If it is provided, the directory is ignored
        and no loading happens unless manually triggered.
        """
        self.__directory = directory
        self.__transforms = transforms
        self.__gamma = gamma
        self.__success_only = success_only

        # Quickly load and scan through the episodes to load them
        # into memory. We're working with a small enough dataset
        # that we can do this all in memory.
        # __episodes maintains a dict, episode ID to a tuple of
        # # of steps and then the total reward (1 or 0)
        self.__episodes: Dict[int, Tuple[int, int]] = {}
        # __steps is every step in the dataset, which is a tuple of
        # observation, action, and reward
        self.__steps: List[Tuple[np.ndarray, Action, float]] = []

        if episodes is None:
            self.load_dataset()
        else:
            for episode_id, observations, actions, rewards in episodes:
                for i in range(len(observations)):
                    self.__steps.append((observations[i], actions[i], rewards[i]))
                self.__episodes[episode_id] = (
                    len(observations),
                    1 if rewards[-1] > 0 else 0,
                )

    def load_dataset(self):
        """
        Load the dataset from the directory provided. The dataset
        is derived from the dataset directory, wherein each episode
        is its own folder; we then read in the observations and
        actions pickles for each episode, which gives us N steps.
        """
        for episode in os.listdir(self.__directory):
            episode_id = int(episode)
            episode_path = os.path.join(self.__directory, episode)

            # Check to see if it is a success or not by checking for the
            # existence of a "success" file in the episode directory
            success = False
            if os.path.exists(os.path.join(episode_path, "success")):
                success = True

            # If we are only looking for successful episodes, and this
            # episode is not successful, skip it
            if self.__success_only and not success:
                continue

            # Load the observations pickle file
            observations = []
            with open(os.path.join(episode_path, "observations.pkl"), "rb") as f:
                observations = load(f)

            observations = [obs["observation"] for obs in observations]

            # Actions pickle file
            actions = []
            with open(os.path.join(episode_path, "actions.pkl"), "rb") as f:
                actions = load(f)

            if success:
                # The rewards is calculable as we know its a successful episode,
                # which uses a sparse reward of 1.0 on success
                rewards = [0] * len(observations)
                rewards[-1] = 1.0
                for i in range(len(observations) - 2, 0, -1):
                    rewards[i] = rewards[i + 1] * self.__gamma
            else:
                rewards = [0] * len(observations)

            self.__episodes[episode_id] = (len(observations), 1 if success else 0)

            for i in range(len(observations)):
                self.__steps.append((observations[i], actions[i], rewards[i]))

    def split(self, percentage: float) -> Tuple[RobotDataset, RobotDataset]:
        """
        split will take in a percentage and return two new RobotDatasets
        that are split based on the percentage provided. The first dataset
        will be the percentage provided, and the second dataset will be
        the remaining percentage.

        Splitting is done as close as possible based on total step count,
        but episodes are not split - thus the split is determine first by
        episode assignment, and then steps.
        """
        total_steps = int(len(self.__steps) * percentage)

        target_episodes = []
        target_steps = 0
        while target_steps < total_steps:
            episode_id = choice(list(self.__episodes.keys()))
            if episode_id in target_episodes:
                continue
            episode_steps, _ = self.__episodes[episode_id]
            target_episodes.append(episode_id)
            target_steps += episode_steps

        # Now that we have a set of target episodes, begin isolating their
        # steps from this dataset to pass into the next. They'll be passed
        # in via the episodes parameter:
        # episodes: Optional[
        #    Tuple[int, List[np.ndarray], List[Action], List[float]]
        # ]
        episodes_target = []
        episodes_other = []
        for episode_id in self.__episodes:
            episode_steps, _ = self.__episodes[episode_id]
            observations = []
            actions = []
            rewards = []

            # Calculate where the episode begins in our total
            # steps list
            start_index = sum(
                [e[0] for i, e in self.__episodes.items() if i < episode_id]
            )
            end_index = start_index + episode_steps

            for i in range(start_index, end_index):
                observations.append(self.__steps[i][0])
                actions.append(self.__steps[i][1])
                rewards.append(self.__steps[i][2])

            if episode_id in target_episodes:
                episodes_target.append((episode_id, observations, actions, rewards))
            else:
                episodes_other.append((episode_id, observations, actions, rewards))

        # Now that we have the episodes, we can create the two datasets
        dataset_a = RobotDataset(
            self.__directory,
            self.__transforms,
            self.__gamma,
            self.__success_only,
            episodes=episodes_target,
        )

        dataset_b = RobotDataset(
            self.__directory,
            self.__transforms,
            self.__gamma,
            self.__success_only,
            episodes=episodes_other,
        )

        return dataset_a, dataset_b

    def __len__(self) -> int:
        return len(self.__steps)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Action, float]:
        return self.__steps[idx]

    def info(self) -> str:
        """
        info will return a string representation of the dataset
        """
        successful_episodes = [e for e, v in self.__episodes.items() if v[1] > 0]
        successful_episode_steps = sum(
            [v[0] for _, v in self.__episodes.items() if v[1] > 0]
        )

        return f"""
Dataset Info:
- {len(self.__steps)} steps
- {len(self.__episodes)} episodes
- {len(successful_episodes)} successful episodes
- {len(self.__episodes) - len(successful_episodes)} failed episodes
Average steps per episode: {len(self.__steps) / len(self.__episodes):.1f}
Average steps per successful episode: {successful_episode_steps / len(successful_episodes):.1f}
        """
