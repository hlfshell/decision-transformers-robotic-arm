from project.trainer import Trainer
from project.dataset import RobotDataset
from project.env import Action, SortEnv
from project.models.qlearning import QLearning

import numpy as np

from typing import Optional, Tuple, List

import random


class QDataset(RobotDataset):

    def __init__(
        self,
        directory: str,
        env: SortEnv,
        model: QLearning,
        transforms=None,
        gamma: float = 0.95,
        epsilon: float = 0.05,
        steps_per_epoch: int = 10_000,
        max_steps_per_episode: int = 120,
        success_only: bool = False,
        episodes: Optional[
            Tuple[int, List[np.ndarray], List[Action], List[float]]
        ] = None,
    ):
        super().__init__(
            directory=directory,
            transforms=transforms,
            gamma=gamma,
            success_only=success_only,
            episodes=episodes,
        )

        self.__model = model
        self.__env = env

        self.epsilon = epsilon
        self.steps_per_epoch = steps_per_epoch
        self.max_steps_per_episode = max_steps_per_episode

        self.__generated_steps: List[Tuple[np.ndarray, Action, float]] = []

    def generate_steps(self):
        self.__generated_steps = []
        successful = 0
        timeout = 0
        early_termination = 0

        while len(self.__generated_steps) < self.steps_per_epoch:
            data = self.rollout()
            if data[-1][2] > 0:
                successful += 1
            elif len(data) >= self.max_steps_per_episode:
                timeout += 1
            else:
                early_termination += 1

            self.__generated_steps.extend(data)
            print(
                f"Generating steps: {len(self.__generated_steps)}/{self.steps_per_epoch} - Successful: {successful} Timeout: {timeout} Early Termination: {early_termination}",
                end="\r",
            )
        print()

    def rollout(self) -> List[Tuple[np.ndarray, Action, float]]:

        steps = 0
        success = False

        sars: List[Tuple[np.ndarray, Action, float]] = []

        observation, _ = self.__env.reset()

        while True:
            steps += 1

            # Determine if epislon-greedy exploration triggers
            if np.random.rand() <= self.epsilon:
                action = random.choice(Action.all())
            else:
                actions = self.__model.get_actions(observation)
                action = actions[0]

            sars.append((observation["observation"], action.one_hot(), 0.0))

            observation, reward, terminated, _, _ = self.__env.step(action)

            if terminated or steps >= self.max_steps_per_episode:
                if reward > 0:
                    success = True
                break

        # If we have a successful episode, we need to calculate the rewards
        # back from 1.0 at the terminal state
        if success:
            sars[-1][2] = 1.0
            for i in range(len(sars) - 2, 0, -1):
                sars[i][2] = sars[i + 1][2] * self.gamma

        return sars

    def __len__(self) -> int:
        return len(self._steps) + len(self.__generated_steps)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Action, float]:
        if idx < len(self._steps):
            return self._steps[idx]
        else:
            return self.__generated_steps[idx - len(self._steps)]
