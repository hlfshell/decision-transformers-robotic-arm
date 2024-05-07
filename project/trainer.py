from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from panda_gym.envs.core import RobotTaskEnv
from torch import Tensor
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader

from project.env import Action

from pickle import dump, load

from abc import ABC, abstractmethod


class RobotDataset(Dataset):

    def __init__(self, directory: str, transforms=None, gamma: float = 0.95):
        self.__directory = directory
        self.__transforms = transforms
        self.__gamma = gamma

        # Quickly load and scan through the episodes to load them
        # into memory. We're working with a small enough dataset
        # that we can do this all in memory.
        self.__episodes = []
        self.__steps: List[Tuple[np.ndarray, Action, float]] = []

        self.__load_dataset()

    def __load_dataset(self):
        """
        Load the dataset from the directory provided. The dataset
        is derived from the dataset directory, wherein each episode
        is its own folder; we then read in the observations and
        actions pickles for each episode, which gives us N steps.
        """
        for episode in os.listdir(self.__directory):
            episode_path = os.path.join(self.__directory, episode)

            # Load the observations pickle file
            observations = []
            with open(os.path.join(episode_path, "observations.pkl"), "rb") as f:
                observations = load(f)

            # Actions pickle file
            actions = []
            with open(os.path.join(episode_path, "actions.pkl"), "rb") as f:
                actions = load(f)

            # The rewards is calculable as we know its a successful episode,
            # which uses a sparse reward of 1.0 on success
            rewards = [
                1.0 / self.__gamma ** (len(observations) - i - 1)
                for i in range(len(observations))
            ]

            for i in range(len(observations)):
                self.__steps.append((observations[i], actions[i], rewards[i]))

    def __len__(self) -> int:
        return len(self.__steps)


class Trainer(ABC):

    def __init__(
        self,
        env: RobotTaskEnv,
        model: torch.nn.Module,
        dataset: Dataset,
        gamma: float = 0.95,
        alpha: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 50,
        episode_length_limit: int = 1000,
        checkpoints_folder: str = "checkpoints",
    ):
        super().__init__()

        self.env = env
        self.model = model
        self.dataset = dataset
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.checkpoints_folder = checkpoints_folder
        self.episode_length_limit = episode_length_limit
        self.epochs = epochs
        self.__current_epoch = 0

        if not os.path.exists(self.checkpoints_folder):
            os.makedirs(self.checkpoints_folder)

        self.loss = MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)

        self.rewards: List[List[float]] = []
        self.losses: List[float] = []

        self.__current_action = "Initializing"

        self.__dataset_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

    def __print_status(self):
        pass

    def create_plot(self, filepath: str):
        pass

    def save(self, checkpoint_name: str):
        """
        save will save the model, stats around progress, and any additional
        data to the checkpoints folder
        """
        save_file: Dict[str, Any] = {
            "current_epoch": self.__current_epoch,
            "rewards": self.rewards,
            "losses": self.losses,
        }

        # Save the save_file to a pickle
        with open(
            os.path.join(self.checkpoints_folder, f"{checkpoint_name}.pkl"), "wb"
        ) as f:
            dump(save_file, f)

        # Save the model
        torch.save(
            self.model.state_dict(),
            os.path.join(self.checkpoints_folder, f"{checkpoint_name}.pth"),
        )

    def load(self, checkpoint_name: str):
        """
        load will load the model, stats around progress, and any additional
        data from the checkpoints folder
        """
        # Load the save_file from a pickle
        with open(
            os.path.join(self.checkpoints_folder, f"{checkpoint_name}.pkl"), "rb"
        ) as f:
            save_file = load(f)

        self.__current_epoch = save_file["current_epoch"]
        self.rewards = save_file["rewards"]
        self.losses = save_file["losses"]

        # Load the model
        self.model.load_state_dict(
            torch.load(os.path.join(self.checkpoints_folder, f"{checkpoint_name}.pth"))
        )

    def run_episode(self) -> Tuple[float, int]:
        """
        run_episode will run a single episode of the environment with the current
        model and return the total reward and the steps
        """
        observation, _ = self.env.reset()

        for step in range(self.episode_length_limit):
            action = self.model(observation)
            observation, reward, terminated, _ = self.env.step(action)

            if terminated:
                break

        return reward, step

    def test_model_performance(self, runs: int = 100) -> Tuple[float, int, int, int]:
        """
        test_model_performance will run the model for a number of episodes and
        return the average reward and the number of successful episodes, and
        average steps taken, and average steps taken per success
        """
        # TODO - clear some kind of global tracker here
        successes = 0
        rewards = 0
        steps = 0
        for i in range(runs):
            self.__curent_action = f"Testing model - {i + 1}/{runs}"
            reward, steps = self.run_episode()
            rewards += reward
            if reward > 0:
                successes += 1
            steps += steps

        avg_reward = rewards / runs
        avg_steps = steps / runs
        avg_steps_per_success = 0 if successes == 0 else int(steps / successes)

        return avg_reward, successes, avg_steps, avg_steps_per_success

    @abstractmethod
    def training_step(
        self, observations: Tensor, actions: Tensor, rewards: Tensor
    ) -> float:
        """
        training_step is a single step of the training loop. It will take in a batch
        of observations, actions, and rewards, and return the loss for the batch
        """
        pass

    def train(self):
        """
        train is our primary training loop. If our trainer has been loaded from
        a checkpoint, it will resume training from where it left off. Otherwise
        we will instantiate and start training the model from scratch based on our
        dataset provided.

        At the end of each epoch, we will save the model to the checkpoints folder,
        then test performance of the model across a number of episodes.
        """

        while self.__current_epoch <= self.epochs:
            batch = 0
            self.__current_epoch += 1

            for observations, actions, rewards in self.__dataset_loader:
                batch += 1
                self.__current_action = f"Training epoch {self.__current_epoch}/{self.epochs} - Batch {batch}/{len(self.dataset)%self.batch_size}"
                self.__print_status()

                # Perform a training step
                loss = self.training_step(
                    Tensor(observations), Tensor(actions), Tensor(rewards)
                )

            # Validation - perform 100 episodes and observe performance of model
            rewards, successes, steps, success_steps = self.test_model_performance()

            self.__current_action = "Saving"
            self.__print_status()
            self.save(f"epoch_{self.__current_epoch}.pth")

        print("")
        print("Training complete!")
