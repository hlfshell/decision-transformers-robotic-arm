from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pickle import load
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from panda_gym.envs.core import RobotTaskEnv
from torch import Tensor
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Dataset

from project.env import Action


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
        validation_dataset: Optional[Dataset] = None,
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
        self.__validation_dataset = validation_dataset

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
        if self.__validation_dataset is not None:
            self.__validation_loader = DataLoader(
                self.__validation_dataset, batch_size=self.batch_size, shuffle=False
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
        # save_file: Dict[str, Any] = {
        #     "current_epoch": self.__current_epoch,
        #     "rewards": self.rewards,
        #     "losses": self.losses,
        # }

        # Save the save_file to a pickle
        # with open(
        #     os.path.join(self.checkpoints_folder, f"{checkpoint_name}.pkl"), "wb"
        # ) as f:
        #     dump(save_file, f)

        # Save the model
        # torch.save(
        #     self.model.state_dict(),
        #     os.path.join(self.checkpoints_folder, f"{checkpoint_name}.pth"),
        # )
        self.model.save(os.path.join(self.checkpoints_folder, f"{checkpoint_name}.pth"))

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
            action = self.get_action_from_model(observation["observation"])
            observation, reward, terminated, _, _ = self.env.step(action)

            if terminated:
                break

        return reward, step

    def test_model_performance(self, runs: int = 1) -> Tuple[float, int, int, int]:
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

    def train(self):
        """
        train is our primary training loop. If our trainer has been loaded from
        a checkpoint, it will resume training from where it left off. Otherwise
        we will instantiate and start training the model from scratch based on our
        dataset provided.

        At the end of each epoch, we will save the model to the checkpoints folder,
        then test performance of the model across a number of episodes.
        """

        while self.__current_epoch < self.epochs:
            batch = 0
            self.__current_epoch += 1

            batch_losses = []
            for observations, actions, rewards in self.__dataset_loader:
                observations = observations.float()
                actions = actions.float()
                rewards = rewards.float()

                batch += 1
                loss = 0.0
                self.__current_action = f"Training epoch {self.__current_epoch}/{self.epochs} - Batch {batch}/{len(self.dataset)%self.batch_size}"
                # print(f"EPOCH: {self.__current_epoch} - BATCH: {batch+1} - LOSS: {loss}", end="\r")
                self.__print_status()

                # Perform a training step
                loss = self.training_step(
                    Tensor(observations), Tensor(actions), Tensor(rewards)
                )
                batch_losses.append(loss.item())
                print(
                    f"Epoch {self.__current_epoch}/{self.epochs} - Batch {batch}/{int(len(self.dataset)/self.batch_size)} - Loss: {loss.item()} - Batch Loss Avg: {sum(batch_losses)/len(batch_losses)}",
                    end="\r",
                )
                self.losses.append(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print()
            print(
                f"Average batch loss for epoch: {sum(batch_losses)/len(batch_losses)}"
            )

            # Validation - perform 100 episodes and observe performance of model
            # rewards, successes, steps, success_steps = self.test_model_performance()
            if self.__validation_dataset is not None:
                validation_losses = []
                with torch.no_grad():
                    for observations, actions, rewards in self.__validation_loader:
                        observations = observations.float()
                        actions = actions.float()
                        rewards = rewards.float()

                        loss = self.training_step(
                            Tensor(observations), Tensor(actions), Tensor(rewards)
                        )
                        validation_losses.append(loss.item())
                    print()
                    print(
                        f"Validation Loss Average: {sum(validation_losses)/len(validation_losses)}"
                    )

            self.__current_action = "Saving"
            self.__print_status()
            self.save(f"epoch_{self.__current_epoch}.pth")

        print("")
        print("Training complete!")

    @abstractmethod
    def training_step(
        self, observations: Tensor, actions: Tensor, rewards: Tensor
    ) -> Tensor:
        """
        training_step is a single step of the training loop. It will take in a batch
        of observations, actions, and rewards, and return the loss for the batch
        """
        pass

    @abstractmethod
    def get_action_from_model(self, observation: np.ndarray) -> np.ndarray:
        """
        get_action_from_model will take in an observation and return the action
        predicted by the model
        """
        pass


class BehavioralCloningTrainer(Trainer):

    def training_step(
        self, observations: Tensor, actions: Tensor, rewards: Tensor
    ) -> Tensor:
        """
        training_step is a single step of the training loop. It will take in a batch
        of observations, actions, and rewards, and return the loss for the batch
        """
        predictions = self.model(observations)
        loss = self.loss(predictions, actions)
        return loss

    def get_action_from_model(self, observation: np.ndarray) -> np.ndarray:
        """
        get_action_from_model will take in an observation and return the action
        predicted by the model
        """
        observation_tensor = torch.tensor(observation, dtype=torch.float32)
        action_raw = self.model(observation_tensor).detach().numpy()
        action_onehot = np.zeros_like(action_raw)
        action_onehot[np.argmax(action_raw)] = 1

        return Action.FromOneHot(action_onehot)


class QLearningTrainer(Trainer):

    def training_step(
        self, observations: Tensor, actions: Tensor, rewards: Tensor
    ) -> Tensor:
        """
        training_step is a single step of the training loop. It will take in a batch
        of observations, actions, and rewards, and return the loss for the batch
        """
        # Our input consists of the 18 observation dimensions and the 12 action
        # dimensions one-hot represented, making a 30 dimensional input vector
        input = torch.cat([observations, actions], dim=1)

        predictions = self.model(input)
        loss = self.loss(predictions, rewards)
        return loss

    def get_action_from_model(self, observation: np.ndarray) -> np.ndarray:
        """
        get_action_from_model will take in an observation and return the action
        predicted by the model
        """
        return self.model.get_actions(observation)[0].one_hot()
