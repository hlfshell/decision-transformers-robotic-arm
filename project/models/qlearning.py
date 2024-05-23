import torch
from torch import Tensor
from torch.nn import Module, Sequential, Linear, LeakyReLU, Sigmoid, Dropout
import numpy as np

from typing import List, Union

from project.env import Action


class QLearning(Module):

    def __init__(self):
        super(QLearning, self).__init__()

        # Our model accepts an input observation of a
        # (18, 1) vector and action vector (12,1) one
        # hot encoded actions, outputting a single
        # value representing the Q score for that
        # action at that state, a score of 0 to 1.
        # self.model = Sequential(
        #     # Linear(30, 512),
        #     # LeakyReLU(),
        #     # Linear(512, 256),
        #     # LeakyReLU(),
        #     # Linear(256, 128),
        #     # LeakyReLU(),
        #     # Linear(128, 64),
        #     # LeakyReLU(),
        #     Linear(30, 64),
        #     LeakyReLU(),
        #     Linear(64, 32),
        #     LeakyReLU(),
        #     Linear(32, 1),
        #     Sigmoid(),
        # )
        # self.model = Sequential(
        #     Linear(30, 64),
        #     LeakyReLU(),
        #     Linear(64, 128),
        #     Dropout(0.2),
        #     LeakyReLU(),
        #     Linear(128, 256),
        #     Dropout(0.2),
        #     LeakyReLU(),
        #     Linear(256, 128),
        #     Dropout(0.2),
        #     LeakyReLU(),
        #     Linear(128, 64),
        #     Dropout(0.2),
        #     LeakyReLU(),
        #     Linear(64, 32),
        #     LeakyReLU(),
        #     Dropout(0.2),
        #     Linear(32, 1),
        #     Sigmoid(),
        # )
        self.model = Sequential(
            Linear(30, 256),
            LeakyReLU(),
            Linear(256, 256),
            Dropout(0.1),
            LeakyReLU(),
            Linear(256, 256),
            Dropout(0.1),
            LeakyReLU(),
            Linear(256, 256),
            Dropout(0.1),
            LeakyReLU(),
            Linear(256, 256),
            Dropout(0.1),
            LeakyReLU(),
            Linear(256, 256),
            Dropout(0.1),
            LeakyReLU(),
            Linear(256, 256),
            Dropout(0.1),
            LeakyReLU(),
            Linear(256, 1),
            Sigmoid(),
        )

    def get_actions(
        self, observations: Union[np.ndarray, Tensor, List[np.ndarray]]
    ) -> List[Action]:
        # Ensure that observations are a numpy array and not a dict
        # or tensor
        if isinstance(observations, dict):
            observations = observations["observation"]
        elif isinstance(observations, Tensor):
            observations = observations.detach().numpy()

        # For Q learning, generate an array of arrays that are our observation plus a
        # chosen action one-hot represented
        action_objects = Action.all()
        action_onehots = [a.one_hot() for a in action_objects]
        actions = np.array(action_onehots)

        # One hot encoded actions + state = 18 + 12 for 30 dimensions on our input
        # vector. We are generating the Q score for each possible action at this timestep
        # and thus
        input = np.zeros((12, 30))
        for i in range(12):
            input[i] = np.concatenate([observations, actions[i]])

        output = self.forward(input)

        # Convert the output array to a list of scores for each corresponding action
        scores = output.detach().numpy()

        # Sort the action_objects by the scores
        action_scores = list(zip(action_objects, scores))
        action_scores.sort(
            key=lambda action_score_pair: action_score_pair[1], reverse=True
        )

        # Return the actions in order of highest score to lowest
        return [action for action, _ in action_scores]

    def forward(self, input: Union[np.ndarray, Tensor, List[np.ndarray]]) -> Tensor:
        # Convert our input to a tensor if it's not already for safety
        if isinstance(input, np.ndarray):
            input_tensor: Tensor = torch.from_numpy(input.astype("float32"))
        elif type(input) is list:
            input_tensor: Tensor = torch.from_numpy(np.array(input).astype("float32"))
        elif type(input) is dict:
            input_tensor: Tensor = torch.from_numpy(
                input["observation"].astype("float32")
            )
        else:
            input_tensor = input

        output = self.model(input_tensor)

        return output

    def save(self, filepath: str):
        torch.save(
            {
                "model": self.model.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str):
        data = torch.load(filepath)
        self.model.load_state_dict(data["model"])
