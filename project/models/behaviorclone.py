import torch
from torch import Tensor
from torch.nn import Module, Sequential, Linear, Softmax, Dropout, ReLU
import numpy as np

from typing import List, Union

from project.env import Action


class BehaviorCloning(Module):

    def __init__(self):
        super(BehaviorCloning, self).__init__()

        # Our model accepts an input observation of a
        # (18, 1) vector and outputs a (12,1) one hot
        # output; so we'll use softmax for the end
        # to represent a probability distribution of
        # the choice being correct.
        # self.model = Sequential(
        #     Linear(18, 32),
        #     ReLU(),
        #     Linear(32, 64),
        #     ReLU(),
        #     Dropout(0.2),
        #     Linear(64, 128),
        #     ReLU(),
        #     Dropout(0.2),
        #     Linear(128, 256),
        #     ReLU(),
        #     Dropout(0.2),
        #     Linear(256, 512),
        #     ReLU(),
        #     Dropout(0.2),
        #     Linear(512, 256),
        #     ReLU(),
        #     Dropout(0.2),
        #     Linear(256, 128),
        #     ReLU(),
        #     Dropout(0.2),
        #     Linear(128, 64),
        #     ReLU(),
        #     Dropout(0.2),
        #     Linear(64, 32),
        #     ReLU(),
        #     Linear(32, 12),
        #     Softmax(),
        # )
        self.model = Sequential(
            Linear(18, 256),
            ReLU(),
            Linear(256, 256),
            Dropout(0.1),
            ReLU(),
            Linear(256, 256),
            Dropout(0.1),
            ReLU(),
            Linear(256, 256),
            Dropout(0.1),
            ReLU(),
            Linear(256, 12),
            Softmax(),
        )

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

    def get_actions(
        self, observations: Union[np.ndarray, Tensor, List[np.ndarray]]
    ) -> List[Action]:
        # Ensure that observations are a numpy array and not a dict
        # or tensor
        if isinstance(observations, dict):
            observations = observations["observation"]
        elif isinstance(observations, Tensor):
            observations = observations.detach().numpy()

        # For behavior cloning, we put in an observation and get out
        # a sigmoid probability of each action being correct
        actions = self.forward(observations)

        action_objects = Action.all()
        action_scores = list(zip(action_objects, actions.detach().numpy()))

        # Sort the actions such that the highest probability is first, etc
        action_scores.sort(key=lambda action_score: action_score[1], reverse=True)

        d = [action for action, _ in action_scores]

        return d

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
