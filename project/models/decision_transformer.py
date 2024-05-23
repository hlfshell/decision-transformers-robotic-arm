import torch
from torch import Tensor
from torch.nn import Module, LayerNorm, Linear, Embedding, Sequential

import numpy as np

from project.env import Action

from typing import List, Union

import transformers


class DecisionTransformer(Module):

    def __init__(self):
        super(DecisionTransformer, self).__init__()

        self.sequence_length = 120
        context_length = 20
        self.embedding_size = 128

        config = transformers.GPT2Config(
            vocab_size=1,
            state_dim=30,
            act_dim=12,
            max_length=context_length,
            max_ep_len=self.sequence_length,
            n_embd=self.embedding_size,
            n_layer=3,
            n_head=1,
            activation_function="relu",
            n_positions=1024,
            n_inner=4 * self.embedding_size,
            # Dropout:
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )

        self.transformer = transformers.GPT2Model(config)

        self.embed_timestep = Embedding(self.sequence_length, self.embedding_size)
        self.embed_return = Linear(1, self.embedding_size)
        self.embed_state = Linear(30, self.embedding_size)
        self.embed_action = Linear(12, self.embedding_size)

        self.embed_ln = LayerNorm(self.embedding_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(self.embedding_size, 30)
        self.predict_action = Sequential(Linear(self.embedding_size, 12))
        self.predict_return = torch.nn.Linear(self.embedding_size, 1)

        self.__prior_observations: np.ndarray = []
        self.__prior_actions: np.ndarray = []
        self.__prior_rewards: np.ndarray = []

    def reset(self):
        self.__prior_observations = []
        self.__prior_actions = []
        # Our rewards are init'ed with a zero tensor for the first bit
        self.__prior_rewards = [torch.zeros(0)]

    def forward(self, input: Union[np.ndarray, torch.Tensor]) -> Tensor:
        batch_size = len(observations)
        sequence_length = 1 + len(self.__prior_observations)

        # Combine prior sequences with current observations
        if len(self.__prior_observations) > 0:
            observations = np.concatenate(
                [self.__prior_observations, observations], axis=1
            )

        # Our timesteps is from 0 to n-1, where n is the number of observations
        timesteps = np.arange(sequence_length)

        # Embed each modality with a different head
        state_embeddings = self.embed_state(observations)
        action_embeddings = self.embed_action(self.__prior_actions)
        returns_embeddings = self.embed_return(self.__prior_rewards)
        time_embeddings = self.embed_timestep(timesteps)

        # Add the time embeddings to the state, action, and return embeddings
        # as if it was an equivalent to positional embeddings (they are)
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Stack our inputs to make the sequence follow the pattern of:
        # R_0, s_0, a_0, R_1, s_1, a_1, ...
        # We expect there to be a vector of size:
        # (batch_size, sequence_length, embedding_size)
        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * sequence_length, self.embe)
        )

        # Our stacked input is now passed to the trasnformer first through our
        # embedded layer
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Stack the attention mask to fit the inputs
        attention_mask = torch.ones((batch_size, self.sequence_length))
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * sequence_length)
        )

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x: Tensor = transformer_outputs["last_hidden_state"]

        # Reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token
        # for s_t
        x = x.reshape(batch_size, sequence_length, 3, self.embedding_size).permute(
            0, 2, 1, 3
        )

        # Predict the next return
        return_predictions = self.predict_return(x[:, 2])
        # Predict the next state
        state_predictions = self.predict_state(x[:, 2])
        # Finally predict the next action
        action_predictions = self.predict_action(x[:, 1])

        return action_predictions
