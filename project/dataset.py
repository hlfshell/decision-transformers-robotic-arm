import os
import sqlite3
from pickle import load
from typing import Dict, List, Tuple
from uuid import uuid4 as uuid

import numpy as np

from project.env import Action


class Step:

    def __init__(
        self,
        episode_id: str,
        observation: np.ndarray,
        action: Action,
        reward: float,
    ):
        self.episode_id = episode_id
        self.observation = observation
        self.action = action
        self.reward = reward

    def to_sql(self) -> Tuple:
        """
        Return the step as a tuple for easy SQL insertion

        In order:
            - episode id
            - reward
            - observation for 18 columns
            - action float (index of one hot)
        """

        return (
            self.episode_id,
            self.reward,
            *self.observation,
            np.argmax(self.action.one_hot()),
        )


class DatasetDB:
    """
    DatasetDB is a class that handles a SQLite connection to a
    database to manage our dataset.
    """

    def __init__(self, sqlite_file: str):
        self.sqlite_file = sqlite_file
        self.conn = sqlite3.connect(self.sqlite_file)

    def write_episode(
        self, observations: List[np.ndarray], actions: List[Action], gamma: float = 0.95
    ) -> None:
        """
        write_episode writes a new episode to the database for each step
        """
        # Calculate the total reward for each step
        reward = 1.0
        steps = len(observations)
        # Bellman backup the reward score - we use 1.0 for complete,
        # and no other reward is possible, so we can just simplify
        # it to be based on the time step index
        scores = [reward / gamma ** (steps - i - 1) for i in range(steps)]

        # Prepare the step to be written
        episode_id = uuid()
        steps = [
            Step(episode_id, observations[i], actions[i], scores[i])
            for i in range(steps)
        ]

        # Write the steps to the database
        for step in steps:
            self.write_step(step)

    def write_step(self, step: Step) -> None:
        """
        write_step writes a new step to the database

        The step table has the following columns:
            - episode_id: str
            - reward: float
            - obs_0 -> obs_1: 18 floats
            - action: int
        """
        sql = """
            INSERT INTO steps
            (
                episode_id,
                reward,
                obs_0,
                obs_1,
                obs_2,
                obs_3,
                obs_4,
                obs_5,
                obs_6,
                obs_7,
                obs_8,
                obs_9,
                obs_10,
                obs_11,
                obs_12,
                obs_13,
                obs_14,
                obs_15,
                obs_16,
                obs_17,
                action
            )
            VALUES(
        """

        for i in range(20):
            if i > 0:
                sql += ", "
            sql += "?"
        sql += ")"

        cursor = self.conn.cursor()
        cursor.execute(sql, step.to_sql())
        self.conn.commit()

    def get_episode(self, episode_id: str) -> List[Step]:
        """
        get_episode retrieves all steps for a given episode
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT
                episode_id,
                reward,
                obs_0,
                obs_1,
                obs_2,
                obs_3,
                obs_4,
                obs_5,
                obs_6,
                obs_7,
                obs_8,
                obs_9,
                obs_10,
                obs_11,
                obs_12,
                obs_13,
                obs_14,
                obs_15,
                obs_16,
                obs_17,
                action
            FROM steps WHERE episode_id = ?
            """,
            (episode_id,),
        )

        steps: List[Step] = []
        for row in cursor.fetchall():
            action = np.zeros(9)
            action[row[20]] = 1.0

            steps.append(
                Step(
                    row[0],
                    np.array(row[2:20]),
                    Action.FromOneHot(action),
                    row[1],
                )
            )
        return steps

    def load_from_folder(self, folder: str) -> None:
        """
        load_from_folder creates a new DatasetDB instance from
        a folder. Typically used to instantiate our database
        from recorded episodes.
        """
        # Each folder is a subset of folders; for each folder in them,
        # we want to load the observations and actions pickle files in
        # each
        for root, dirs, files in os.walk(folder):
            for d in dirs:
                actions_pickle_file = os.path.join(root, d, "actions.pkl")
                observations_pickle_file = os.path.join(root, d, "observations.pkl")

                with open(actions_pickle_file, "rb") as f:
                    actions: List[np.ndarray] = load(f)

                with open(observations_pickle_file, "rb") as f:
                    observations: List[np.ndarray] = load(f)

                self.write_episode(observations, actions)
