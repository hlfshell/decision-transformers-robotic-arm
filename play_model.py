from project.env import SortEnv, Action
from project.models.behaviorclone import BehaviorCloning
from project.models.qlearning import QLearning
from project.recorder import Recorder

from time import sleep
import numpy as np

env = SortEnv()
observation = np.zeros((18,))
# model = BehaviorCloning()
# model.load("checkpoints/epoch_4956.pth.pth")
model = QLearning()
model.load("model_checkpoints/q_learning.pth")
# recorder = Recorder("tmp_recordings", save_images=True)
step = 0

from project.controller import GamePad, KeyboardController

last_action = None

observation, info = env.reset()

while True:
    actions = model.get_actions(observation)
    action = actions.pop(0)
    # while (
    #     last_action is not None
    #     and (last_action.is_a(Action.CLOSE_GRIPPER) or action.is_a(Action.OPEN_GRIPPER))
    #     and (action.is_a(Action.CLOSE_GRIPPER) or action.is_a(Action.OPEN_GRIPPER))
    # ):
    #     print("Repeated action")
    #     action = actions.pop(0)

    last_action = action

    # recorder.record(step, observation, action, env)

    print(f"Step: {step}", action)

    observation, reward, terminated, truncated, info = env.step(action)
    step += 1

    sleep(0.1)

    if terminated or step >= 120:
        step = 0
        observations = []
        actions = []
        print("Complete")
        # recorder.save(False)
        observation, info = env.reset()
