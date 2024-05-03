from project.env import SortEnv, Action
from project.controller import KeyboardController
from project.recorder import Recorder

from time import sleep

env = SortEnv()
controller = KeyboardController()
recorder = None

FPS = 24
run_id = -1

for frame in range(10_000):
    run_id += 1
    if recorder is None:
        recorder = Recorder(f"runs/{run_id}", save_images=True)

    action = controller.get_action()
    if action is None:
        sleep(1 / FPS)
        continue

    observation, reward, terminated, truncated, info = env.step(action)
    recorder.record(frame, observation, action, env)

    if terminated:
        print("TERMINATED")
        observation, info = env.reset()

        if reward == 1:
            print("SAVING")
            recorder.save()

        recorder = None
