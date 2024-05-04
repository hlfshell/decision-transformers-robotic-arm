from project.env import SortEnv, Action
from project.controller import GamePad, KeyboardController
from project.recorder import Recorder

from time import sleep

env = SortEnv()
# controller = KeyboardController()
controller = GamePad()
recorder = None

FPS = 24
run_id = 0
step = 0

while True:
    if recorder is None:
        recorder = Recorder(f"runs/{run_id}", save_images=True)

    action = controller.get_action()
    if action is None:
        sleep(1 / FPS)
        continue

    observation, reward, terminated, truncated, info = env.step(action)
    recorder.record(step, observation, action, env)

    step += 1

    if terminated:
        print("TERMINATED")
        run_id += 1
        step = 0
        observation, info = env.reset()

        if reward == 1:
            print("SAVING")
            recorder.save()

        recorder = None
