from project.env import SortEnv, Action
from project.controller import GamePad, KeyboardController
from project.recorder import Recorder

from time import sleep

import os


env = SortEnv(color_cheat=True)
# controller = KeyboardController()
controller = GamePad()
recorder = None

runs_dir = "./data"

FPS = 24
step = 0


def prepare_data_folder(folder: str) -> int:
    """
    Makes sure that the folder is created. Then it checks for existing
    directories within. Since these are labeled with the index we left
    off, we determine if there is existing data and, if so, what index
    we left off on. Return that index; 0 if none.
    """
    os.makedirs(folder, exist_ok=True)

    run_id = 0
    for folder in os.listdir(folder):
        try:
            run_id = max(run_id, int(folder) + 1)
        except ValueError:
            pass

    return run_id


def record_episode(env: SortEnv, recorder: Recorder, run_id: int) -> bool:
    observation, _ = env.reset()

    while True:
        action = controller.get_action()
        kill = controller.kill_episode_button()

        if kill:
            print("Early termination - restarting")
            controller.reset()
            return False

        if action is None:
            sleep(1 / FPS)
            continue

        observation, reward, terminated, _, _ = env.step(action)
        recorder.record(step, observation, action, env)

        if terminated:
            print("Saving")
            controller.reset()
            if reward == 1:
                print("Successful run")
                recorder.save(True)
                return True

            recorder.save(False)
            print("Failed run")
            return True


run_id = prepare_data_folder(runs_dir)
print("run id", run_id)

while True:
    print(f"Starting episode {run_id}")
    recorder = Recorder(f"{runs_dir}/{run_id}", save_images=True)
    success = record_episode(env, recorder, run_id)
    if success:
        run_id += 1
        recorder = None
