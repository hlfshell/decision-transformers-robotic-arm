from project.trainer import QLearningTrainer

# from project.dataset import RobotDataset
from project.q_dataset import QDataset
from project.env import SortEnv
from project.models.qlearning import QLearning

from time import time

env = SortEnv(render_mode="rgb_array", renderer="Tiny")

model = QLearning()

steps_max = 100_000
steps = 10_000
step_increment = 2_500

dataset = QDataset("data", env, model, steps_per_epoch=steps)


def on_epoch_start():
    global steps
    start = time()
    dataset.generate_steps()
    print(f"Generated {steps} steps in {time() - start:.2f}s")

    if steps < steps_max:
        steps += step_increment
    dataset.steps_per_epoch = steps


trainer = QLearningTrainer(
    env,
    model,
    dataset,
    epochs=5_000,
    checkpoints_folder="q_checkpoints",
    learning_rate=1e-4,
    on_epoch_start=on_epoch_start,
)
trainer.start_from()
trainer.train()
