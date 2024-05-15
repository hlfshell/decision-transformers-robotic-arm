from project.trainer import BehavioralCloningTrainer
from project.dataset import RobotDataset
from project.env import SortEnv
from project.models.behaviorclone import BehaviorCloning


env = SortEnv(render_mode="rgb_array")
dataset = RobotDataset("data", success_only=True)
dataset, validation = dataset.split(0.9)

print()
print("Datasets:")
print("Training Dataset:")
print(dataset.info())
print("Validation Dataset:")
print(validation.info())
print()

model = BehaviorCloning()
trainer = BehavioralCloningTrainer(
    env,
    model,
    dataset,
    validation_dataset=validation,
    epochs=5_000,
    alpha=1e-4,
)
trainer.start_from()
trainer.train()
