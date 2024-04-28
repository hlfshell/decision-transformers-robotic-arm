from project.env import SortEnv


env = SortEnv()

for frame in range(10_000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    print("          ", end="\r")
    print(frame + 1, end="\r")

    if terminated:
        observation, info = env.reset()

print("")
