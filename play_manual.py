import gym
import random
from roll_and_rake.envs.model_v0.enums import GameMove

env = gym.make("RollAndRake-v0")

obs = env.reset()
env.render()
done = False

while not done:
    action = 0

    is_action_legal = False

    while not is_action_legal:
        action_input = input("Enter move: ")
        if action_input in GameMove.__members__:
            action = GameMove[action_input].value
            is_action_legal = True
        else:
            print("The action provided is illegal, try again\n")

    obs, reward, done, info = env.step(action)
    print(f"immediate reward: {reward}")
    print(f"done: {done}")
    env.render(mode="human")
    print("\n")

env.render(mode="human")
