import gym
import random
from roll_and_rake.envs.model_v0.enums import GameMove

env = gym.make("RollAndRake-v0")

is_random_no_pass_agent = False

rewards = []

for episode in range(1, 100 + 1):

    state = env.reset()

    done = False
    episode_reward = 0

    while not done:
        action = 0
        
        if is_random_no_pass_agent:
            if len(env.legal_actions) >= 2:
                useful_actions = list(filter(lambda x: x != 141, env.legal_actions))
                action = random.choice(useful_actions)
            else:
                action = env.legal_actions[0]
        else:
            action = random.choice(env.legal_actions)

        new_state, reward, done, _ = env.step(action)

        episode_reward += reward
        state = new_state

    rewards.append(episode_reward)

print(rewards)
