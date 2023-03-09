from cmath import log
import os.path
import collections
from statistics import mean
import tensorflow as tf
import roll_and_rake
import numpy as np
import gym
import tensorflow_probability as tfp

env = gym.make("RollAndRake-v0")

class ActorModel(tf.keras.Model):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

        self.d1 = tf.keras.layers.Dense(128, activation="relu")
        self.d2 = tf.keras.layers.Dense(128, activation="relu")
        self.d3 = tf.keras.layers.Dense(action_space, activation=None)
        self.add = tf.keras.layers.Add()
        self.out = tf.keras.layers.Dense(action_space, activation="softmax")

    def call(self, obs, legal_actions):
        x = self.d1(obs)
        x = self.d2(x)
        x = self.d3(x)
        
        # mapping illegal actions to negative values
        y = 1 - legal_actions
        y = y * -1e8

        x = self.add([x, y])
        x = self.out(x)

        return x


# Reference
# https://github.com/abhisheksuran/Reinforcement_Learning/blob/master/Reinforce_(PG)_ReUploaded.ipynb

class Agent(object):
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

        self._actor = ActorModel(action_space=action_space)

        if os.path.isfile("actor_model_v0_zero_two.h5"):
            self._actor(np.zeros((1, 164)), np.zeros((1, 142)))
            self._actor.load_weights("actor_model_v0_zero_two.h5")
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 0.2

    def get_obs_legal_actions(self, observation):
        obs = tf.slice(observation,[0], [self.state_space - self.action_space])
        obs = tf.reshape(obs, shape=(1, self.state_space - self.action_space))
        legal_actions = tf.slice(observation,[self.state_space - self.action_space], [self.action_space])
        legal_actions = tf.reshape(legal_actions, shape=(1, self.action_space))
        return obs, legal_actions

    def mask_illegal_actions_from(self, probs, legal_actions_indices):
        probs = probs.numpy()[0]
        legal_actions_indices = np.array(legal_actions_indices)
        probs[legal_actions_indices == 0] = 0.0
        return probs

    def act(self, state):
        obs, legal_actions = self.get_obs_legal_actions(state)
        probs = self._actor(obs, legal_actions)
        legal_actions_indices = state[self.state_space-self.action_space:]
        mask_probs = self.mask_illegal_actions_from(probs, legal_actions_indices)
        # print("probs:")
        # print(probs)
        dist = tfp.distributions.Categorical(probs=mask_probs, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy())

    def get_loss(self, probs, action, reward):
        dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32)
        log_prob = dist.log_prob(action)

        loss = -log_prob * reward
        # print("loss:")
        # print(loss)
        return loss

    def train(self, states, rewards, actions):
        reward_sum = 0
        discounted_rewards = []
        
        for r in rewards[::-1]:
            reward_sum = r + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        for state, reward, action in zip(states, discounted_rewards, actions):
            obs, legal_actions = self.get_obs_legal_actions(state)
            # print("fitting:")
            # print(f"reward {reward}")
            
            with tf.GradientTape() as tape:
                tape.watch(obs)
                tape.watch(legal_actions)

                probs = self._actor(obs, legal_actions, training=True)
                loss = self.get_loss(probs, action, reward)

            grads = tape.gradient(loss, self._actor.variables)

            # print(loss)
            # print(grads)
            self.optimizer.apply_gradients(zip(grads, self._actor.variables))

    def save_model(self):
        self._actor.save_weights('actor_model_v0_zero_two.h5')

agent = Agent(state_space=env.observation_space.shape[0], action_space=env.action_space.n)
last_hundred_rewards = collections.deque(maxlen=10)

for episode in range(1, 1_000_000 + 1):
  if episode % 25000 == 0:
    print(f"running episode: {episode}")

  done = False
  state = env.reset()
  total_reward = 0
  
  rewards = []
  states = []
  actions = []
  
  while not done:
    # env.render()
    action = agent.act(state)
    if action not in env.legal_actions:
        action = int(np.random.choice(env.legal_actions, size=1))
    next_state, reward, done, _ = env.step(action)
    rewards.append(reward)
    states.append(state)
    actions.append(action)
    state = next_state
    total_reward += reward
    
    if done:
      agent.train(states, rewards, actions)
      last_hundred_rewards.append(total_reward)

      if episode > 0 and episode % 25_000 == 0:
        log_str = f"Episode {episode}"
        log_str += "\nLast 10 episodes rewards:"
        log_str += ", ".join(map(str, list(last_hundred_rewards)[-5:]))
        log_str += f"Average reward for last 10 episodes {round(mean(list(last_hundred_rewards)), 2)}"
        log_str += "\n"

        f = open("train_reinforce_v0_zero_two.txt", "a")
        f.write(log_str)
        f.close()

      if episode > 0 and episode % 25_000 == 0:
        agent.save_model()
