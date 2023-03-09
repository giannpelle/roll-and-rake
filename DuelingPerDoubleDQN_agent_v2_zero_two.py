#!/usr/bin/env python

import os.path
import collections
import tensorflow as tf
import roll_and_rake
import tensorflow_probability as tfp

from typing import NamedTuple
from PER_utils import SumTree

import numpy as np
import random
import time
import statistics

import gym

from keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Add
from tensorflow.keras.optimizers import Adam
from keras import backend as K

env = gym.make("RollAndRake-v2")

class StepExperience(NamedTuple):
    """
    represents a single step experience perceived from the agent
    """
    state: int
    action: int
    reward: float
    new_state: int
    done: bool

class Memory:
    """
    Implementation of Memory was taken from:
    https://github.com/jaromiru/AI-blog/blob/master/Seaquest-DDQN-PER.py

    experiences are stored as ( s, a, r, s_ ) in SumTree
    """
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

    def get_size(self):
        return len(self.tree.data)

class ValueModel(tf.keras.Model):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

        self.d1 = tf.keras.layers.Dense(128, activation="relu", kernel_initializer='he_uniform')
        self.d2 = tf.keras.layers.Dense(128, activation="relu", kernel_initializer='he_uniform')
        self.d3 = tf.keras.layers.Dense(1, activation="relu")
        self.d4 = tf.keras.layers.Dense(action_space, activation="relu")
        self.add1 = tf.keras.layers.Add()
        self.add2 = tf.keras.layers.Add()

    def call(self, obs, legal_actions):
        x = self.d1(obs)
        x = self.d2(x)

        state_value = self.d3(x)

        action_advantage = self.d4(x)
        action_advantage = Lambda(lambda a: a - K.mean(a, keepdims=True), output_shape=(len(legal_actions),))(action_advantage)

        sum_value = self.add1([state_value, action_advantage])
        
        # mapping illegal actions to negative values
        y = 1 - legal_actions
        y = y * -1e8

        x = self.add2([sum_value, y])

        return x

class DuelingPerDoubleDQNModel(object):
    """
    implementation of a neural network to derive a value approximation function
    for each (state, action) pair
    The hyper-parameters of the neural networks and
    the epsilon, decay values of the epsilon-greedy policy
    were taken from this implementation (ONLY the values of the parameters)
    https://github.com/Khev/RL-practice-keras/blob/master/DDQN/agent.py
    https://github.com/Khev/RL-practice-keras/blob/master/DDQN/write_up_for_openai.ipynb

    target network is a more stable model for the state-action value approximation, 
    whereas the online network is a more dynamic model updated at each time step. 
    We use it to get the behavior policy
    target network weights are updated at each time step towards the online network weigths 
    based on the tau parameter
    """
    def __init__(self, observation_space, action_space, memory_size=5_000, batch_size=32, alpha=0.001, gamma=0.2, tau=0.1, learning_start=500):
        self.memory = Memory(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # collect this many experiences before learning
        self.learning_start = learning_start

        self.observation_space = observation_space
        self.action_space = action_space

        self.online_network = ValueModel(action_space=action_space)
        self.target_network = ValueModel(action_space=action_space)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        if os.path.isfile("online_network_model_v2_zero_two.h5"):
            self.online_network(np.zeros((1, 164)), np.zeros((1, 142)))
            self.target_network(np.zeros((1, 164)), np.zeros((1, 142)))
            self.online_network.load_weights("online_network_model_v2_zero_two.h5")
            self.target_network.load_weights("online_network_model_v2_zero_two.h5")

    def get_obs_legal_actions(self, observation):
        obs = tf.slice(observation, [0], [self.observation_space - self.action_space])
        obs = tf.reshape(obs, shape=(1, self.observation_space - self.action_space))
        legal_actions = tf.slice(observation, [self.observation_space - self.action_space], [self.action_space])
        legal_actions = tf.reshape(legal_actions, shape=(1, self.action_space))
        return obs, legal_actions

    def get_greedy_action_for(self, *, state):
        obs, legal_actions = self.get_obs_legal_actions(state)
        q_values = self.online_network(obs, legal_actions)
        return np.argmax(q_values)

    def remember(self, experience):
        obs, legal_actions = self.get_obs_legal_actions(experience.state)
        new_obs, new_legal_actions = self.get_obs_legal_actions(experience.new_state)
        q_value = experience.reward
        if not experience.done:
            new_action = np.argmax(self.online_network(new_obs, new_legal_actions))
            q_value = experience.reward + self.gamma * self.target_network(new_obs, new_legal_actions)[0][new_action]
        delta = abs(self.target_network(obs, legal_actions)[0][experience.action] - q_value)
        self.memory.add(delta, experience)

    def experience_replay(self):
        if self.memory.get_size() < self.learning_start:
            return

        batches = self.memory.sample(self.batch_size)

        # vectorized approach for speed performance concernes
        obs_batch = np.zeros((self.batch_size, self.observation_space - self.action_space))
        legal_actions_batch = np.zeros((self.batch_size, self.action_space))
        action_batch = []
        reward_batch = []
        new_obs_batch = np.zeros((self.batch_size, self.observation_space - self.action_space))
        new_legal_actions_batch = np.zeros((self.batch_size, self.action_space))
        done_batch = []

        for index, (tree_index, experience) in enumerate(batches):
            obs, legal_actions = self.get_obs_legal_actions(experience.state)
            obs_batch[index] = obs
            legal_actions_batch[index] = legal_actions
            action_batch.append(experience.action)
            reward_batch.append(experience.reward)
            new_obs, new_legal_actions = self.get_obs_legal_actions(experience.new_state)
            new_obs_batch[index] = new_obs
            new_legal_actions_batch[index] = new_legal_actions
            done_batch.append(experience.done)

        online_network_predict_state_batch = self.online_network(obs_batch, legal_actions_batch).numpy()
        online_network_predict_new_state_batch = self.online_network(new_obs_batch, new_legal_actions_batch)
        target_network_predict_new_state_batch = self.target_network(new_obs_batch, new_legal_actions_batch)
        
        index_experience_q_values = []

        for index, (tree_index, experience) in enumerate(batches):
            q_update = experience.reward
            if not experience.done:
                new_action = np.argmax(online_network_predict_new_state_batch[index])
                q_update = experience.reward + self.gamma * target_network_predict_new_state_batch[index][new_action]
            
            online_network_predict_state_batch[index][action_batch[index]] = q_update
            index_experience_q_values.append((tree_index, experience, q_update, index))
            
        with tf.GradientTape() as tape:
            y_pred = self.online_network(obs_batch, legal_actions_batch, training=True)
            loss_value = tf.keras.losses.mean_squared_error(online_network_predict_state_batch, y_pred)
            
        grads = tape.gradient(loss_value, self.online_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_network.trainable_variables))

        self.update_target_network()

        target_network_predict_state_batch = self.target_network(obs_batch, legal_actions_batch)

        for index, experience, q_value, batch_index in index_experience_q_values:
            delta = abs(target_network_predict_state_batch[batch_index][experience.action] - q_value)
            self.memory.update(index, delta)

    def update_target_network(self):
        online_network_weights = self.online_network.get_weights()
        target_network_weights = self.target_network.get_weights()
        new_weights = []

        for online_weight, target_weight in zip(online_network_weights, target_network_weights):
            new_weights.append(target_weight * (1 - self.tau) + online_weight * self.tau)
        self.target_network.set_weights(new_weights)

    def save_model(self):
        self.online_network.save_weights('online_network_model_v2_zero_two.h5')

class DuelingPerDoubleDQNAgent(object):
    """
    Implementation of the Dueling Prioritized Double Deep Q-Learning RL technique
    """

    def __init__(self, *, env):
        self.env = env
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.model = DuelingPerDoubleDQNModel(observation_space=env.observation_space.shape[0], action_space=env.action_space.n)

    def epsilon_greedy_policy(self, *, state, epsilon):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon
        """
        legal_actions = state[-self.action_space:]
        greedy_action = self.model.get_greedy_action_for(state=state)
        probs = legal_actions * (epsilon / sum(legal_actions))
        probs[greedy_action] += 1 - epsilon
        return int(np.random.choice(range(self.action_space), p=probs))

    def off_policy_train(self):
        """
        agent training phase
        """
        #print("\nTRAINING (with DuelingPerDoubleDQN learning method)")
        observations_count = self.env.observation_space.shape[0]
        actions_count = self.env.action_space.n

        episode = 0
        rewards = []

        epsilon = 0.5
        min_epsilon = 0.1
        decay = 0.9998

        for episode in range(1, 10_000 + 1):
            if episode % 100 == 0 and episode > 0:
                log_str = f"running episode: {episode}"
                log_str += f"\nlast ten training rewards: {rewards[-10:]}"
                log_str += f"\nlast 100 rewards mean: {statistics.mean(rewards[-100:])}"
                log_str += "\n"

                f = open("train_dueling_v2_zero_two.txt", "a")
                f.write(log_str)
                f.close()

            steps = 0
            state = self.env.reset()

            done = False
            episode_reward = 0

            while not done:
                steps += 1
                action = self.epsilon_greedy_policy(state=state, epsilon=epsilon)
                if action not in env.legal_actions:
                    action = int(np.random.choice(env.legal_actions, size=1))

                new_state, reward, done, _ = self.env.step(action)
                experience = StepExperience(state, action, reward, new_state, done)
                
                # start = time.time()
                self.model.remember(experience)
                # remember_time = time.time()
                # print(f"remember time: {remember_time - start}")
                self.model.experience_replay()
                # experience_time = time.time()
                # print(f"experience time: {experience_time - remember_time}")
 
                episode_reward += reward
                state = new_state

            if episode % 200 == 0:
                self.model.save_model()

            epsilon = max(epsilon * decay, min_epsilon)

            rewards.append(episode_reward)

    # def run_optimal(self):
    #     """
    #     run the agent in the given environment following the policy being calculated
    #     """
    #     observations_count = self.env.observation_space.shape[0]
    #     observation = self.env.reset()
    #     state = np.reshape(observation, [1, observations_count])

    #     done = False
    #     episode_reward = 0

    #     while not done:
    #         action = self.model.get_greedy_action_for(state=state)

    #         new_observation, reward, done, _ = self.env.step(action)
    #         new_state = np.reshape(new_observation, [1, observations_count])
    #         self.env.render()
    #         time.sleep(0.3)

    #         episode_reward += reward
    #         state = new_state

    #     print()
    #     print(f"Reward following the optimal policy: {episode_reward}")
    #     self.env.close()

dueling_per_double_dqn_agent = DuelingPerDoubleDQNAgent(env=env)
dueling_per_double_dqn_agent.off_policy_train()
