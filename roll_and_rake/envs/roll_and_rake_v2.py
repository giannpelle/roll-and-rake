import gym
import numpy as np
import random

from .model_v2.roll_and_rake_state import RollAndRakeState
from .model_v2.enums import GameMove, MoveType, RenderType

class RollAndRakeEnvV2(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RollAndRakeEnvV2, self).__init__()
        self.name = 'roll_and_rake'

        self.game_state = RollAndRakeState()

        self.action_space = gym.spaces.Discrete(MoveType.Pass.value)

        available_dice_space = len(self.game_state.available_dice) * 6
        binary_sections_space = sum(map(lambda x: len(x.tick_list), self.game_state.sections))
        green_value_space = 1
        legal_actions_space = self.action_space.n

        self.observation_space = gym.spaces.Box(0, 1, (available_dice_space + binary_sections_space + green_value_space + legal_actions_space, ))
        
    @property
    def observation(self):
        return self.game_state.to_observation()

    @property
    def legal_actions(self):
        return np.array(self.game_state.get_legal_env_actions_indices())

    def score_game(self):
        return self.game_state.get_current_score()

    def step(self, env_action_index):
        starting_score = self.game_state.get_current_score()

        # print(f"Action taken: {GameMove(env_action_index)}")

        self.game_state.step(with_env_action_index=env_action_index)
        new_state = self.game_state.to_observation()
        reward = self.game_state.get_current_score() - starting_score
        done = self.game_state.is_done

        self.done = done
        return new_state, reward, done, {}

    def reset(self):
        self.game_state.reset()
        self.done = False
        # print('\n\n---- NEW GAME ----')

        return self.observation

    def render(self, mode='human'):

        if mode == "human":
            if self.done:
                print('\n\nGAME OVER\n\n')

            print(self.game_state)

        elif mode== "train":
            if self.done:
                print('\n\nGAME OVER\n\n')
                
            print(f"Current score: {self.game_state.get_current_score()}")

        elif mode=="play":
            print(f"Current score: {self.game_state.get_current_score()}")
