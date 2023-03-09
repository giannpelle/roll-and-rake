from cmath import exp
import unittest
import numpy as np

import os
import sys

parent_dir_path = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(parent_dir_path)

from roll_and_rake.envs.model_v0.enums import Color, GameMove, MoveType
from roll_and_rake.envs.model_v0.classes import ContinuosSection, Die
from roll_and_rake.envs.model_v0.roll_and_rake_state import RollAndRakeState

class RollAndRakeStateTest(unittest.TestCase):

    def test_one_hot_encoding_zero(self):
        current_state = RollAndRakeState()
        one_hot = current_state._get_one_hot_encoding(of_value=0, with_max_bit_size=7)
        np.testing.assert_array_equal(one_hot, [1, 0, 0, 0, 0, 0, 0])

    def test_one_hot_encoding_standard(self):
        current_state = RollAndRakeState()
        one_hot = current_state._get_one_hot_encoding(of_value=3, with_max_bit_size=7)
        np.testing.assert_array_equal(one_hot, [0, 0, 0, 1, 0, 0, 0])

    def test_one_hot_encoding_max(self):
        current_state = RollAndRakeState()
        one_hot = current_state._get_one_hot_encoding(of_value=6, with_max_bit_size=7)
        np.testing.assert_array_equal(one_hot, [0, 0, 0, 0, 0, 0, 1])

    def test_one_hot_encoding_out_of_bounds(self):
        current_state = RollAndRakeState()
        with self.assertRaises(Exception) as context:
            current_state._get_one_hot_encoding(of_value=9, with_max_bit_size=7)

        self.assertTrue("Value out of bounds" in str(context.exception))

    def test_harvick_one_step(self):
        current_state = RollAndRakeState()
        current_state.available_dice = [
            Die(Color.Orange, 4),
            Die(Color.Orange, 5),
            Die(Color.Orange, 6),
            Die(Color.Brown, 4),
            Die(Color.Brown, 4),
            Die(Color.Green, 4),
        ]
        
        current_state.step(with_env_action_index=63+7 - 1)
        current_state.step(with_env_action_index=63+63+1 - 1)

        np.testing.assert_array_equal(current_state.sections[0].tick_list[:3], [9 for _ in range(3)])

    def test_elliott_one_step_2_dice(self):
        current_state = RollAndRakeState()
        current_state.available_dice = [
            Die(Color.Orange, 6),
            Die(Color.Orange, 5),
            Die(Color.Orange, 2),
            Die(Color.Brown, 4),
            Die(Color.Brown, 4),
            Die(Color.Green, 4),
        ]
        
        current_state.step(with_env_action_index=63+3 - 1)
        current_state.step(with_env_action_index=63+63+2 - 1)

        np.testing.assert_array_equal(current_state.sections[1].tick_list[:2], [6, 5])

    def test_elliott_one_step_3_dice(self):
        current_state = RollAndRakeState()
        current_state.available_dice = [
            Die(Color.Orange, 6),
            Die(Color.Orange, 5),
            Die(Color.Orange, 5),
            Die(Color.Brown, 4),
            Die(Color.Brown, 4),
            Die(Color.Green, 4),
        ]
        
        current_state.step(with_env_action_index=63+7 - 1)
        current_state.step(with_env_action_index=63+63+2 - 1)

        np.testing.assert_array_equal(current_state.sections[1].tick_list[:3], [6, 5, 5])

    def test_busch_one_step(self):
        current_state = RollAndRakeState()
        current_state.available_dice = [
            Die(Color.Orange, 6),
            Die(Color.Orange, 4),
            Die(Color.Orange, 2),
            Die(Color.Brown, 4),
            Die(Color.Brown, 4),
            Die(Color.Green, 4),
        ]
        
        current_state.step(with_env_action_index=63+10 - 1)
        current_state.step(with_env_action_index=63+63+3 - 1)

        np.testing.assert_array_equal(current_state.sections[2].tick_list[:2], [9 for _ in range(2)])

    def test_newman_one_step(self):
        current_state = RollAndRakeState()
        current_state.available_dice = [
            Die(Color.Orange, 6),
            Die(Color.Orange, 4),
            Die(Color.Orange, 2),
            Die(Color.Brown, 4),
            Die(Color.Brown, 4),
            Die(Color.Green, 4),
        ]
        
        current_state.step(with_env_action_index=63+32 - 1)
        current_state.step(with_env_action_index=63+63+4 - 1)
        current_state.step(with_env_action_index=63+63+9+4 - 1)

        self.assertEqual(current_state.sections[3].tick_list[3], 9)

    def test_johnson_one_step_2_dice(self):
        current_state = RollAndRakeState()
        current_state.available_dice = [
            Die(Color.Orange, 6),
            Die(Color.Orange, 4),
            Die(Color.Orange, 2),
            Die(Color.Brown, 3),
            Die(Color.Brown, 4),
            Die(Color.Green, 3),
        ]
        
        current_state.step(with_env_action_index=63+40 - 1)
        current_state.step(with_env_action_index=63+63+5 - 1)

        np.testing.assert_array_equal(current_state.sections[4].tick_list[:2], [9 for _ in range(2)])

    def test_johnson_one_step_3_dice(self):
        current_state = RollAndRakeState()
        current_state.available_dice = [
            Die(Color.Orange, 6),
            Die(Color.Orange, 4),
            Die(Color.Orange, 2),
            Die(Color.Brown, 3),
            Die(Color.Brown, 3),
            Die(Color.Green, 3),
        ]
        
        current_state.step(with_env_action_index=63+56 - 1)
        current_state.step(with_env_action_index=63+63+5 - 1)

        np.testing.assert_array_equal(current_state.sections[4].tick_list[:3], [9 for _ in range(3)])

    def test_suarez_one_step_2_dice(self):
        current_state = RollAndRakeState()
        current_state.available_dice = [
            Die(Color.Orange, 6),
            Die(Color.Orange, 6),
            Die(Color.Orange, 2),
            Die(Color.Brown, 3),
            Die(Color.Brown, 4),
            Die(Color.Green, 3),
        ]
        
        current_state.step(with_env_action_index=63+3 - 1)
        current_state.step(with_env_action_index=63+63+6 - 1)

        np.testing.assert_array_equal(current_state.sections[5].tick_list[:2], [9 for _ in range(2)])

    def test_suarez_one_step_3_dice(self):
        current_state = RollAndRakeState()
        current_state.available_dice = [
            Die(Color.Orange, 6),
            Die(Color.Orange, 6),
            Die(Color.Orange, 6),
            Die(Color.Brown, 3),
            Die(Color.Brown, 4),
            Die(Color.Green, 3),
        ]
        
        current_state.step(with_env_action_index=63+7 - 1)
        current_state.step(with_env_action_index=63+63+6 - 1)

        np.testing.assert_array_equal(current_state.sections[5].tick_list[:3], [9 for _ in range(3)])

    def test_earnhardt_one_step(self):
        current_state = RollAndRakeState()
        current_state.available_dice = [
            Die(Color.Orange, 6),
            Die(Color.Orange, 6),
            Die(Color.Orange, 2),
            Die(Color.Brown, 1),
            Die(Color.Brown, 1),
            Die(Color.Green, 3),
        ]
        
        current_state.step(with_env_action_index=63+24 - 1)
        current_state.step(with_env_action_index=63+63+7 - 1)

        np.testing.assert_array_equal(current_state.sections[6].tick_list[:2], [9 for _ in range(2)])

    def test_patrick_one_step_2_dice(self):
        current_state = RollAndRakeState()
        current_state.available_dice = [
            Die(Color.Orange, 6),
            Die(Color.Orange, 5),
            Die(Color.Orange, 2),
            Die(Color.Brown, 1),
            Die(Color.Brown, 1),
            Die(Color.Green, 5),
        ]
        
        current_state.step(with_env_action_index=63+34 - 1)
        current_state.step(with_env_action_index=63+63+8 - 1)

        np.testing.assert_array_equal(current_state.sections[7].tick_list[:2], [9 for _ in range(2)])

    def test_patrick_one_step_3_dice(self):
        current_state = RollAndRakeState()
        current_state.available_dice = [
            Die(Color.Orange, 5),
            Die(Color.Orange, 5),
            Die(Color.Orange, 2),
            Die(Color.Brown, 1),
            Die(Color.Brown, 1),
            Die(Color.Green, 5),
        ]
        
        current_state.step(with_env_action_index=63+35 - 1)
        current_state.step(with_env_action_index=63+63+8 - 1)

        np.testing.assert_array_equal(current_state.sections[7].tick_list[:3], [9 for _ in range(3)])

    def test_hamlin_one_step_ordered(self):
        current_state = RollAndRakeState()
        current_state.available_dice = [
            Die(Color.Orange, 6),
            Die(Color.Orange, 2),
            Die(Color.Orange, 2),
            Die(Color.Brown, 1),
            Die(Color.Brown, 4),
            Die(Color.Green, 5),
        ]
        
        current_state.step(with_env_action_index=63+49 - 1)
        current_state.step(with_env_action_index=63+63+9 - 1)

        np.testing.assert_array_equal(current_state.sections[8].tick_list[:3], [9 for _ in range(3)])

    def test_hamlin_one_step_same_values(self):
        current_state = RollAndRakeState()
        current_state.available_dice = [
            Die(Color.Orange, 4),
            Die(Color.Orange, 5),
            Die(Color.Orange, 5),
            Die(Color.Brown, 1),
            Die(Color.Brown, 4),
            Die(Color.Green, 4),
        ]
        
        current_state.step(with_env_action_index=63+49 - 1)
        current_state.step(with_env_action_index=63+63+9 - 1)

        np.testing.assert_array_equal(current_state.sections[8].tick_list[:3], [9 for _ in range(3)])

    def test_dice_chosen_first(self):
        current_state = RollAndRakeState()
        dice_indices = current_state._get_dice_chosen_indices_from(action_index=1)
        np.testing.assert_array_equal(dice_indices, [0])

    def test_dice_chosen_all(self):
        current_state = RollAndRakeState()
        dice_indices = current_state._get_dice_chosen_indices_from(action_index=63)
        np.testing.assert_array_equal(dice_indices, list(range(6)))

    def test_dice_chosen_second_and_fifth(self):
        current_state = RollAndRakeState()
        dice_indices = current_state._get_dice_chosen_indices_from(action_index=18)
        np.testing.assert_array_equal(dice_indices, [1, 4])

    def test_dice_chosen_zero_index(self):
        current_state = RollAndRakeState()
        
        np.testing.assert_array_equal(current_state._get_dice_chosen_indices_from(action_index=0), [])

    def test_is_action_legal_all_reroll(self):
        current_state = RollAndRakeState()

        np.testing.assert_array_equal(current_state.get_legal_env_actions_indices()[:63], list(range(63)))

    def test_is_action_legal_no_reroll_available(self):
        current_state = RollAndRakeState()
        current_state.rerolls_available = 0

        self.assertTrue(min(current_state.get_legal_env_actions_indices()) > 62)

    def test_is_action_legal_one_die_unavailable(self):
        current_state = RollAndRakeState()
        current_state.dice_combination_choices_available = 0
        current_state.available_dice[1] = Die(Color.Orange, 0)

        self.assertTrue(len(current_state.get_legal_env_actions_indices()) == 31 + 1)

    def test_is_action_legal_no_dice_available(self):
        # no reroll, no take
        current_state = RollAndRakeState()
        current_state.available_dice = [
            Die(Color.Orange, 0),
            Die(Color.Orange, 0),
            Die(Color.Orange, 0),
            Die(Color.Brown, 0),
            Die(Color.Brown, 0),
            Die(Color.Green, 0)
        ]

        self.assertTrue(min(current_state.get_legal_env_actions_indices()) > 62+63)

    def test_harvick_take_actions(self):
        current_state = RollAndRakeState()
        current_state.rerolls_available = 0
        current_state.available_dice = [
            Die(Color.Orange, 1),
            Die(Color.Orange, 2),
            Die(Color.Orange, 3),
            Die(Color.Brown, 0),
            Die(Color.Brown, 0),
            Die(Color.Green, 0)
        ]

        self.assertTrue(current_state.get_legal_env_actions_indices() == [GameMove["T123"].value, GameMove["Pass"].value])

    def test_harvick_elliott_take_actions(self):
        current_state = RollAndRakeState()
        current_state.rerolls_available = 0
        current_state.available_dice = [
            Die(Color.Orange, 4),
            Die(Color.Orange, 5),
            Die(Color.Orange, 6),
            Die(Color.Brown, 0),
            Die(Color.Brown, 0),
            Die(Color.Green, 0)
        ]

        self.assertTrue(current_state.get_legal_env_actions_indices() == [GameMove["T13"].value, GameMove["T23"].value, GameMove["T123"].value, GameMove["Pass"].value])

    def test_elliott_suarez_take_actions(self):
        current_state = RollAndRakeState()
        current_state.rerolls_available = 0
        current_state.available_dice = [
            Die(Color.Orange, 6),
            Die(Color.Orange, 6),
            Die(Color.Orange, 6),
            Die(Color.Brown, 0),
            Die(Color.Brown, 0),
            Die(Color.Green, 0)
        ]

        self.assertTrue(current_state.get_legal_env_actions_indices() == [GameMove["T12"].value, GameMove["T13"].value, GameMove["T23"].value, GameMove["T123"].value, GameMove["Pass"].value])

    def test_busch_newman_johnson_earnhardt_patrick_hamlin_take_actions(self):
        current_state = RollAndRakeState()
        current_state.rerolls_available = 0
        current_state.available_dice = [
            Die(Color.Orange, 0),
            Die(Color.Orange, 0),
            Die(Color.Orange, 5),
            Die(Color.Brown, 5),
            Die(Color.Brown, 5),
            Die(Color.Green, 5)
        ]

        self.assertTrue(current_state.get_legal_env_actions_indices() == [GameMove["T34"].value, GameMove["T35"].value, GameMove["T45"].value, GameMove["T6"].value, GameMove["T36"].value, GameMove["T46"].value, GameMove["T346"].value, GameMove["T56"].value, GameMove["T356"].value, GameMove["T456"].value, GameMove["Pass"].value])

