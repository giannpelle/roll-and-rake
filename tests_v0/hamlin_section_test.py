from contextvars import copy_context
import unittest
import numpy as np

import os
import sys

parent_dir_path = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(parent_dir_path)

from roll_and_rake.envs.model_v0.enums import Color
from roll_and_rake.envs.model_v0.classes import ContinuosSection, Die
from roll_and_rake.envs.utils_v0.utils import get_section_metadata

class HamlinSectionTest(unittest.TestCase):

    def test_hamlin_section_wrong_dice(self):
        hamlin_section_metadata = get_section_metadata(with_name="Hamlin")
        hamlin_section = ContinuosSection(hamlin_section_metadata)

        expected_result = hamlin_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 5))

        hamlin_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(hamlin_section.tick_list, expected_result)

    def test_hamlin_section_wrong_dice_values(self):
        hamlin_section_metadata = get_section_metadata(with_name="Hamlin")
        hamlin_section = ContinuosSection(hamlin_section_metadata)

        expected_result = hamlin_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 2))
        dice.append(Die(Color.Brown, 4))
        dice.append(Die(Color.Green, 5))

        hamlin_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(hamlin_section.tick_list, expected_result)

    def test_hamlin_section_more_dice_than_needed(self):
        hamlin_section_metadata = get_section_metadata(with_name="Hamlin")
        hamlin_section = ContinuosSection(hamlin_section_metadata)

        expected_result = hamlin_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Brown, 4))
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Orange, 3))

        hamlin_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(hamlin_section.tick_list, expected_result)
    
    def test_hamlin_section_empty(self):
        hamlin_section_metadata = get_section_metadata(with_name="Hamlin")
        hamlin_section = ContinuosSection(hamlin_section_metadata)

        expected_result = hamlin_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Green, 4))
        dice.append(Die(Color.Orange, 5))

        hamlin_section.make_use_of(dice=dice)

        expected_result[:3] = 9
        np.testing.assert_array_equal(hamlin_section.tick_list, expected_result)

    def test_hamlin_section_with_advantage(self):
        hamlin_section_metadata = get_section_metadata(with_name="Hamlin")
        hamlin_section = ContinuosSection(hamlin_section_metadata)

        expected_result = hamlin_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Green, 4))
        dice.append(Die(Color.Orange, 5))

        hamlin_section.tick_list[0] = 9

        hamlin_section.make_use_of(dice=dice)

        expected_result[:3] = 9
        np.testing.assert_array_equal(hamlin_section.tick_list, expected_result)

    def test_hamlin_section_second_row(self):
        hamlin_section_metadata = get_section_metadata(with_name="Hamlin")
        hamlin_section = ContinuosSection(hamlin_section_metadata)

        expected_result = hamlin_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Green, 3))
        dice.append(Die(Color.Brown, 3))

        hamlin_section.tick_list[:3] = 9

        hamlin_section.make_use_of(dice=dice)

        expected_result[:3] = 9
        expected_result[3:6] = 9
        np.testing.assert_array_equal(hamlin_section.tick_list, expected_result)

    def test_hamlin_section_full(self):
        hamlin_section_metadata = get_section_metadata(with_name="Hamlin")
        hamlin_section = ContinuosSection(hamlin_section_metadata)

        expected_result = hamlin_section.tick_list.copy()
        expected_result[:15] = 9

        dice = []
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Brown, 4))
        dice.append(Die(Color.Green, 5))

        hamlin_section.tick_list[:15] = 9

        hamlin_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(hamlin_section.tick_list, expected_result)
    
    def test_hamlin_scoring_empty(self):
        hamlin_section_metadata = get_section_metadata(with_name="Hamlin")
        hamlin_section = ContinuosSection(hamlin_section_metadata)

        expected_result = hamlin_section.tick_list.copy()

        score = hamlin_section.get_score()
        self.assertEqual(score, 0)

    def test_hamlin_scoring_empty_with_advantage(self):
        hamlin_section_metadata = get_section_metadata(with_name="Hamlin")
        hamlin_section = ContinuosSection(hamlin_section_metadata)

        expected_result = hamlin_section.tick_list.copy()

        hamlin_section.tick_list[0] = 9
        score = hamlin_section.get_score()
        self.assertEqual(score, 0)

    def test_hamlin_scoring_1_row_filled(self):
        hamlin_section_metadata = get_section_metadata(with_name="Hamlin")
        hamlin_section = ContinuosSection(hamlin_section_metadata)

        expected_result = hamlin_section.tick_list.copy()

        hamlin_section.tick_list[:3] = 9
        score = hamlin_section.get_score()
        self.assertEqual(score, 0)

    def test_hamlin_scoring_1_row_filled_with_advantage(self):
        hamlin_section_metadata = get_section_metadata(with_name="Hamlin")
        hamlin_section = ContinuosSection(hamlin_section_metadata)

        expected_result = hamlin_section.tick_list.copy()

        hamlin_section.tick_list[:3] = 9
        hamlin_section.tick_list[3] = 9
        score = hamlin_section.get_score()
        self.assertEqual(score, 0)

    def test_hamlin_scoring_3_row_filled(self):
        hamlin_section_metadata = get_section_metadata(with_name="Hamlin")
        hamlin_section = ContinuosSection(hamlin_section_metadata)

        expected_result = hamlin_section.tick_list.copy()

        hamlin_section.tick_list[:9] = 1
        score = hamlin_section.get_score()
        self.assertEqual(score, 18)

    def test_hamlin_scoring_3_row_filled_with_advantage(self):
        hamlin_section_metadata = get_section_metadata(with_name="Hamlin")
        hamlin_section = ContinuosSection(hamlin_section_metadata)

        expected_result = hamlin_section.tick_list.copy()

        hamlin_section.tick_list[:9] = 9
        hamlin_section.tick_list[9] = 9
        score = hamlin_section.get_score()
        self.assertEqual(score, 18)

    def test_hamlin_scoring_fully_filled(self):
        hamlin_section_metadata = get_section_metadata(with_name="Hamlin")
        hamlin_section = ContinuosSection(hamlin_section_metadata)

        expected_result = hamlin_section.tick_list.copy()

        hamlin_section.tick_list[:15] = 9
        score = hamlin_section.get_score()
        self.assertEqual(score, 30)

    # def test_hamlin_no_bonus(self):
    #     dice = []
    #     dice.append(Die(Color.Brown, 3))
    #     dice.append(Die(Color.Orange, 4))
    #     dice.append(Die(Color.Green, 5))

    #     bonuses = hamlin_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 0)

    # def test_hamlin_first_bonus(self):
    #     dice = []
    #     dice.append(Die(Color.Brown, 3))
    #     dice.append(Die(Color.Orange, 4))
    #     dice.append(Die(Color.Green, 5))

    #     hamlin_section.tick_list[0, :] = 1

    #     bonuses = hamlin_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.PlusOne)
    #     self.assertEqual(bonuses[0].color, Color.Green)

    # def test_hamlin_no_first_bonus_with_advantage(self):
    #     dice = []
    #     dice.append(Die(Color.Brown, 3))
    #     dice.append(Die(Color.Orange, 4))
    #     dice.append(Die(Color.Green, 5))

    #     hamlin_section.tick_list[0, :] = 1
    #     hamlin_section.tick_list[1, 1] = 1

    #     bonuses = hamlin_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 0)


if __name__ == '__main__':
    unittest.main()