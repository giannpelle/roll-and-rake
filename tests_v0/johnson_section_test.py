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

class JohnsonSectionTest(unittest.TestCase):

    def test_johnson_section_wrong_dice(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 4))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 4))

        johnson_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(johnson_section.tick_list, expected_result)

    def test_johnson_section_wrong_dice_values(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 2))
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Orange, 4))

        johnson_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(johnson_section.tick_list, expected_result)

    def test_johnson_section_more_dice_than_needed(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Brown, 5))
        dice.append(Die(Color.Orange, 4))

        johnson_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(johnson_section.tick_list, expected_result)
    
    def test_johnson_section_empty_2_dice(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Brown, 5))

        johnson_section.make_use_of(dice=dice)

        expected_result[:2] = 9
        np.testing.assert_array_equal(johnson_section.tick_list, expected_result)

    def test_johnson_section_empty_3_dice(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 4))
        dice.append(Die(Color.Brown, 4))
        dice.append(Die(Color.Brown, 4))

        johnson_section.make_use_of(dice=dice)

        expected_result[:3] = 9
        np.testing.assert_array_equal(johnson_section.tick_list, expected_result)

    def test_johnson_section_with_advantage_2_dice(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Brown, 5))

        johnson_section.tick_list[0] = 9

        johnson_section.make_use_of(dice=dice)

        expected_result[:3] = 9
        np.testing.assert_array_equal(johnson_section.tick_list, expected_result)

    def test_johnson_section_with_advantage_3_dice(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Brown, 5))
        dice.append(Die(Color.Brown, 5))

        johnson_section.tick_list[0] = 9

        johnson_section.make_use_of(dice=dice)

        expected_result[:3] = 9
        np.testing.assert_array_equal(johnson_section.tick_list, expected_result)

    def test_johnson_section_1_row_filled(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Brown, 5))
        dice.append(Die(Color.Brown, 5))

        johnson_section.tick_list[:3] = 9

        johnson_section.make_use_of(dice=dice)

        expected_result[:3] = 9
        expected_result[3:6] = 9
        np.testing.assert_array_equal(johnson_section.tick_list, expected_result)

    def test_johnson_section_first_3_row_filled(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Brown, 5))
        dice.append(Die(Color.Brown, 5))

        johnson_section.tick_list[:9] = 9

        johnson_section.make_use_of(dice=dice)

        expected_result[:12] = 9
        np.testing.assert_array_equal(johnson_section.tick_list, expected_result)

    def test_johnson_section_first_3_row_filled_with_advantage_2_dice(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Brown, 5))

        johnson_section.tick_list[:9] = 9
        johnson_section.tick_list[9] = 9

        johnson_section.make_use_of(dice=dice)

        expected_result[:12] = 9
        np.testing.assert_array_equal(johnson_section.tick_list, expected_result)

    def test_johnson_section_full(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()
        expected_result[:18] = 9

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Brown, 5))
        dice.append(Die(Color.Brown, 5))

        johnson_section.tick_list[:18] = 9

        johnson_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(johnson_section.tick_list, expected_result)

    
    def test_johnson_scoring_empty(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        score = johnson_section.get_score()
        self.assertEqual(score, 0)

    def test_johnson_scoring_empty_with_advantage(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        johnson_section.tick_list[0] = 9
        score = johnson_section.get_score()
        self.assertEqual(score, 0)

    def test_johnson_scoring_1_row_filled(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        johnson_section.tick_list[:3] = 9
        score = johnson_section.get_score()
        self.assertEqual(score, 0)

    def test_johnson_scoring_1_row_filled_with_advantage(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        johnson_section.tick_list[:3] = 9
        johnson_section.tick_list[3] = 9
        score = johnson_section.get_score()
        self.assertEqual(score, 0)

    def test_johnson_scoring_3_row_filled(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        johnson_section.tick_list[:9] = 9
        score = johnson_section.get_score()
        self.assertEqual(score, 15)

    def test_johnson_scoring_3_row_filled_with_advantage(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        johnson_section.tick_list[:9] = 9
        johnson_section.tick_list[9] = 9
        score = johnson_section.get_score()
        self.assertEqual(score, 15)

    def test_johnson_scoring_4_row_filled(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        johnson_section.tick_list[:12] = 9
        score = johnson_section.get_score()
        self.assertEqual(score, 15)

    def test_johnson_scoring_4_row_filled_with_advantage(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        johnson_section.tick_list[:12] = 9
        johnson_section.tick_list[12] = 9
        score = johnson_section.get_score()
        self.assertEqual(score, 15)

    def test_johnson_scoring_fully_filled(self):
        johnson_section_metadata = get_section_metadata(with_name="Johnson")
        johnson_section = ContinuosSection(johnson_section_metadata)

        expected_result = johnson_section.tick_list.copy()

        johnson_section.tick_list[:18] = 9
        score = johnson_section.get_score()
        self.assertEqual(score, 25)

    # def test_johnson_no_bonus(self):
    #     dice = []
    #     dice.append(Die(Color.Green, 6))
    #     dice.append(Die(Color.Brown, 6))

    #     bonuses = johnson_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 0)

    # def test_johnson_first_bonus(self):
    #     dice = []
    #     dice.append(Die(Color.Green, 6))
    #     dice.append(Die(Color.Brown, 6))
    #     dice.append(Die(Color.Brown, 6))

    #     bonuses = johnson_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.Hazelnut)
    #     self.assertEqual(bonuses[0].color, Color.Green)

    # def test_johnson_first_bonus_with_advantage(self):
    #     dice = []
    #     dice.append(Die(Color.Green, 6))
    #     dice.append(Die(Color.Brown, 6))

    #     johnson_section.tick_list[0, 1] = 1

    #     bonuses = johnson_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.Hazelnut)
    #     self.assertEqual(bonuses[0].color, Color.Green)

    # def test_johnson_first_bonus_later(self):
    #     dice = []
    #     dice.append(Die(Color.Green, 6))
    #     dice.append(Die(Color.Brown, 6))
    #     dice.append(Die(Color.Brown, 6))

    #     johnson_section.tick_list[0, :3] = 1
    #     johnson_section.tick_list[1, :3] = 1
    #     johnson_section.tick_list[2, :3] = 1

    #     bonuses = johnson_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.Hazelnut)
    #     self.assertEqual(bonuses[0].color, Color.Green)

    # def test_johnson_first_bonus_with_advantage_later(self):
    #     dice = []
    #     dice.append(Die(Color.Green, 6))
    #     dice.append(Die(Color.Brown, 6))

    #     johnson_section.tick_list[0, :3] = 1
    #     johnson_section.tick_list[0, 5] = 1
    #     johnson_section.tick_list[1, :3] = 1
    #     johnson_section.tick_list[2, :3] = 1

    #     bonuses = johnson_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.Hazelnut)
    #     self.assertEqual(bonuses[0].color, Color.Green)

    # def test_johnson_no_first_bonus_later(self):
    #     dice = []
    #     dice.append(Die(Color.Green, 6))
    #     dice.append(Die(Color.Brown, 6))
    #     dice.append(Die(Color.Brown, 6))

    #     johnson_section.tick_list[0, :4] = 1
    #     johnson_section.tick_list[1, :3] = 1
    #     johnson_section.tick_list[2, :3] = 1

    #     bonuses = johnson_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 0)

if __name__ == '__main__':
    unittest.main()