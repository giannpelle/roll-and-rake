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

class PatrickSectionTest(unittest.TestCase):

    def test_patrick_section_wrong_dice(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 4))
        dice.append(Die(Color.Brown, 4))
        dice.append(Die(Color.Brown, 4))

        patrick_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(patrick_section.tick_list, expected_result)

    def test_patrick_section_wrong_dice_values(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 2))
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Orange, 4))

        patrick_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(patrick_section.tick_list, expected_result)

    def test_patrick_section_more_dice_than_needed(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Brown, 3))

        patrick_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(patrick_section.tick_list, expected_result)
    
    def test_patrick_section_empty_2_dice(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Orange, 5))

        patrick_section.make_use_of(dice=dice)

        expected_result[:2] = 9
        np.testing.assert_array_equal(patrick_section.tick_list, expected_result)

    def test_patrick_section_empty_3_dice(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 4))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 4))

        patrick_section.make_use_of(dice=dice)

        expected_result[:3] = 9
        np.testing.assert_array_equal(patrick_section.tick_list, expected_result)

    def test_patrick_section_with_advantage_2_dice(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Orange, 5))

        patrick_section.tick_list[0] = 9

        patrick_section.make_use_of(dice=dice)

        expected_result[:3] = 9
        np.testing.assert_array_equal(patrick_section.tick_list, expected_result)

    def test_patrick_section_with_advantage_3_dice(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Orange, 5))

        patrick_section.tick_list[0] = 9

        patrick_section.make_use_of(dice=dice)

        expected_result[:3] = 9
        np.testing.assert_array_equal(patrick_section.tick_list, expected_result)

    def test_patrick_section_1_row_filled(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Orange, 5))

        patrick_section.tick_list[:3] = 9

        patrick_section.make_use_of(dice=dice)

        expected_result[:6] = 9
        np.testing.assert_array_equal(patrick_section.tick_list, expected_result)

    def test_patrick_section_first_3_row_filled(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Orange, 5))

        patrick_section.tick_list[:9] = 9

        patrick_section.make_use_of(dice=dice)

        expected_result[:12] = 9
        np.testing.assert_array_equal(patrick_section.tick_list, expected_result)

    def test_patrick_section_first_3_row_filled_with_advantage_2_dice(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Orange, 5))

        patrick_section.tick_list[:9] = 9
        patrick_section.tick_list[9] = 9

        patrick_section.make_use_of(dice=dice)

        expected_result[:12] = 9
        np.testing.assert_array_equal(patrick_section.tick_list, expected_result)

    def test_patrick_section_4_row_filled(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Orange, 5))

        patrick_section.tick_list[:12] = 9

        patrick_section.make_use_of(dice=dice)

        expected_result[:15] = 9
        np.testing.assert_array_equal(patrick_section.tick_list, expected_result)

    def test_patrick_section_full(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()
        expected_result[:18] = 9

        dice = []
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Orange, 5))

        patrick_section.tick_list[:18] = 9

        patrick_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(patrick_section.tick_list, expected_result)


    def test_patrick_scoring_empty(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        score = patrick_section.get_score()
        self.assertEqual(score, 0)

    def test_patrick_scoring_empty_with_advantage(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        patrick_section.tick_list[0] = 9
        score = patrick_section.get_score()
        self.assertEqual(score, 0)

    def test_patrick_scoring_1_row_filled(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        patrick_section.tick_list[:3] = 9
        score = patrick_section.get_score()
        self.assertEqual(score, 0)

    def test_patrick_scoring_1_row_filled_with_advantage(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        patrick_section.tick_list[:3] = 9
        patrick_section.tick_list[3] = 9
        score = patrick_section.get_score()
        self.assertEqual(score, 0)

    def test_patrick_scoring_3_row_filled(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        patrick_section.tick_list[:9] = 9
        score = patrick_section.get_score()
        self.assertEqual(score, 12)

    def test_patrick_scoring_3_row_filled_with_advantage(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        patrick_section.tick_list[:9] = 9
        patrick_section.tick_list[9] = 9
        score = patrick_section.get_score()
        self.assertEqual(score, 12)

    def test_patrick_scoring_4_row_filled(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        patrick_section.tick_list[:12] = 9
        score = patrick_section.get_score()
        self.assertEqual(score, 12)

    def test_patrick_scoring_4_row_filled_with_advantage(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        patrick_section.tick_list[:12] = 9
        patrick_section.tick_list[12] = 9
        score = patrick_section.get_score()
        self.assertEqual(score, 12)

    def test_patrick_scoring_fully_filled(self):
        patrick_section_metadata = get_section_metadata(with_name="Patrick")
        patrick_section = ContinuosSection(patrick_section_metadata)

        expected_result = patrick_section.tick_list.copy()

        patrick_section.tick_list[:18] = 9
        score = patrick_section.get_score()
        self.assertEqual(score, 28)

    # def test_patrick_no_bonus(self):
    #     dice = []
    #     dice.append(Die(Color.Green, 6))
    #     dice.append(Die(Color.Orange, 6))

    #     bonuses = patrick_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 0)

    # def test_patrick_first_bonus(self):
    #     dice = []
    #     dice.append(Die(Color.Green, 6))
    #     dice.append(Die(Color.Orange, 6))
    #     dice.append(Die(Color.Orange, 6))

    #     bonuses = patrick_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.Hazelnut)
    #     self.assertEqual(bonuses[0].color, Color.Orange)

    # def test_patrick_first_bonus_with_advantage(self):
    #     dice = []
    #     dice.append(Die(Color.Green, 6))
    #     dice.append(Die(Color.Orange, 6))

    #     patrick_section.tick_list[0, 1] = 1

    #     bonuses = patrick_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.Hazelnut)
    #     self.assertEqual(bonuses[0].color, Color.Orange)

    # def test_patrick_first_bonus_later(self):
    #     dice = []
    #     dice.append(Die(Color.Green, 6))
    #     dice.append(Die(Color.Orange, 6))
    #     dice.append(Die(Color.Orange, 6))

    #     patrick_section.tick_list[0, :3] = 1
    #     patrick_section.tick_list[1, :3] = 1
    #     patrick_section.tick_list[2, :3] = 1

    #     bonuses = patrick_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.Hazelnut)
    #     self.assertEqual(bonuses[0].color, Color.Orange)

    # def test_patrick_first_bonus_with_advantage_later(self):
    #     dice = []
    #     dice.append(Die(Color.Green, 6))
    #     dice.append(Die(Color.Orange, 6))

    #     patrick_section.tick_list[0, :3] = 1
    #     patrick_section.tick_list[0, 5] = 1
    #     patrick_section.tick_list[1, :3] = 1
    #     patrick_section.tick_list[2, :3] = 1

    #     bonuses = patrick_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.Hazelnut)
    #     self.assertEqual(bonuses[0].color, Color.Orange)

    # def test_patrick_no_first_bonus_later(self):
    #     dice = []
    #     dice.append(Die(Color.Green, 6))
    #     dice.append(Die(Color.Orange, 6))
    #     dice.append(Die(Color.Orange, 6))

    #     patrick_section.tick_list[0, :4] = 1
    #     patrick_section.tick_list[1, :3] = 1
    #     patrick_section.tick_list[2, :3] = 1

    #     bonuses = patrick_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 0)

if __name__ == '__main__':
    unittest.main()