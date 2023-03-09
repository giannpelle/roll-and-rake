import unittest
import numpy as np

import os
import sys

parent_dir_path = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(parent_dir_path)

from roll_and_rake.envs.model_v0.enums import Color, BonusType
from roll_and_rake.envs.model_v0.classes import ContinuosSection, Die
from roll_and_rake.envs.utils_v0.utils import get_section_metadata

class ElliottSectionTest(unittest.TestCase):

    def test_elliott_section_wrong_dice(self):
        elliott_section_metadata = get_section_metadata(with_name="Elliott")
        elliott_section = ContinuosSection(elliott_section_metadata)

        expected_result = elliott_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Green, 5))

        elliott_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(elliott_section.tick_list, expected_result)

    def test_elliott_section_wrong_dice_values(self):
        elliott_section_metadata = get_section_metadata(with_name="Elliott")
        elliott_section = ContinuosSection(elliott_section_metadata)

        expected_result = elliott_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 2))
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Orange, 4))

        elliott_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(elliott_section.tick_list, expected_result)

    def test_elliott_section_more_dice_than_needed(self):
        elliott_section_metadata = get_section_metadata(with_name="Elliott")
        elliott_section = ContinuosSection(elliott_section_metadata)

        expected_result = elliott_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 2))
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 6))

        elliott_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(elliott_section.tick_list, expected_result)
    
    def test_elliott_section_empty_2_dice(self):
        elliott_section_metadata = get_section_metadata(with_name="Elliott")
        elliott_section = ContinuosSection(elliott_section_metadata)

        expected_result = elliott_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Orange, 6))

        elliott_section.make_use_of(dice=dice)

        expected_result[0] = 5
        expected_result[1] = 6
        np.testing.assert_array_equal(elliott_section.tick_list, expected_result)

    def test_elliott_section_empty_3_dice(self):
        elliott_section_metadata = get_section_metadata(with_name="Elliott")
        elliott_section = ContinuosSection(elliott_section_metadata)

        expected_result = elliott_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Orange, 6))

        elliott_section.make_use_of(dice=dice)

        expected_result[0] = 4
        expected_result[1] = 5
        expected_result[2] = 6
        np.testing.assert_array_equal(elliott_section.tick_list, expected_result)

    def test_elliott_section_with_advantage_2_dice(self):
        elliott_section_metadata = get_section_metadata(with_name="Elliott")
        elliott_section = ContinuosSection(elliott_section_metadata)

        expected_result = elliott_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Orange, 6))

        elliott_section.tick_list[0] = 9

        elliott_section.make_use_of(dice=dice)

        expected_result[0] = 9
        expected_result[1] = 5
        expected_result[2] = 6

        np.testing.assert_array_equal(elliott_section.tick_list, expected_result)

    def test_elliott_section_with_advantage_3_dice(self):
        elliott_section_metadata = get_section_metadata(with_name="Elliott")
        elliott_section = ContinuosSection(elliott_section_metadata)

        expected_result = elliott_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Orange, 5))

        elliott_section.tick_list[0] = 9

        elliott_section.make_use_of(dice=dice)

        expected_result[:3] = 5
        np.testing.assert_array_equal(elliott_section.tick_list, expected_result)

    def test_elliott_section_second_row(self):
        elliott_section_metadata = get_section_metadata(with_name="Elliott")
        elliott_section = ContinuosSection(elliott_section_metadata)

        expected_result = elliott_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 6))
        dice.append(Die(Color.Orange, 6))
        dice.append(Die(Color.Orange, 6))

        elliott_section.tick_list[:3] = 4

        elliott_section.make_use_of(dice=dice)

        expected_result[:3] = 4
        expected_result[3:6] = 6
        np.testing.assert_array_equal(elliott_section.tick_list, expected_result)

    def test_elliott_section_full(self):
        elliott_section_metadata = get_section_metadata(with_name="Elliott")
        elliott_section = ContinuosSection(elliott_section_metadata)

        expected_result = elliott_section.tick_list.copy()
        expected_result[:9] = 4

        dice = []
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Orange, 6))

        elliott_section.tick_list[:3] = 4
        elliott_section.tick_list[3:6] = 4
        elliott_section.tick_list[6:9] = 4

        elliott_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(elliott_section.tick_list, expected_result)

    def test_elliott_scoring_empty(self):
        elliott_section_metadata = get_section_metadata(with_name="Elliott")
        elliott_section = ContinuosSection(elliott_section_metadata)

        expected_result = elliott_section.tick_list.copy()

        score = elliott_section.get_score()
        self.assertEqual(score, 0)

    def test_elliott_scoring_empty_with_advantage(self):
        elliott_section_metadata = get_section_metadata(with_name="Elliott")
        elliott_section = ContinuosSection(elliott_section_metadata)

        expected_result = elliott_section.tick_list.copy()

        elliott_section.tick_list[0] = 9
        score = elliott_section.get_score()
        self.assertEqual(score, 0)

    def test_elliott_scoring_1_row_filled(self):
        elliott_section_metadata = get_section_metadata(with_name="Elliott")
        elliott_section = ContinuosSection(elliott_section_metadata)

        expected_result = elliott_section.tick_list.copy()

        elliott_section.tick_list[0] = 5
        elliott_section.tick_list[1] = 6

        score = elliott_section.get_score()
        self.assertEqual(score, 11)

    def test_elliott_scoring_1_row_filled_with_advantage(self):
        elliott_section_metadata = get_section_metadata(with_name="Elliott")
        elliott_section = ContinuosSection(elliott_section_metadata)

        expected_result = elliott_section.tick_list.copy()

        elliott_section.tick_list[0] = 5
        elliott_section.tick_list[1] = 6
        elliott_section.tick_list[3] = 9
        score = elliott_section.get_score()
        self.assertEqual(score, 11)

    def test_elliott_scoring_2_row_filled_ascending(self):
        elliott_section_metadata = get_section_metadata(with_name="Elliott")
        elliott_section = ContinuosSection(elliott_section_metadata)

        expected_result = elliott_section.tick_list.copy()

        elliott_section.tick_list[0] = 5
        elliott_section.tick_list[1] = 6
        elliott_section.tick_list[3] = 5
        elliott_section.tick_list[4] = 5
        elliott_section.tick_list[5] = 5
        score = elliott_section.get_score()
        self.assertEqual(score, 15)

    def test_elliott_scoring_2_row_filled_descending(self):
        elliott_section_metadata = get_section_metadata(with_name="Elliott")
        elliott_section = ContinuosSection(elliott_section_metadata)

        expected_result = elliott_section.tick_list.copy()

        elliott_section.tick_list[0] = 5
        elliott_section.tick_list[1] = 5
        elliott_section.tick_list[2] = 5
        elliott_section.tick_list[3] = 5
        elliott_section.tick_list[4] = 6
        score = elliott_section.get_score()
        self.assertEqual(score, 15)

    def test_elliott_scoring_fully_filled(self):
        elliott_section_metadata = get_section_metadata(with_name="Elliott")
        elliott_section = ContinuosSection(elliott_section_metadata)

        expected_result = elliott_section.tick_list.copy()

        elliott_section.tick_list[0] = 5
        elliott_section.tick_list[1] = 6
        elliott_section.tick_list[3] = 6
        elliott_section.tick_list[4] = 6
        elliott_section.tick_list[5] = 6
        elliott_section.tick_list[6] = 5
        elliott_section.tick_list[7] = 5
        elliott_section.tick_list[8] = 5
        score = elliott_section.get_score()
        self.assertEqual(score, 18)

    # def test_elliott_no_bonus(self):

    #     dice = []
    #     dice.append(Die(Color.Orange, 5))
    #     dice.append(Die(Color.Orange, 6))

    #     elliott_section.tick_list[0, :] = 1

    #     bonuses = elliott_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 0)

    # def test_elliott_first_bonus(self):

    #     dice = []
    #     dice.append(Die(Color.Orange, 3))
    #     dice.append(Die(Color.Orange, 4))
    #     dice.append(Die(Color.Orange, 5))

    #     elliott_section.tick_list[0, :] = 1

    #     bonuses = elliott_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.Hazelnut)
    #     self.assertEqual(bonuses[0].color, Color.Orange)

    # def test_elliott_second_bonus_with_advantage(self):

    #     dice = []
    #     dice.append(Die(Color.Orange, 3))
    #     dice.append(Die(Color.Orange, 4))
    #     dice.append(Die(Color.Orange, 5))

    #     elliott_section.tick_list[0, :] = 1
    #     elliott_section.tick_list[1, :] = 1

    #     bonuses = elliott_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.Hazelnut)
    #     self.assertEqual(bonuses[0].color, Color.Brown)


if __name__ == '__main__':
    unittest.main()