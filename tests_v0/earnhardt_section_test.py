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

from roll_and_rake.envs.model_v0.enums import Color
from roll_and_rake.envs.model_v0.classes import ContinuosSection, Die
from roll_and_rake.envs.utils_v0.utils import get_section_metadata

class EarnhardtSectionTest(unittest.TestCase):

    def test_earnhardt_section_wrong_dice(self):
        earnhardt_section_metadata = get_section_metadata(with_name="Earnhardt")
        earnhardt_section = ContinuosSection(earnhardt_section_metadata)

        expected_result = earnhardt_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Green, 3))

        earnhardt_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(earnhardt_section.tick_list, expected_result)

    def test_earnhardt_section_wrong_dice_values(self):
        earnhardt_section_metadata = get_section_metadata(with_name="Earnhardt")
        earnhardt_section = ContinuosSection(earnhardt_section_metadata)

        expected_result = earnhardt_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 2))
        dice.append(Die(Color.Brown, 4))

        earnhardt_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(earnhardt_section.tick_list, expected_result)
    
    def test_earnhardt_section_more_dice_than_needed(self):
        earnhardt_section_metadata = get_section_metadata(with_name="Earnhardt")
        earnhardt_section = ContinuosSection(earnhardt_section_metadata)

        expected_result = earnhardt_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Orange, 3))

        earnhardt_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(earnhardt_section.tick_list, expected_result)

    def test_earnhardt_section_empty(self):
        earnhardt_section_metadata = get_section_metadata(with_name="Earnhardt")
        earnhardt_section = ContinuosSection(earnhardt_section_metadata)

        expected_result = earnhardt_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Brown, 3))

        earnhardt_section.make_use_of(dice=dice)

        expected_result[:2] = 9
        np.testing.assert_array_equal(earnhardt_section.tick_list, expected_result)

    def test_earnhardt_section_with_advantage(self):
        earnhardt_section_metadata = get_section_metadata(with_name="Earnhardt")
        earnhardt_section = ContinuosSection(earnhardt_section_metadata)

        expected_result = earnhardt_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Brown, 3))

        earnhardt_section.tick_list[0] = 9

        earnhardt_section.make_use_of(dice=dice)

        expected_result[:2] = 9
        np.testing.assert_array_equal(earnhardt_section.tick_list, expected_result)

    def test_earnhardt_section_second_row(self):
        earnhardt_section_metadata = get_section_metadata(with_name="Earnhardt")
        earnhardt_section = ContinuosSection(earnhardt_section_metadata)

        expected_result = earnhardt_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Brown, 3))

        earnhardt_section.tick_list[:2] = 9

        earnhardt_section.make_use_of(dice=dice)

        expected_result[:2] = 9
        expected_result[:4] = 9
        np.testing.assert_array_equal(earnhardt_section.tick_list, expected_result)

    def test_earnhardt_section_full(self):
        earnhardt_section_metadata = get_section_metadata(with_name="Earnhardt")
        earnhardt_section = ContinuosSection(earnhardt_section_metadata)

        expected_result = earnhardt_section.tick_list.copy()
        expected_result[:10] = 9

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Brown, 3))

        earnhardt_section.tick_list[:10] = 9

        earnhardt_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(earnhardt_section.tick_list, expected_result)

    def test_earnhardt_scoring_empty(self):
        earnhardt_section_metadata = get_section_metadata(with_name="Earnhardt")
        earnhardt_section = ContinuosSection(earnhardt_section_metadata)

        expected_result = earnhardt_section.tick_list.copy()

        score = earnhardt_section.get_score()
        self.assertEqual(score, 0)

    def test_earnhardt_scoring_empty_with_advantage(self):
        earnhardt_section_metadata = get_section_metadata(with_name="Earnhardt")
        earnhardt_section = ContinuosSection(earnhardt_section_metadata)

        expected_result = earnhardt_section.tick_list.copy()

        earnhardt_section.tick_list[0] = 9
        score = earnhardt_section.get_score()
        self.assertEqual(score, 0)

    def test_earnhardt_scoring_1_row_filled(self):
        earnhardt_section_metadata = get_section_metadata(with_name="Earnhardt")
        earnhardt_section = ContinuosSection(earnhardt_section_metadata)

        expected_result = earnhardt_section.tick_list.copy()

        earnhardt_section.tick_list[:2] = 9
        score = earnhardt_section.get_score()
        self.assertEqual(score, 0)

    def test_earnhardt_scoring_1_row_filled_with_advantage(self):
        earnhardt_section_metadata = get_section_metadata(with_name="Earnhardt")
        earnhardt_section = ContinuosSection(earnhardt_section_metadata)

        expected_result = earnhardt_section.tick_list.copy()

        earnhardt_section.tick_list[:2] = 9
        earnhardt_section.tick_list[2] = 9
        score = earnhardt_section.get_score()
        self.assertEqual(score, 0)

    def test_earnhardt_scoring_3_row_filled(self):
        earnhardt_section_metadata = get_section_metadata(with_name="Earnhardt")
        earnhardt_section = ContinuosSection(earnhardt_section_metadata)

        expected_result = earnhardt_section.tick_list.copy()

        earnhardt_section.tick_list[:2] = 9
        earnhardt_section.tick_list[:4] = 9
        earnhardt_section.tick_list[:6] = 9
        score = earnhardt_section.get_score()
        self.assertEqual(score, 18)

    def test_earnhardt_scoring_3_row_filled_with_advantage(self):
        earnhardt_section_metadata = get_section_metadata(with_name="Earnhardt")
        earnhardt_section = ContinuosSection(earnhardt_section_metadata)

        expected_result = earnhardt_section.tick_list.copy()

        earnhardt_section.tick_list[:2] = 9
        earnhardt_section.tick_list[:4] = 9
        earnhardt_section.tick_list[:6] = 9
        earnhardt_section.tick_list[6] = 9
        score = earnhardt_section.get_score()
        self.assertEqual(score, 18)

    def test_earnhardt_scoring_fully_filled(self):
        earnhardt_section_metadata = get_section_metadata(with_name="Earnhardt")
        earnhardt_section = ContinuosSection(earnhardt_section_metadata)

        expected_result = earnhardt_section.tick_list.copy()

        earnhardt_section.tick_list[:2] = 9
        earnhardt_section.tick_list[:4] = 9
        earnhardt_section.tick_list[:6] = 9
        earnhardt_section.tick_list[:8] = 9
        earnhardt_section.tick_list[:10] = 9
        score = earnhardt_section.get_score()
        self.assertEqual(score, 30)

    # def test_earnhardt_no_bonus(self):

    #     dice = []
    #     dice.append(Die(Color.Brown, 3))
    #     dice.append(Die(Color.Brown, 3))

    #     bonuses = earnhardt_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 0)

    # def test_earnhardt_first_bonus(self):

    #     dice = []
    #     dice.append(Die(Color.Brown, 4))
    #     dice.append(Die(Color.Brown, 4))

    #     earnhardt_section.tick_list[0, :] = 1
    #     earnhardt_section.tick_list[1, :] = 1

    #     bonuses = earnhardt_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.PlusOne)
    #     self.assertEqual(bonuses[0].color, Color.Orange)

    # def test_earnhardt_no_first_bonus_with_advantage(self):

    #     dice = []
    #     dice.append(Die(Color.Brown, 4))
    #     dice.append(Die(Color.Brown, 4))

    #     earnhardt_section.tick_list[0, :] = 1
    #     earnhardt_section.tick_list[1, :] = 1
    #     earnhardt_section.tick_list[2, 1] = 1

    #     bonuses = earnhardt_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 0)


if __name__ == '__main__':
    unittest.main()