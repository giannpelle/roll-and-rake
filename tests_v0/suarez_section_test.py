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

class SuarezSectionTest(unittest.TestCase):

    def test_suarez_section_wrong_dice(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Green, 5))

        suarez_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(suarez_section.tick_list, expected_result)

    def test_suarez_section_wrong_dice_values(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 2))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 5))

        suarez_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(suarez_section.tick_list, expected_result)

    def test_suarez_section_more_dice_than_needed(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Green, 5))

        suarez_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(suarez_section.tick_list, expected_result)
    
    def test_suarez_section_empty_2_dice(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 4))

        suarez_section.make_use_of(dice=dice)

        expected_result[:2] = 9
        np.testing.assert_array_equal(suarez_section.tick_list, expected_result)

    def test_suarez_section_empty_3_dice(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 4))

        suarez_section.make_use_of(dice=dice)

        expected_result[:3] = 9
        np.testing.assert_array_equal(suarez_section.tick_list, expected_result)

    def test_suarez_section_with_advantage_2_dice(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Orange, 3))

        suarez_section.tick_list[0] = 9

        suarez_section.make_use_of(dice=dice)

        expected_result[:3] = 9
        np.testing.assert_array_equal(suarez_section.tick_list, expected_result)

    def test_suarez_section_with_advantage_3_dice(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 2))
        dice.append(Die(Color.Orange, 2))
        dice.append(Die(Color.Orange, 2))

        suarez_section.tick_list[0] = 9

        suarez_section.make_use_of(dice=dice)

        expected_result[:3] = 9
        np.testing.assert_array_equal(suarez_section.tick_list, expected_result)

    def test_suarez_section_second_row(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Orange, 3))

        suarez_section.tick_list[:3] = 9

        suarez_section.make_use_of(dice=dice)

        expected_result[:6] = 9
        np.testing.assert_array_equal(suarez_section.tick_list, expected_result)

    def test_suarez_section_last_row_with_advantage_3_dice(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 2))
        dice.append(Die(Color.Orange, 2))
        dice.append(Die(Color.Orange, 2))

        suarez_section.tick_list[:12] = 9
        suarez_section.tick_list[12] = 9

        suarez_section.make_use_of(dice=dice)

        expected_result[:14] = 9
        np.testing.assert_array_equal(suarez_section.tick_list, expected_result)
    
    def test_suarez_section_full(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()
        expected_result[:14] = 9

        dice = []
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 5))

        suarez_section.tick_list[:14] = 9

        suarez_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(suarez_section.tick_list, expected_result)

    def test_suarez_scoring_empty(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()

        score = suarez_section.get_score()
        self.assertEqual(score, 0)

    def test_suarez_scoring_empty_with_advantage(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()

        suarez_section.tick_list[0] = 9
        score = suarez_section.get_score()
        self.assertEqual(score, 0)

    def test_suarez_scoring_1_row_filled(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()

        suarez_section.tick_list[:3] = 9
        score = suarez_section.get_score()
        self.assertEqual(score, 0)

    def test_suarez_scoring_1_row_filled_with_advantage(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()

        suarez_section.tick_list[:3] = 9
        suarez_section.tick_list[3] = 9
        score = suarez_section.get_score()
        self.assertEqual(score, 0)

    def test_suarez_scoring_2_row_filled(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()

        suarez_section.tick_list[:3] = 9
        suarez_section.tick_list[3:6] = 9
        score = suarez_section.get_score()
        self.assertEqual(score, 8)

    def test_suarez_scoring_2_row_filled_with_advantage(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()

        suarez_section.tick_list[:3] = 9
        suarez_section.tick_list[3:6] = 9
        suarez_section.tick_list[6] = 9
        score = suarez_section.get_score()
        self.assertEqual(score, 8)

    def test_suarez_scoring_fully_filled(self):
        suarez_section_metadata = get_section_metadata(with_name="Suarez")
        suarez_section = ContinuosSection(suarez_section_metadata)

        expected_result = suarez_section.tick_list.copy()

        suarez_section.tick_list[:14] = 9
        score = suarez_section.get_score()
        self.assertEqual(score, 26)

    # def test_suarez_no_bonus(self):
    #     dice = []
    #     dice.append(Die(Color.Orange, 3))
    #     dice.append(Die(Color.Orange, 3))

    #     bonuses = suarez_section.make_use_of(dice=dice)
        
    #     self.assertEqual(len(bonuses), 0)

    # def test_suarez_first_bonus(self):
    #     dice = []
    #     dice.append(Die(Color.Orange, 3))
    #     dice.append(Die(Color.Orange, 3))
    #     dice.append(Die(Color.Orange, 3))

    #     bonuses = suarez_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.SmallExtraPoints)

    # def test_suarez_only_one_second_bonus_with_advantage(self):
    #     dice = []
    #     dice.append(Die(Color.Orange, 3))
    #     dice.append(Die(Color.Orange, 3))
    #     dice.append(Die(Color.Orange, 3))

    #     suarez_section.tick_list[0, :] = 1
    #     suarez_section.tick_list[1, 1] = 1

    #     bonuses = suarez_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.MediumExtraPoints)


if __name__ == '__main__':
    unittest.main()