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
from roll_and_rake.envs.model_v0.classes import Die, ContinuosSection
from roll_and_rake.envs.utils_v0.utils import get_section_metadata

class HarvickSectionTest(unittest.TestCase):

    def test_harvick_section_wrong_dice(self):
        harvick_section_metadata = get_section_metadata(with_name="Harvick")
        harvick_section = ContinuosSection(harvick_section_metadata)

        expected_result = harvick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Green, 5))

        check_flag = harvick_section.check_dice_requirements(dice=dice)
        self.assertEqual(check_flag, False)

    def test_harvick_section_wrong_dice_values(self):
        harvick_section_metadata = get_section_metadata(with_name="Harvick")
        harvick_section = ContinuosSection(harvick_section_metadata)

        expected_result = harvick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 2))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 5))

        check_flag = harvick_section.check_dice_requirements(dice=dice)
        self.assertEqual(check_flag, False)

    def test_harvick_section_more_dice_than_needed(self):
        harvick_section_metadata = get_section_metadata(with_name="Harvick")
        harvick_section = ContinuosSection(harvick_section_metadata)

        expected_result = harvick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Green, 1))

        check_flag = harvick_section.check_dice_requirements(dice=dice)
        self.assertEqual(check_flag, False)

    def test_harvick_section_correct_dice(self):
        harvick_section_metadata = get_section_metadata(with_name="Harvick")
        harvick_section = ContinuosSection(harvick_section_metadata)

        expected_result = harvick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 5))

        check_flag = harvick_section.check_dice_requirements(dice=dice)
        self.assertEqual(check_flag, True)
    
    def test_harvick_section_empty(self):
        harvick_section_metadata = get_section_metadata(with_name="Harvick")
        harvick_section = ContinuosSection(harvick_section_metadata)

        expected_result = harvick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 5))

        harvick_section.make_use_of(dice=dice)

        expected_result[:3] = 9
        np.testing.assert_array_equal(harvick_section.tick_list, expected_result)

    def test_harvick_section_with_advantage(self):
        harvick_section_metadata = get_section_metadata(with_name="Harvick")
        harvick_section = ContinuosSection(harvick_section_metadata)

        expected_result = harvick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 5))

        harvick_section.tick_list[0] = 9

        harvick_section.make_use_of(dice=dice)

        expected_result[:3] = 9
        np.testing.assert_array_equal(harvick_section.tick_list, expected_result)

    def test_harvick_section_second_row(self):
        harvick_section_metadata = get_section_metadata(with_name="Harvick")
        harvick_section = ContinuosSection(harvick_section_metadata)

        expected_result = harvick_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 5))

        harvick_section.tick_list[:3] = 9

        harvick_section.make_use_of(dice=dice)

        expected_result[:3] = 9
        expected_result[3:6] = 9
        np.testing.assert_array_equal(harvick_section.tick_list, expected_result)

    def test_harvick_section_full(self):
        harvick_section_metadata = get_section_metadata(with_name="Harvick")
        harvick_section = ContinuosSection(harvick_section_metadata)

        expected_result = harvick_section.tick_list.copy()
        expected_result[:12] = 9

        dice = []
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 5))

        harvick_section.tick_list[:12] = 9

                

        harvick_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(harvick_section.tick_list, expected_result)

    def test_harvick_scoring_empty(self):
        harvick_section_metadata = get_section_metadata(with_name="Harvick")
        harvick_section = ContinuosSection(harvick_section_metadata)

        expected_result = harvick_section.tick_list.copy()

        score = harvick_section.get_score()
        self.assertEqual(score, 0)

    def test_harvick_scoring_empty_with_advantage(self):
        harvick_section_metadata = get_section_metadata(with_name="Harvick")
        harvick_section = ContinuosSection(harvick_section_metadata)

        expected_result = harvick_section.tick_list.copy()

        harvick_section.tick_list[0] = 9
        score = harvick_section.get_score()
        self.assertEqual(score, 0)

    def test_harvick_scoring_1_row_filled(self):
        harvick_section_metadata = get_section_metadata(with_name="Harvick")
        harvick_section = ContinuosSection(harvick_section_metadata)

        expected_result = harvick_section.tick_list.copy()

        harvick_section.tick_list[:3] = 9
        score = harvick_section.get_score()
        self.assertEqual(score, 4)

    def test_harvick_scoring_1_row_filled_with_advantage(self):
        harvick_section_metadata = get_section_metadata(with_name="Harvick")
        harvick_section = ContinuosSection(harvick_section_metadata)

        expected_result = harvick_section.tick_list.copy()

        harvick_section.tick_list[:3] = 9
        harvick_section.tick_list[3] = 9
        score = harvick_section.get_score()
        self.assertEqual(score, 4)

    def test_harvick_scoring_fully_filled(self):
        harvick_section_metadata = get_section_metadata(with_name="Harvick")
        harvick_section = ContinuosSection(harvick_section_metadata)

        expected_result = harvick_section.tick_list.copy()

        harvick_section.tick_list[:3] = 9
        harvick_section.tick_list[:6] = 9
        harvick_section.tick_list[:9] = 9
        harvick_section.tick_list[:12] = 9
        score = harvick_section.get_score()
        self.assertEqual(score, 26)

    # def test_harvick_no_bonus(self):
    #     dice = []
    #     dice.append(Die(Color.Orange, 3))
    #     dice.append(Die(Color.Orange, 4))
    #     dice.append(Die(Color.Orange, 5))

    #     bonuses = harvick_section.apply(dice)
    #     self.assertEqual(len(bonuses), 0)

    # def test_harvick_first_bonus(self):
    #     dice = []
    #     dice.append(Die(Color.Orange, 3))
    #     dice.append(Die(Color.Orange, 4))
    #     dice.append(Die(Color.Orange, 5))

    #     harvick_section.tick_list[0, :] = 1

    #     bonuses = harvick_section.apply(dice)
    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.PlusOne)
    #     self.assertEqual(bonuses[0].color, Color.Brown)

    # def test_harvick_second_bonus(self):
    #     dice = []
    #     dice.append(Die(Color.Orange, 3))
    #     dice.append(Die(Color.Orange, 4))
    #     dice.append(Die(Color.Orange, 5))

    #     harvick_section.tick_list[0, :] = 1
    #     harvick_section.tick_list[1, :] = 1
    #     harvick_section.tick_list[2, :] = 1

    #     bonuses = harvick_section.apply(dice)
    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.Hazelnut)
    #     self.assertEqual(bonuses[0].color, Color.Green)


if __name__ == '__main__':
    unittest.main()