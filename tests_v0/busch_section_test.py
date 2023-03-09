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

class BuschSectionTest(unittest.TestCase):

    def test_busch_section_wrong_dice(self):
        busch_section_metadata = get_section_metadata(with_name="Busch")
        busch_section = ContinuosSection(busch_section_metadata)

        expected_result = busch_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Green, 5))

        busch_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(busch_section.tick_list, expected_result)

    def test_busch_section_wrong_dice_values(self):
        busch_section_metadata = get_section_metadata(with_name="Busch")
        busch_section = ContinuosSection(busch_section_metadata)

        expected_result = busch_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 2))
        dice.append(Die(Color.Orange, 4))

        busch_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(busch_section.tick_list, expected_result)
    
    def test_busch_section_more_dice_than_needed(self):
        busch_section_metadata = get_section_metadata(with_name="Busch")
        busch_section = ContinuosSection(busch_section_metadata)

        expected_result = busch_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Orange, 3))
        dice.append(Die(Color.Green, 3))

        busch_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(busch_section.tick_list, expected_result)

    def test_busch_section_empty(self):
        busch_section_metadata = get_section_metadata(with_name="Busch")
        busch_section = ContinuosSection(busch_section_metadata)

        expected_result = busch_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Orange, 3))

        busch_section.make_use_of(dice=dice)

        expected_result[:2] = 9
        np.testing.assert_array_equal(busch_section.tick_list, expected_result)

    def test_busch_section_with_advantage(self):
        busch_section_metadata = get_section_metadata(with_name="Busch")
        busch_section = ContinuosSection(busch_section_metadata)

        expected_result = busch_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Orange, 3))

        busch_section.tick_list[0] = 9

        busch_section.make_use_of(dice=dice)

        expected_result[:2] = 9
        np.testing.assert_array_equal(busch_section.tick_list, expected_result)

    def test_busch_section_second_row(self):
        busch_section_metadata = get_section_metadata(with_name="Busch")
        busch_section = ContinuosSection(busch_section_metadata)

        expected_result = busch_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Orange, 3))

        busch_section.tick_list[:2] = 9

        busch_section.make_use_of(dice=dice)

        expected_result[:2] = 9
        expected_result[:4] = 9
        np.testing.assert_array_equal(busch_section.tick_list, expected_result)

    def test_busch_section_full(self):
        busch_section_metadata = get_section_metadata(with_name="Busch")
        busch_section = ContinuosSection(busch_section_metadata)

        expected_result = busch_section.tick_list.copy()
        expected_result[:12] = 9

        dice = []
        dice.append(Die(Color.Brown, 3))
        dice.append(Die(Color.Orange, 3))

        busch_section.tick_list[:12] = 9

        busch_section.make_use_of(dice=dice)
        np.testing.assert_array_equal(busch_section.tick_list, expected_result)

    def test_busch_scoring_empty(self):
        busch_section_metadata = get_section_metadata(with_name="Busch")
        busch_section = ContinuosSection(busch_section_metadata)

        expected_result = busch_section.tick_list.copy()

        score = busch_section.get_score()
        self.assertEqual(score, 0)

    def test_busch_scoring_empty_with_advantage(self):
        busch_section_metadata = get_section_metadata(with_name="Busch")
        busch_section = ContinuosSection(busch_section_metadata)

        expected_result = busch_section.tick_list.copy()

        busch_section.tick_list[0] = 9
        score = busch_section.get_score()
        self.assertEqual(score, 0)

    def test_busch_scoring_1_row_filled(self):
        busch_section_metadata = get_section_metadata(with_name="Busch")
        busch_section = ContinuosSection(busch_section_metadata)

        expected_result = busch_section.tick_list.copy()

        busch_section.tick_list[0] = 9
        score = busch_section.get_score()
        self.assertEqual(score, 0)

    def test_busch_scoring_1_row_filled_with_advantage(self):
        busch_section_metadata = get_section_metadata(with_name="Busch")
        busch_section = ContinuosSection(busch_section_metadata)

        expected_result = busch_section.tick_list.copy()

        busch_section.tick_list[:2] = 9
        busch_section.tick_list[2] = 9
        score = busch_section.get_score()
        self.assertEqual(score, 0)

    def test_busch_scoring_3_row_filled(self):
        busch_section_metadata = get_section_metadata(with_name="Busch")
        busch_section = ContinuosSection(busch_section_metadata)

        expected_result = busch_section.tick_list.copy()

        busch_section.tick_list[:2] = 9
        busch_section.tick_list[:4] = 9
        busch_section.tick_list[:6] = 9
        score = busch_section.get_score()
        self.assertEqual(score, 8)

    def test_busch_scoring_3_row_filled_with_advantage(self):
        busch_section_metadata = get_section_metadata(with_name="Busch")
        busch_section = ContinuosSection(busch_section_metadata)

        expected_result = busch_section.tick_list.copy()

        busch_section.tick_list[:2] = 9
        busch_section.tick_list[:4] = 9
        busch_section.tick_list[:6] = 9
        busch_section.tick_list[6] = 9
        score = busch_section.get_score()
        self.assertEqual(score, 8)

    def test_busch_scoring_fully_filled(self):
        busch_section_metadata = get_section_metadata(with_name="Busch")
        busch_section = ContinuosSection(busch_section_metadata)

        expected_result = busch_section.tick_list.copy()

        busch_section.tick_list[:2] = 9
        busch_section.tick_list[:4] = 9
        busch_section.tick_list[:6] = 9
        busch_section.tick_list[:8] = 9
        busch_section.tick_list[:10] = 9
        busch_section.tick_list[:12] = 9
        score = busch_section.get_score()
        self.assertEqual(score, 16)

    # def test_busch_no_bonus(self):

    #     dice = []
    #     dice.append(Die(Color.Brown, 3))
    #     dice.append(Die(Color.Orange, 3))

    #     bonuses = busch_section.make_use_of(dice=dice)
    #     self.assertEqual(len(bonuses), 0)

    # def test_busch_first_bonus(self):

    #     dice = []
    #     dice.append(Die(Color.Brown, 4))
    #     dice.append(Die(Color.Orange, 4))

    #     busch_section.tick_list[0, :] = 1
    #     busch_section.tick_list[1, :] = 1
    #     busch_section.tick_list[2, :] = 1
    #     busch_section.tick_list[3, :] = 1

    #     bonuses = busch_section.make_use_of(dice=dice)

    #     self.assertEqual(len(bonuses), 1)
    #     self.assertEqual(bonuses[0].type, BonusType.Hazelnut)
    #     self.assertEqual(bonuses[0].color, Color.Brown)

    
if __name__ == '__main__':
    unittest.main()