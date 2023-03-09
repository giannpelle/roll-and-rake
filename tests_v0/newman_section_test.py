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
from roll_and_rake.envs.model_v0.classes import IrregularSection, Die
from roll_and_rake.envs.utils_v0.utils import get_section_metadata

class NewmanSectionTest(unittest.TestCase):

    def test_newman_section_wrong_dice(self):
        newman_section_metadata = get_section_metadata(with_name="Newman")
        newman_section = IrregularSection(newman_section_metadata)

        expected_result = newman_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Brown, 3))

        newman_section.make_use_of(dice=dice, with_env_choices=[2])
        np.testing.assert_array_equal(newman_section.tick_list, expected_result)

    def test_newman_section_wrong_dice_values(self):
        newman_section_metadata = get_section_metadata(with_name="Newman")
        newman_section = IrregularSection(newman_section_metadata)

        expected_result = newman_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 2))

        newman_section.tick_list[1] = 9

        newman_section.make_use_of(dice=dice, with_env_choices=[1])

        expected_result[1] = 9
        np.testing.assert_array_equal(newman_section.tick_list, expected_result)

    def test_newman_section_more_dice_than_needed(self):
        newman_section_metadata = get_section_metadata(with_name="Newman")
        newman_section = IrregularSection(newman_section_metadata)

        expected_result = newman_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 3))
        dice.append(Die(Color.Green, 2))

        newman_section.tick_list[1] = 9

        newman_section.make_use_of(dice=dice, with_env_choices=[1])

        expected_result[1] = 9
        np.testing.assert_array_equal(newman_section.tick_list, expected_result)
    
    def test_newman_section_empty(self):
        newman_section_metadata = get_section_metadata(with_name="Newman")
        newman_section = IrregularSection(newman_section_metadata)

        expected_result = newman_section.tick_list.copy()

        dice = []
        dice.append(Die(Color.Green, 3))

        newman_section.make_use_of(dice=dice, with_env_choices=[2])

        expected_result[2] = 9
        np.testing.assert_array_equal(newman_section.tick_list, expected_result)

    def test_newman_section_full(self):
        newman_section_metadata = get_section_metadata(with_name="Newman")
        newman_section = IrregularSection(newman_section_metadata)

        expected_result = newman_section.tick_list.copy()
        expected_result[:6] = 9

        dice = []
        dice.append(Die(Color.Green, 3))

        newman_section.tick_list[:6] = 9

        newman_section.make_use_of(dice=dice, with_env_choices=[2])
        np.testing.assert_array_equal(newman_section.tick_list, expected_result)


    def test_newman_scoring_empty(self):
        newman_section_metadata = get_section_metadata(with_name="Newman")
        newman_section = IrregularSection(newman_section_metadata)

        expected_result = newman_section.tick_list.copy()

        score = newman_section.get_score()
        self.assertEqual(score, 0)

    def test_newman_2_random_row_filled(self):
        newman_section_metadata = get_section_metadata(with_name="Newman")
        newman_section = IrregularSection(newman_section_metadata)

        expected_result = newman_section.tick_list.copy()

        newman_section.tick_list[0] = 9
        newman_section.tick_list[4] = 9

        score = newman_section.get_score()
        self.assertEqual(score, 3)

    def test_newman_scoring_full(self):
        newman_section_metadata = get_section_metadata(with_name="Newman")
        newman_section = IrregularSection(newman_section_metadata)

        expected_result = newman_section.tick_list.copy()

        newman_section.tick_list[:6] = 9

        score = newman_section.get_score()
        self.assertEqual(score, 25)

    # def test_newman_no_bonus(self):
    #     dice = []
    #     dice.append(Die(Color.Green, 3))

    #     bonuses = newman_section.make_use_of(dice=dice)
        
    #     self.assertEqual(len(bonuses), 0)

if __name__ == '__main__':
    unittest.main()