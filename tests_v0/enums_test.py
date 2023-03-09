import unittest
import numpy as np

import os
import sys

parent_dir_path = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(parent_dir_path)

from roll_and_rake.envs.model_v0.enums import BonusType, Color, DiceCondition, ScoringType
from roll_and_rake.envs.model_v0.enums import vertical_rendering, horizontal_rendering, horizontal_shared_rendering, irregular_rendering
from roll_and_rake.envs.model_v0.classes import Bonus, Die

class EnumTest(unittest.TestCase):

    def test_dice_condition_consecutive_empty(self):
        self.assertFalse(DiceCondition.Consecutive.value([]))

    def test_dice_condition_consecutive_wrong(self):
        dice = []
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Orange, 5))

        self.assertFalse(DiceCondition.Consecutive.value(dice))
    
    def test_dice_condition_consecutive_wright(self):
        dice = []
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Orange, 6))

        self.assertTrue(DiceCondition.Consecutive.value(dice))

    def test_dice_condition_consecutive_brown_green_orange_empty(self):
        self.assertFalse(DiceCondition.ConsecutiveBrownGreenOrange.value([]))

    def test_dice_condition_consecutive_brown_green_orange_wrong_1(self):
        dice = []
        dice.append(Die(Color.Brown, 4))
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Green, 6))

        self.assertFalse(DiceCondition.ConsecutiveBrownGreenOrange.value(dice))

    def test_dice_condition_consecutive_brown_green_orange_wrong_2(self):
        dice = []
        dice.append(Die(Color.Brown, 4))
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Orange, 5))

        self.assertFalse(DiceCondition.ConsecutiveBrownGreenOrange.value(dice))
    
    def test_dice_condition_consecutive_brown_green_orange_wright(self):
        dice = []
        dice.append(Die(Color.Brown, 4))
        dice.append(Die(Color.Green, 5))
        dice.append(Die(Color.Orange, 6))

        self.assertTrue(DiceCondition.ConsecutiveBrownGreenOrange.value(dice))

    def test_dice_condition_greater_than_nine_empty(self):
        self.assertFalse(DiceCondition.SumGreaterThanNine.value([]))

    def test_dice_condition_greater_than_nine_wrong(self):
        dice = []
        dice.append(Die(Color.Orange, 1))
        dice.append(Die(Color.Orange, 2))
        dice.append(Die(Color.Orange, 3))

        self.assertFalse(DiceCondition.SumGreaterThanNine.value(dice))
    
    def test_dice_condition_greater_than_nine_wright(self):
        dice = []
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Orange, 6))

        self.assertTrue(DiceCondition.SumGreaterThanNine.value(dice))

    def test_dice_condition_same_value_empty(self):
        self.assertFalse(DiceCondition.SameValue.value([]))

    def test_dice_condition_same_value_wrong(self):
        dice = []
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 5))
        dice.append(Die(Color.Orange, 5))

        self.assertFalse(DiceCondition.SameValue.value(dice))
    
    def test_dice_condition_same_value_wright(self):
        dice = []
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 4))
        dice.append(Die(Color.Orange, 4))

        self.assertTrue(DiceCondition.SameValue.value(dice))

    def test_dice_condition_equal_to_one_empty(self):
        self.assertFalse(DiceCondition.IsEqualToOne.value([]))

    def test_dice_condition_equal_to_one_wrong(self):
        dice = []
        dice.append(Die(Color.Orange, 1))
        dice.append(Die(Color.Orange, 1))
        dice.append(Die(Color.Orange, 2))

        self.assertFalse(DiceCondition.IsEqualToOne.value(dice))
    
    def test_dice_condition_equal_to_one_wright(self):
        dice = []
        dice.append(Die(Color.Orange, 1))
        dice.append(Die(Color.Orange, 1))

        self.assertTrue(DiceCondition.IsEqualToOne.value(dice))

    def test_basic_scoring_empty(self):
        tick_list = [0 for _ in range(9)]
        bonuses = [Bonus(BonusType.Points, [4], 2)]

        self.assertEqual(ScoringType.Basic.value(tick_list=tick_list, bonuses=bonuses), 0)

    def test_basic_scoring_full(self):
        tick_list = [1 for _ in range(9)]
        bonuses = [Bonus(BonusType.Points, [4], 2)]

        self.assertEqual(ScoringType.Basic.value(tick_list=tick_list, bonuses=bonuses), 4)

    def test_set_collection_scoring_empty(self):
        tick_list = [0 for _ in range(9)]
        bonuses = [Bonus(BonusType.SetCollection, [0, 1, 3, 6, 10, 15], 2)]

        self.assertEqual(ScoringType.SetCollection.value(tick_list=tick_list, bonuses=bonuses), 0)

    def test_set_collection_scoring_full(self):
        tick_list = [1 for _ in range(9)]
        bonuses = [Bonus(BonusType.SetCollection, [0, 1, 3, 6, 10, 15], 2)]

        self.assertEqual(ScoringType.SetCollection.value(tick_list=tick_list, bonuses=bonuses), 1)

    def test_highest_row_sum_scoring_empty(self):
        tick_list = [0 for _ in range(9)]
        row_lengths = [3 for _ in range(3)]

        self.assertEqual(ScoringType.HighestRowSum.value(tick_list=tick_list, row_lengths=row_lengths), 0)

    def test_highest_row_sum_scoring_full(self):
        tick_list = [1 for _ in range(9)]
        tick_list[:3] = [5, 5, 6]
        row_lengths = [3 for _ in range(3)]

        self.assertEqual(ScoringType.HighestRowSum.value(tick_list=tick_list, row_lengths=row_lengths), 16)

    def test_harvick_rendering_empty(self):
        tick_list = ["0" for _ in range(12)]
        rendering = vertical_rendering(tick_list=tick_list, col_lengths=[3, 3, 3, 3])
        expected = "0 0 0 0\n0 0 0 0\n0 0 0 0"
        self.assertEqual(rendering, expected)

    def test_harvick_rendering_first_row(self):
        tick_list = ["0" for _ in range(12)]
        tick_list[:3] = ["9" for _ in range(3)]
        rendering = vertical_rendering(tick_list=tick_list, col_lengths=[3, 3, 3, 3])
        expected = "9 0 0 0\n9 0 0 0\n9 0 0 0"
        self.assertEqual(rendering, expected)

    def test_harvick_rendering_full(self):
        tick_list = ["9" for _ in range(12)]
        rendering = vertical_rendering(tick_list=tick_list, col_lengths=[3, 3, 3, 3])
        expected = "9 9 9 9\n9 9 9 9\n9 9 9 9"
        self.assertEqual(rendering, expected)

    def test_suarez_rendering_empty(self):
        tick_list = ["0" for _ in range(14)]
        rendering = vertical_rendering(tick_list=tick_list, col_lengths=[3, 3, 3, 3, 2])
        expected = "0 0 0 0 0\n0 0 0 0 0\n0 0 0 0  "
        self.assertEqual(rendering, expected)

    def test_suarez_rendering_first_row(self):
        tick_list = ["0" for _ in range(14)]
        tick_list[:2] = ["9" for _ in range(2)]
        rendering = vertical_rendering(tick_list=tick_list, col_lengths=[3, 3, 3, 3, 2])
        expected = "9 0 0 0 0\n9 0 0 0 0\n0 0 0 0  "
        self.assertEqual(rendering, expected)

    def test_suarez_rendering_full(self):
        tick_list = ["9" for _ in range(14)]
        rendering = vertical_rendering(tick_list=tick_list, col_lengths=[3, 3, 3, 3, 2])
        expected = "9 9 9 9 9\n9 9 9 9 9\n9 9 9 9  "
        self.assertEqual(rendering, expected)

    def test_elliott_rendering_empty(self):
        tick_list = ["0" for _ in range(9)]
        rendering = horizontal_rendering(tick_list=tick_list, row_lengths=[3, 3, 3], split_size=3)
        expected = "0 0 0\n0 0 0\n0 0 0"
        self.assertEqual(rendering, expected)

    def test_elliott_rendering_first_row(self):
        tick_list = ["0" for _ in range(9)]
        tick_list[:2] = ["9" for _ in range(2)]
        rendering = horizontal_rendering(tick_list=tick_list, row_lengths=[3, 3, 3], split_size=3)
        expected = "9 9 0\n0 0 0\n0 0 0"
        self.assertEqual(rendering, expected)

    def test_elliott_rendering_full(self):
        tick_list = ["9" for _ in range(9)]
        rendering = horizontal_rendering(tick_list=tick_list, row_lengths=[3, 3, 3], split_size=3)
        expected = "9 9 9\n9 9 9\n9 9 9"
        self.assertEqual(rendering, expected)

    def test_busch_rendering_empty(self):
        tick_list = ["0" for _ in range(12)]
        rendering = horizontal_rendering(tick_list=tick_list, row_lengths=[2, 2, 2, 2, 2, 2], split_size=3)
        expected = "0 0    0 0\n0 0    0 0\n0 0    0 0"
        self.assertEqual(rendering, expected)

    def test_busch_rendering_first_row(self):
        tick_list = ["0" for _ in range(12)]
        tick_list[:2] = ["9" for _ in range(2)]
        rendering = horizontal_rendering(tick_list=tick_list, row_lengths=[2, 2, 2, 2, 2, 2], split_size=3)
        expected = "9 9    0 0\n0 0    0 0\n0 0    0 0"
        self.assertEqual(rendering, expected)

    def test_busch_rendering_full(self):
        tick_list = ["9" for _ in range(12)]
        rendering = horizontal_rendering(tick_list=tick_list, row_lengths=[2, 2, 2, 2, 2, 2], split_size=3)
        expected = "9 9    9 9\n9 9    9 9\n9 9    9 9"
        self.assertEqual(rendering, expected)

    def test_earnhardt_rendering_empty(self):
        tick_list = ["0" for _ in range(10)]
        rendering = horizontal_rendering(tick_list=tick_list, row_lengths=[2, 2, 2, 2, 2], split_size=3)
        expected = "0 0    0 0\n0 0    0 0\n0 0"
        self.assertEqual(rendering, expected)

    def test_earnhardt_rendering_first_row(self):
        tick_list = ["0" for _ in range(10)]
        tick_list[:2] = ["9" for _ in range(2)]
        rendering = horizontal_rendering(tick_list=tick_list, row_lengths=[2, 2, 2, 2, 2], split_size=3)
        expected = "9 9    0 0\n0 0    0 0\n0 0"
        self.assertEqual(rendering, expected)

    def test_earnhardt_rendering_full(self):
        tick_list = ["9" for _ in range(10)]
        rendering = horizontal_rendering(tick_list=tick_list, row_lengths=[2, 2, 2, 2, 2], split_size=3)
        expected = "9 9    9 9\n9 9    9 9\n9 9"
        self.assertEqual(rendering, expected)

    def test_hamlin_rendering_empty(self):
        tick_list = ["0" for _ in range(15)]
        rendering = horizontal_rendering(tick_list=tick_list, row_lengths=[3, 3, 3, 3, 3], split_size=3)
        expected = "0 0 0    0 0 0\n0 0 0    0 0 0\n0 0 0"
        self.assertEqual(rendering, expected)

    def test_hamlin_rendering_first_row(self):
        tick_list = ["0" for _ in range(15)]
        tick_list[:3] = ["9" for _ in range(3)]
        rendering = horizontal_rendering(tick_list=tick_list, row_lengths=[3, 3, 3, 3, 3], split_size=3)
        expected = "9 9 9    0 0 0\n0 0 0    0 0 0\n0 0 0"
        self.assertEqual(rendering, expected)

    def test_hamlin_rendering_full(self):
        tick_list = ["9" for _ in range(15)]
        rendering = horizontal_rendering(tick_list=tick_list, row_lengths=[3, 3, 3, 3, 3], split_size=3)
        expected = "9 9 9    9 9 9\n9 9 9    9 9 9\n9 9 9"
        self.assertEqual(rendering, expected)

    def test_johnson_rendering_empty(self):
        tick_list = ["0" for _ in range(18)]
        rendering = horizontal_shared_rendering(tick_list=tick_list, row_lengths=[3, 3, 3, 3, 3], split_size=3)
        expected = "0 0 0 0 0\n0 0 0 0 0\n0 0 0 0 0"
        self.assertEqual(rendering, expected)

    def test_johnson_rendering_first_row_full(self):
        tick_list = ["0" for _ in range(18)]
        tick_list[:2] = ["9" for _ in range(2)]
        tick_list[9:12] = ["9" for _ in range(3)]
        rendering = horizontal_shared_rendering(tick_list=tick_list, row_lengths=[3, 3, 3, 3, 3, 3], split_size=3)
        expected = "9 9 9 9 9\n0 0 0 0 0\n0 0 0 0 0"
        self.assertEqual(rendering, expected)

    def test_johnson_rendering_first_row_with_hole(self):
        tick_list = ["0" for _ in range(18)]
        tick_list[:2] = ["9" for _ in range(2)]
        tick_list[9:11] = ["9" for _ in range(2)]
        rendering = horizontal_shared_rendering(tick_list=tick_list, row_lengths=[3, 3, 3, 3, 3, 3], split_size=3)
        expected = "9 9 0 9 9\n0 0 0 0 0\n0 0 0 0 0"
        self.assertEqual(rendering, expected)

    def test_johnson_rendering_full(self):
        tick_list = ["9" for _ in range(18)]
        rendering = horizontal_shared_rendering(tick_list=tick_list, row_lengths=[3, 3, 3, 3, 3, 3], split_size=3)
        expected = "9 9 9 9 9\n9 9 9 9 9\n9 9 9 9 9"
        self.assertEqual(rendering, expected)

    def test_patrick_rendering_empty(self):
        tick_list = ["0" for _ in range(18)]
        rendering = horizontal_shared_rendering(tick_list=tick_list, row_lengths=[3, 3, 3, 3, 3], split_size=3)
        expected = "0 0 0 0 0\n0 0 0 0 0\n0 0 0 0 0"
        self.assertEqual(rendering, expected)

    def test_patrick_rendering_first_row_full(self):
        tick_list = ["0" for _ in range(18)]
        tick_list[:2] = ["9" for _ in range(2)]
        tick_list[9:12] = ["9" for _ in range(3)]
        rendering = horizontal_shared_rendering(tick_list=tick_list, row_lengths=[3, 3, 3, 3, 3, 3], split_size=3)
        expected = "9 9 9 9 9\n0 0 0 0 0\n0 0 0 0 0"
        self.assertEqual(rendering, expected)

    def test_patrick_rendering_first_row_with_hole(self):
        tick_list = ["0" for _ in range(18)]
        tick_list[:2] = ["9" for _ in range(2)]
        tick_list[9:11] = ["9" for _ in range(2)]
        rendering = horizontal_shared_rendering(tick_list=tick_list, row_lengths=[3, 3, 3, 3, 3, 3], split_size=3)
        expected = "9 9 0 9 9\n0 0 0 0 0\n0 0 0 0 0"
        self.assertEqual(rendering, expected)

    def test_patrick_rendering_full(self):
        tick_list = ["9" for _ in range(18)]
        rendering = horizontal_shared_rendering(tick_list=tick_list, row_lengths=[3, 3, 3, 3, 3, 3], split_size=3)
        expected = "9 9 9 9 9\n9 9 9 9 9\n9 9 9 9 9"
        self.assertEqual(rendering, expected)

    def test_newman_rendering_empty(self):
        tick_list = ["0" for _ in range(6)]
        rendering = irregular_rendering(tick_list=tick_list)
        expected = "0 0 0 0 0 0"
        self.assertEqual(rendering, expected)
    
    def test_newman_rendering_first_item(self):
        tick_list = ["0" for _ in range(6)]
        tick_list[0] = "9"
        rendering = irregular_rendering(tick_list=tick_list)
        expected = "9 0 0 0 0 0"
        self.assertEqual(rendering, expected)

    def test_newman_rendering_full(self):
        tick_list = ["9" for _ in range(6)]
        rendering = irregular_rendering(tick_list=tick_list)
        expected = "9 9 9 9 9 9"
        self.assertEqual(rendering, expected)