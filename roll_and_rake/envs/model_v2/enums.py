from enum import Enum, auto
from functools import partial, reduce
import numpy as np
from math import ceil
import re

class Color(Enum):
    Orange = "\U0001F7E7"
    Brown = "\U0001F7EB"
    Green = "\U0001F7E9"

class BonusType(Enum):
    Points = "\U0001F3C6"
    SmallExtraPoints = "2\U0001F3C6"
    MediumExtraPoints = "3\U0001F3C6"
    LargeExtraPoints = "5\U0001F3C6"
    PlusOneOrange = "\U0001F341\U0001F7E0"
    PlusOneBrown = "\U0001F341\U0001F7E4"
    PlusOneGreen = "\U0001F341\U0001F7E2"
    HazelnutOrange = "\U0001F95C\U0001F7E0"
    HazelnutBrown = "\U0001F95C\U0001F7E4"
    HazelnutGreen = "\U0001F95C\U0001F7E2"
    SetCollection = "\U0001F341"

class MoveType(Enum):
    ChooseCategory = 8
    Pass = auto()

class GameMove(Enum):
    Harvick = 0
    Busch = auto()
    Newman = auto()
    Johnson = auto()
    Suarez = auto()
    Earnhardt = auto()
    Patrick = auto()
    Hamlin = auto()
    Pass = auto()

def one_tick_per_die_converter(dice):
    return [9] * len(dice)

def each_value_per_die_converter(dice):
    return list(map(lambda x: x.value, dice))

class DiceToTickConverter(Enum):
    OneTickPerDie = partial(one_tick_per_die_converter)
    EachValuePerDie = partial(each_value_per_die_converter)


def continuos_tick_function(tick_list, row_lengths, min_ticks_to_fullfill_row, ticks):
    row_lengths_indices = [[i] * v for i, v in enumerate(row_lengths)]
    row_lengths_indices_flatten = [y for x in row_lengths_indices for y in x]
    elements_different_than_zero_indices = np.where(tick_list != 0)[0]
    first_empty_cell_index = 0
    if list(elements_different_than_zero_indices):
        first_empty_cell_index = elements_different_than_zero_indices[-1] + 1

    current_row_index = row_lengths_indices_flatten[first_empty_cell_index]
    row_lengths_indices_arr = np.array(row_lengths_indices_flatten)
    current_row_arr_slice_indices = np.where(row_lengths_indices_arr == current_row_index)[0]
    current_row_first_empty_cell_index = 0

    if tick_list[current_row_arr_slice_indices][min_ticks_to_fullfill_row - 1] != 0:
        current_row_index += 1
        current_row_arr_slice_indices = np.where(row_lengths_indices_arr == current_row_index)[0]
    else:
        current_row_first_empty_cell_index = list(tick_list[current_row_arr_slice_indices]).index(0)

    max_elements_to_add = max(row_lengths[current_row_index] - current_row_first_empty_cell_index, 0)
    
    # if the new ticks would go out of bounds but 
    # the ticks already present are neutral (9) or empty (0)
    # then we overwrite thei values 
    if max_elements_to_add < len(ticks) and \
        len(tick_list[current_row_arr_slice_indices]) >= len(ticks) and \
        all([tick in [0, 9] for tick in tick_list[current_row_arr_slice_indices[-len(ticks):]]]):
        tick_list[current_row_arr_slice_indices[-len(ticks):]] = ticks
        return

    tick_list[current_row_arr_slice_indices[current_row_first_empty_cell_index:current_row_first_empty_cell_index + min(len(ticks), max_elements_to_add)]] = ticks[:min(len(ticks), max_elements_to_add)]

def irregular_tick_function(tick_list, env_choices):
    for choice in env_choices:
        tick_list[choice] = 9

class TickStrategy(Enum):
    ContinuosTick = partial(continuos_tick_function)
    IrregularTick = partial(irregular_tick_function)


def consecutive(dice):
    # check if dice have consecutive values
    if not dice:
        return False
    
    sorted_dice_values = [die.value for die in dice].copy()
    sorted_dice_values.sort()
    
    for i in range(len(sorted_dice_values) - 1):
        if sorted_dice_values[i+1] != sorted_dice_values[i] + 1:
            return False
    
    return True

def consecutive_brown_green_orange(dice):
    brown_die = [die for die in dice if die.color.value == Color.Brown.value]
    green_die = [die for die in dice if die.color.value == Color.Green.value]
    orange_die = [die for die in dice if die.color.value == Color.Orange.value]

    if len(brown_die) != 1 or len(green_die) != 1 or len(orange_die) != 1:
        return False
    
    ordered_dice = brown_die + green_die + orange_die
    sorted_dice_values = [die.value for die in ordered_dice]

    for i in range(len(sorted_dice_values) - 1):
        if sorted_dice_values[i+1] != sorted_dice_values[i] + 1:
            return False

    return True

def sum_greater_than_nine(dice):
    # check if dice sum is greater than 9
    return sum(map(lambda x: x.value, dice)) > 9

def same_value(dice):
    return not len(set(map(lambda x: x.value, dice))) != 1

def is_equal_to_one(dice):
    return False if not dice else all(map(lambda x: x.value == 1, dice))

def is_equal_to_two(dice):
    return False if not dice else all(map(lambda x: x.value == 2, dice))

def is_equal_to_three(dice):
    return False if not dice else all(map(lambda x: x.value == 3, dice))

def is_equal_to_four(dice):
    return False if not dice else all(map(lambda x: x.value == 4, dice))
    
def is_equal_to_five(dice):
    return False if not dice else all(map(lambda x: x.value == 5, dice))

def is_equal_to_six(dice):
    return False if not dice else all(map(lambda x: x.value == 6, dice))

def no_condition(dice):
    return True

class DiceCondition(Enum):
    Consecutive = partial(consecutive)
    ConsecutiveBrownGreenOrange = partial(consecutive_brown_green_orange)
    SumGreaterThanNine = partial(sum_greater_than_nine)
    SameValue = partial(same_value)
    IsEqualToOne = partial(is_equal_to_one)
    IsEqualToTwo = partial(is_equal_to_two)
    IsEqualToThree = partial(is_equal_to_three)
    IsEqualToFour = partial(is_equal_to_four)
    IsEqualToFive = partial(is_equal_to_five)
    IsEqualToSix = partial(is_equal_to_six)
    NoCondition = partial(no_condition)
    

def basic_scoring(tick_list, bonuses):
    if not bonuses:
        return 0

    score = 0
    for bonus in bonuses:
        if bonus.type == BonusType.Points and tick_list[bonus.location] != 0:
            score += bonus.value[0]

    return score

def set_collection_scoring(tick_list, bonuses):
    if not bonuses:
        return 0

    set_collection_points = bonuses[0].value
    items_counter = 0
    for bonus in bonuses:
        if bonus.type == BonusType.SetCollection and tick_list[bonus.location] != 0:
            items_counter += 1

    return set_collection_points[items_counter]

def highest_row_sum_scoring(tick_list, row_lengths):
    row_lengths_indices = [[i] * v for i, v in enumerate(row_lengths)]
    row_lengths_indices_flatten = [y for x in row_lengths_indices for y in x]

    rows_sums = []
    for index in range(len(row_lengths)):
        rows_sums.append(sum([tick_list[i] if tick_list[i] != 9 else 0 for i, v in enumerate(row_lengths_indices_flatten) if v == index]))
        
    return max(rows_sums)

class ScoringType(Enum):
    Basic = partial(basic_scoring)
    SetCollection = partial(set_collection_scoring)
    HighestRowSum = partial(highest_row_sum_scoring)


def vertical_rendering(tick_list, col_lengths):
    max_tick_length = max([len(x) for x in tick_list])
    tick_list = [x.ljust(max_tick_length) for x in tick_list]
    tick_list = np.array(tick_list)

    col_lengths_indices = [[i] * v for i, v in enumerate(col_lengths)]
    col_lengths_indices_flatten = np.array([y for x in col_lengths_indices for y in x])

    cols = [tick_list[np.where(col_lengths_indices_flatten == i)] for i in range(len(col_lengths))]
    longest_col = max([len(x) for x in cols])
    out = ""

    rows = []
    for i in range(longest_col):
        row = []
        for col in cols:
            if len(col) > i:
                row.append(col[i])
            else:
                row.append(" ")
        rows.append(row)

    for row in rows:
        row_str = reduce(lambda a, b: a + " " + b, row)
        out += row_str + "\n"

    if out[-1] == "\n":
        out = out[:-1]

    return out

def horizontal_rendering(tick_list, row_lengths, split_size=3):
    max_tick_length = max([len(x) for x in tick_list])
    tick_list = [x.ljust(max_tick_length) for x in tick_list]
    tick_list = np.array(tick_list)
    
    row_lengths_indices = [[i] * v for i, v in enumerate(row_lengths)]
    row_lengths_indices_flatten = np.array([y for x in row_lengths_indices for y in x])
    longest_row_characters = max([len(x) for x in tick_list])

    rows = [tick_list[np.where(row_lengths_indices_flatten == i)] for i in range(len(row_lengths))]
    string_rows = list(map(lambda row: reduce(lambda a, b: a + " " + b, row), rows))
    out = "\n".join(string_rows)

    if split_size != 0 and len(out.splitlines()) > split_size:
        splits = [x.split('\n') for x in re.findall(f'((?:[^\n]+\n?){{1,{split_size}}})', out)]
        longest_split = max([len(x) for x in splits])
        out = ""

        for i in range(longest_split):
            current_row = [split[i] for split in splits if len(split) > i]
            row = reduce(lambda a, b: a.ljust(longest_row_characters) + "    " + b.ljust(longest_row_characters) + "\n", current_row)
            out += row

        if out[-1] == "\n":
            out = out[:-1]

    return out

def horizontal_shared_rendering(tick_list, row_lengths, split_size=3):
    max_tick_length = max([len(x) for x in tick_list])
    tick_list = [x.ljust(max_tick_length) for x in tick_list]
    tick_list = np.array(tick_list)
    
    row_lengths_indices = [[i] * v for i, v in enumerate(row_lengths)]
    row_lengths_indices_flatten = np.array([y for x in row_lengths_indices for y in x])

    rows = [tick_list[np.where(row_lengths_indices_flatten == i)] for i in range(len(row_lengths))]

    if split_size != 0 and len(rows) > split_size:
        splits = [rows[i:i + split_size] for i in range(0, len(rows), split_size)]
        if len(splits) != 2:
            raise Exception("Wrong rendering for current section")
        out = ""

        for i in range(split_size):
            current_row = [split[i] for split in splits if len(split) > i]
            current_row_flatten = [y for x in current_row for y in x]
            if len(current_row_flatten) >= split_size:
                first_row = current_row_flatten[:3]
                second_row = current_row_flatten[::-1][:3]
                current_row_flatten = first_row[:2] + [max(first_row[-1], second_row[0])] + second_row[-2:]
            
            string_row = " ".join(current_row_flatten)
            out += string_row + "\n"

        if out[-1] == "\n":
            out = out[:-1]

    else:
        string_rows = list(map(lambda row: reduce(lambda a, b: a + " " + b, row), rows))
        out = "\n".join(string_rows)

    return out

def irregular_rendering(tick_list):
    max_tick_length = max([len(x) for x in tick_list])
    tick_list = [x.ljust(max_tick_length) for x in tick_list]

    return " ".join([x for x in tick_list])

class RenderType(Enum):
    Vertical = partial(vertical_rendering)
    Horizontal = partial(horizontal_rendering)
    HorizontalShared = partial(horizontal_shared_rendering)
    Irregular = partial(irregular_rendering)