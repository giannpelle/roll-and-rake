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

class GamePhase(Enum):
    Reroll = auto()
    DiceChoice = auto()
    SectionChoice = auto()
    InnerSectionChoice = auto()

class MoveType(Enum):
    Reroll = 63
    TakeDiceCombination = 63 + (63)
    ChooseCategory = 9 + (63 + 63)
    ChooseInnerCategory = 6 + (9 + 63 + 63)
    Pass = auto()

class GameMove(Enum):
    R1 = 0
    R2 = auto()
    R12 = auto()
    R3 = auto()
    R13 = auto()
    R23 = auto()
    R123 = auto()
    R4 = auto()
    R14 = auto()
    R24 = auto()
    R124 = auto()
    R34 = auto()
    R134 = auto()
    R234 = auto()
    R1234 = auto()
    R5 = auto()
    R15 = auto()
    R25 = auto()
    R125 = auto()
    R35 = auto()
    R135 = auto()
    R235 = auto()
    R1235 = auto()
    R45 = auto()
    R145 = auto()
    R245 = auto()
    R1245 = auto()
    R345 = auto()
    R1345 = auto()
    R2345 = auto()
    R12345 = auto()
    R6 = auto()
    R16 = auto()
    R26 = auto()
    R126 = auto()
    R36 = auto()
    R136 = auto()
    R236 = auto()
    R1236 = auto()
    R46 = auto()
    R146 = auto()
    R246 = auto()
    R1246 = auto()
    R346 = auto()
    R1346 = auto()
    R2346 = auto()
    R12346 = auto()
    R56 = auto()
    R156 = auto()
    R256 = auto()
    R1256 = auto()
    R356 = auto()
    R1356 = auto()
    R2356 = auto()
    R12356 = auto()
    R456 = auto()
    R1456 = auto()
    R2456 = auto()
    R12456 = auto()
    R3456 = auto()
    R13456 = auto()
    R23456 = auto()
    R123456 = auto()
    T1 = auto()
    T2 = auto()
    T12 = auto()
    T3 = auto()
    T13 = auto()
    T23 = auto()
    T123 = auto()
    T4 = auto()
    T14 = auto()
    T24 = auto()
    T124 = auto()
    T34 = auto()
    T134 = auto()
    T234 = auto()
    T1234 = auto()
    T5 = auto()
    T15 = auto()
    T25 = auto()
    T125 = auto()
    T35 = auto()
    T135 = auto()
    T235 = auto()
    T1235 = auto()
    T45 = auto()
    T145 = auto()
    T245 = auto()
    T1245 = auto()
    T345 = auto()
    T1345 = auto()
    T2345 = auto()
    T12345 = auto()
    T6 = auto()
    T16 = auto()
    T26 = auto()
    T126 = auto()
    T36 = auto()
    T136 = auto()
    T236 = auto()
    T1236 = auto()
    T46 = auto()
    T146 = auto()
    T246 = auto()
    T1246 = auto()
    T346 = auto()
    T1346 = auto()
    T2346 = auto()
    T12346 = auto()
    T56 = auto()
    T156 = auto()
    T256 = auto()
    T1256 = auto()
    T356 = auto()
    T1356 = auto()
    T2356 = auto()
    T12356 = auto()
    T456 = auto()
    T1456 = auto()
    T2456 = auto()
    T12456 = auto()
    T3456 = auto()
    T13456 = auto()
    T23456 = auto()
    T123456 = auto()
    HarvickCategory = auto()
    ElliottCategory = auto()
    BuschCategory = auto()
    NewmanCategory = auto()
    JohnsonCategory = auto()
    SuarezCategory = auto()
    EarnhardtCategory = auto()
    PatrickCategory = auto()
    HamlinCategory = auto()
    NewmanFirst = auto()
    NewmanSecond = auto()
    NewmanThird = auto()
    NewmanFourth = auto()
    NewmanFifth = auto()
    NewmanSixth = auto()
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