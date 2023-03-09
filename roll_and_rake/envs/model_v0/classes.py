import numpy as np
import collections
from typing import NamedTuple, Callable
from .enums import BonusType, Color, DiceToTickConverter, RenderType, ScoringType, TickStrategy

class Die(NamedTuple):
    color: Color
    value: int

class DiceRequirement(NamedTuple):
    colors: list
    condition: Callable[[list], bool]

class Bonus(NamedTuple):
    type: BonusType
    value: list
    location: int

class SectionMetadataContinuos(NamedTuple):
    name: str
    tick_strategy: TickStrategy
    dice_to_tick_converter: DiceToTickConverter
    ticks_number: int
    row_lengths: list
    min_ticks_to_fullfill_row: int
    dice_requirements: list
    bonuses: list
    scoring_type: ScoringType
    render_type: RenderType

class SectionMetadataIrregular(NamedTuple):
    name: str
    tick_strategy: TickStrategy
    dice_to_tick_converter: DiceToTickConverter
    ticks_number: int
    dice_requirements: list
    tick_conditions: list
    bonuses: list
    scoring_type: ScoringType
    render_type: RenderType

class Section(object):

    def __init__(self, name, tick_strategy, dice_to_tick_converter, ticks_number, dice_requirements, bonuses, scoring_type, render_type):
        self.name = name
        self.tick_strategy = tick_strategy
        self.dice_to_tick_converter = dice_to_tick_converter
        self.tick_list = np.zeros(ticks_number)
        self.dice_requirements = dice_requirements
        self.bonuses = bonuses
        self.scoring_type = scoring_type
        self.render_type = render_type

class ContinuosSection(Section):

    def __init__(self, section_metadata_continuos):
        name = section_metadata_continuos.name
        tick_strategy = section_metadata_continuos.tick_strategy
        dice_to_tick_converter = section_metadata_continuos.dice_to_tick_converter
        ticks_number = section_metadata_continuos.ticks_number
        dice_requirements = section_metadata_continuos.dice_requirements
        bonuses = section_metadata_continuos.bonuses
        scoring_type = section_metadata_continuos.scoring_type
        render_type = section_metadata_continuos.render_type
        super().__init__(name, tick_strategy, dice_to_tick_converter, ticks_number, dice_requirements, bonuses, scoring_type, render_type)
        
        self.row_lengths = section_metadata_continuos.row_lengths
        self.min_ticks_to_fullfill_row = section_metadata_continuos.min_ticks_to_fullfill_row  

    def check_dice_requirements(self, *, dice):
        dice_colors = list(map(lambda x: x.color, dice))
        
        for dice_requirement in self.dice_requirements:
            dice_colors_required = list(map(lambda x: Color[x], dice_requirement.colors))
            compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

            dice_color_condition = compare(dice_colors, dice_colors_required)
            dice_value_condition = dice_requirement.condition(dice)
            
            if dice_color_condition and dice_value_condition:
                return True

        return False

    def is_full(self):
        row_lengths_indices = [[i] * v for i, v in enumerate(self.row_lengths)]
        row_lengths_indices_flatten = [y for x in row_lengths_indices for y in x]
        last_row_indices = [i for i, v in enumerate(row_lengths_indices_flatten) if v == len(self.row_lengths) - 1]
        last_row = [self.tick_list[i] for i in last_row_indices]
        return last_row[self.min_ticks_to_fullfill_row - 1] != 0

    def make_use_of(self, *, dice):
        # check legal dice combination
        if not self.check_dice_requirements(dice=dice):
            return

        if self.is_full():
            return

        ticks = self.dice_to_tick_converter.value(dice)

        starting_tick_grid = self.tick_list.copy()
        self.tick_strategy.value(self.tick_list, self.row_lengths, self.min_ticks_to_fullfill_row, ticks)

        ticked_list = self.tick_list - starting_tick_grid
        updated_locations_indices = np.where(ticked_list != 0)[0]
        bonuses_unlocked = list(filter(lambda x: x.location in updated_locations_indices, self.bonuses))
        # use bonuses_unlocked

    def get_score(self):
        if self.scoring_type.name == ScoringType.HighestRowSum.name:
            return self.scoring_type.value(self.tick_list, self.row_lengths)
        elif self.scoring_type.name == ScoringType.Basic.name:
            return self.scoring_type.value(self.tick_list, self.bonuses)
        elif self.scoring_type.name == ScoringType.SetCollection.name:
            return self.scoring_type.value(self.tick_list, self.bonuses)

class IrregularSection(Section):

    def __init__(self, section_metadata_irregular):
        name = section_metadata_irregular.name
        tick_strategy = section_metadata_irregular.tick_strategy
        dice_to_tick_converter = section_metadata_irregular.dice_to_tick_converter
        ticks_number = section_metadata_irregular.ticks_number
        dice_requirements = section_metadata_irregular.dice_requirements
        bonuses = section_metadata_irregular.bonuses
        scoring_type = section_metadata_irregular.scoring_type
        render_type = section_metadata_irregular.render_type
        super().__init__(name, tick_strategy, dice_to_tick_converter, ticks_number, dice_requirements, bonuses, scoring_type, render_type)

        self.tick_conditions = section_metadata_irregular.tick_conditions

    def check_dice_requirements(self, *, dice, env_choices):
        dice_colors = list(map(lambda x: x.color, dice))

        dice_combination_check = False
        
        for dice_requirement in self.dice_requirements:
            dice_colors_required = list(map(lambda x: Color[x], dice_requirement.colors))
            compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

            dice_color_condition = compare(dice_colors, dice_colors_required)
            dice_value_condition = dice_requirement.condition(dice)
            
            if dice_color_condition and dice_value_condition:
                dice_combination_check = True

        tick_conditions_check = True

        for choice in env_choices:
            if not self.tick_conditions[choice].value(dice) or self.tick_list[choice] != 0:
                tick_conditions_check = False

        return dice_combination_check and tick_conditions_check

    def is_full(self):
        return all(self.tick_list != 0.0)

    def make_use_of(self, *, dice, with_env_choices):
        choices = with_env_choices
        # check legal dice combination
        if not self.check_dice_requirements(dice=dice, env_choices=choices):
            return

        if self.is_full():
            return

        starting_tick_grid = self.tick_list.copy()
        self.tick_strategy.value(self.tick_list, choices)

        ticked_list = self.tick_list - starting_tick_grid
        updated_locations_indices = np.where(ticked_list != 0)[0]
        bonuses_unlocked = list(filter(lambda x: x.location in updated_locations_indices, self.bonuses))
        # use bonuses_unlocked

    def get_score(self):
        if self.scoring_type.name == ScoringType.Basic.name:
            return self.scoring_type.value(self.tick_list, self.bonuses)
        elif self.scoring_type.name == ScoringType.SetCollection.name:
            return self.scoring_type.value(self.tick_list, self.bonuses)










