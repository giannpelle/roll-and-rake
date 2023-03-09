from time import time
from xml.parsers.expat import model
from .classes import ContinuosSection, Die, Color, IrregularSection, Section, SectionMetadataContinuos, SectionMetadataIrregular
from .enums import BonusType, MoveType, RenderType, GameMove
import numpy as np
from operator import add
from functools import reduce
from collections import Counter
import re

import os
import sys
import random


parent_dir_path = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(parent_dir_path)

from utils_v2.utils import get_sections_metadata


class RollAndRakeState(object):
    def __init__(self):
        self.current_turn = 0
        self.green_track = 60
        self.max_time_value = 60
        self.current_dice_combination = []
        self.bonuses_available = []
        self.is_done = False
        
        self.available_dice, self.green_die_value = self.generate_new_dice()

        self.sections = []
        sections_metadata = get_sections_metadata()
        for section_metadata in sections_metadata:
            if self._is_instance(section_metadata, SectionMetadataContinuos):
                self.sections.append(ContinuosSection(section_metadata))
            elif self._is_instance(section_metadata, SectionMetadataIrregular):
                self.sections.append(IrregularSection(section_metadata))
    
    def reset(self):
        self.current_turn = 0
        self.green_track = 60
        self.current_dice_combination = []
        self.bonuses_available = []
        self.is_done = False

        self.available_dice, self.green_die_value = self.generate_new_dice()

        for section in self.sections:
            section.tick_list[:] = 0

    def generate_new_dice(self, orange_dice_num=3, brown_dice_num=2, green_dice_num=1):
        orange_dice = [Die(Color.Orange, random.randrange(6) + 1) for _ in range(orange_dice_num)]
        brown_dice = [Die(Color.Brown, random.randrange(6) + 1) for _ in range(brown_dice_num)]
        green_dice = [Die(Color.Green, random.randrange(6) + 1) for _ in range(green_dice_num)]
        
        green_die_value = green_dice[0].value
        return (orange_dice + brown_dice + green_dice, green_die_value)
    
    def _is_instance(self, obj, cls):
        obj_class = str(type(obj)).split(".")[-1]
        cls_class = str(cls).split(".")[-1]
        return obj_class == cls_class

    def _get_one_hot_encoding(self, *, of_value, with_max_bit_size):
        value = of_value
        max_bit_size = with_max_bit_size
        
        if value > max_bit_size:
            raise Exception("Value out of bounds")

        # modified version for die values between 1 and 6
        return [1 if (i + 1) == value else 0 for i in range(max_bit_size)]
        # original edition for die values between 0 and 6
        # return [1 if i == value else 0 for i in range(max_bit_size)]

    def _get_binary_encoding(self, *, of_tick_list):
        tick_list = of_tick_list

        return [1 if x != 0 else 0 for x in tick_list]

    def to_observation(self):
        
        one_hot_available_dice = [self._get_one_hot_encoding(of_value=die.value, with_max_bit_size=6) for die in self.available_dice]
        one_hot_available_dice_flatten = [y for x in one_hot_available_dice for y in x]

        binary_sections = [self._get_binary_encoding(of_tick_list=section.tick_list) for section in self.sections]
        binary_sections_flatten = [y for x in binary_sections for y in x]
        
        time_value = self.green_track / self.max_time_value
        metadata_arr = [time_value]

        legal_actions = [1 if self.is_action_legal(env_action_index=i) else 0 for i in range(MoveType.Pass.value)]
        return np.array(one_hot_available_dice_flatten + binary_sections_flatten + metadata_arr + legal_actions)

    def get_legal_env_actions_indices(self):
        return [i for i in range(MoveType.Pass.value) if self.is_action_legal(env_action_index=i)]

    def is_action_legal(self, *, env_action_index):
        if env_action_index < 0 or env_action_index >= MoveType.Pass.value:
            raise Exception("action index out of bounds")

        if env_action_index < MoveType.ChooseCategory.value:
            
            dice_indices = []
            match env_action_index:
                case 0:
                    dice_indices = [7]
                case 1:
                    dice_indices = [9, 10, 12, 17, 18, 20]
                case 2:
                    dice_indices = [32]
                case 3:
                    dice_indices = [40, 48, 56]
                case 4:
                    dice_indices = [3, 5, 6, 7]
                case 5:
                    dice_indices = [24]
                case 6:
                    dice_indices = [33, 34, 35, 36, 37, 38]
                case 7:
                    dice_indices = [41, 42, 44, 49, 50, 52]

            section = self.sections[env_action_index]

            for dice_indice in dice_indices:
                dice_combination = self._get_dice_combination(with_action_index=dice_indice)
                
                if self._is_instance(section, ContinuosSection):
                    if section.check_dice_requirements(dice=dice_combination) and not section.is_full():
                        return True
                elif self._is_instance(section, IrregularSection):
                    for choice in range(6): # number of ticks in Newman section (green dice)
                        if section.check_dice_requirements(dice=dice_combination, env_choices=[choice]) and not section.is_full():
                            return True    

            return False

        return True

    def _enumerate_all_dice_combinations(self):
        dice_combinations = []
        for env_action_index in range(2**6 - 1):
            action_index = env_action_index + 1
            dice_combinations.append(self._get_dice_combination(with_action_index=action_index))
        return dice_combinations

    def step(self, *, with_env_action_index):
        env_action_index = with_env_action_index
        # action_index = env_action_index + 1

        # print(f"step: action chosen ({GameMove(env_action_index)})")

        if not self.is_action_legal(env_action_index=env_action_index):
            # print(self)
            print(f"Error: The provided action is illegal {env_action_index}")
            # print(f"legal actions: {self.get_legal_env_actions_indices()}")
            return

        if env_action_index < MoveType.ChooseCategory.value:
            
            dice_indices = []
            match env_action_index:
                case 0:
                    dice_indices = [7]
                case 1:
                    dice_indices = [9, 10, 12, 17, 18, 20]
                case 2:
                    dice_indices = [32]
                case 3:
                    dice_indices = [40, 48, 56]
                case 4:
                    dice_indices = [3, 5, 6, 7]
                case 5:
                    dice_indices = [24]
                case 6:
                    dice_indices = [33, 34, 35, 36, 37, 38]
                case 7:
                    dice_indices = [41, 42, 44, 49, 50, 52]

            section = self.sections[env_action_index]

            for dice_indice in dice_indices:
                dice_combination = self._get_dice_combination(with_action_index=dice_indice)
                
                if self._is_instance(section, ContinuosSection):
                    if section.check_dice_requirements(dice=dice_combination) and not section.is_full():
                        section.make_use_of(dice=dice_combination)
                        break
                elif self._is_instance(section, IrregularSection):
                    for choice in range(6): # number of ticks in Newman section (green dice)
                        if section.check_dice_requirements(dice=dice_combination, env_choices=[choice]) and not section.is_full():
                            section.make_use_of(dice=dice_combination, with_env_choices=[self.green_die_value - 1])
                            break
        

        self._end_turn()


    def _end_turn(self):
        self.green_track = max(self.green_track - self.green_die_value, 0)

        self.current_turn += 1

        self.current_dice_combination = []
        
        self.available_dice, self.green_die_value = self.generate_new_dice()

        if self.green_track <= 0:
            self.is_done = True
    
    def _get_dice_chosen_indices_from(self, *, action_index):
        if action_index >= 2**len(self.available_dice):
            raise Exception("Index out of bounds")

        bits_array = np.array([int(x) for x in (bin(action_index)[2:])[::-1]])
        dice_chosen_indices = np.where(bits_array == 1)[0]
        return dice_chosen_indices

    def _get_dice_combination(self, *, with_action_index):
        action_index = with_action_index
        
        dice_chosen_indices = self._get_dice_chosen_indices_from(action_index=action_index)        
        for dice_index in dice_chosen_indices:
            if self.available_dice[dice_index].value == 0:
                return []
        
        return [die for index, die in enumerate(self.available_dice) if index in dice_chosen_indices]

    def _take_dice_combination(self, *, with_action_index):
        action_index = with_action_index

        dice_combination = self._get_dice_combination(with_action_index=action_index)
        self.available_dice = [die if die not in dice_combination else Die(die.color, 0) for die in self.available_dice]
        self.current_dice_combination = dice_combination
      
    def get_current_score(self):
        score = 0
        for section in self.sections:
            score += section.get_score()

        return score

    def _get_sections_tick_lists(self):
        sections = []
        for section in self.sections:
            sections.append(section.tick_list)
        return sections

    def _get_sections_sheets(self):
        sections_sheets = []
        for section in self.sections:
            max_emoji_num = Counter(map(lambda x: x.location, section.bonuses)).most_common(1)[0][1] + 1
            sheet = ["" for x in section.tick_list]
            for bonus in section.bonuses:
                bonus_str = bonus.type.value
                if bonus.type.value == BonusType.Points.value:
                    bonus_str = str(bonus.value) + bonus.type.value if len(bonus.value) > 1 else str(bonus.value[0]) + bonus.type.value
                sheet[bonus.location] = bonus_str if sheet[bonus.location] == "" else sheet[bonus.location] + "_" + bonus_str

            sheet = [tick if len(re.findall(r'[^\w\s,]', tick)) == max_emoji_num else "\U00003030" * (max_emoji_num - len(re.findall(r'[^\w\s,]', tick))) + tick for tick in sheet]
            sheet = [tick + " " for tick in sheet]
            sections_sheets.append(sheet)
        
        return sections_sheets

    def regularize(self, text):
        lines = text.split('\n')
        max_line_length = max([len(line) for line in lines])
        max_emoji_in_one_line = max([len(re.findall(r'[^\w\s,]', line)) for line in lines])
        
        regularized_text = ""
        for line in lines:
            emoji_in_line = len(re.findall(r'[^\w\s,]', line))
            if len(line) < max_line_length:
                emoji_to_add = max_emoji_in_one_line - emoji_in_line
                regularized_text += line + " " * (max_line_length - len(line) - emoji_to_add) + "\U0001F51A" * emoji_to_add + "\n"
            else:
                regularized_text += line + "\n"
        return regularized_text[:-1]

    def __str__ (self):
        title = f"Turn - {self.current_turn}"

        details_arr = []
        # details_arr.append(f"game_phase: {self.game_phase.name}")
        # details_arr.append(f"rerolls_available: {self.rerolls_available}")
        # details_arr.append(f"dice_combination_choices_available: {self.dice_combination_choices_available}")
        details_arr.append(f"remaining time: {self.green_track} (out of 60)")
        details_arr.append(f"current_score: {self.get_current_score()}")
        details = "\n".join(details_arr)

        available_dice = "available dice:" + "\n"
        available_dice += " ".join([str(die.value) + ":" + die.color.value for die in self.available_dice])

        sections_tick_list_sheets_str = self._get_sections_sheets()
        sections_tick_lists_str = [[str(int(x)) for x in section_tick_list] for section_tick_list in self._get_sections_tick_lists()]
        
        if len(sections_tick_list_sheets_str) != len(self.sections) or len(sections_tick_lists_str) != len(self.sections):
            raise Exception("Something went wrong while serializing the tick lists of sections")
        
        sections_str = []
        for i in range(len(self.sections)):
            section = self.sections[i]
            sheet_str = ""
            tick_list_str = ""

            if section.render_type.value == RenderType.Irregular.value:
                sheet_str = section.render_type.value(sections_tick_list_sheets_str[i])
                tick_list_str = section.render_type.value(sections_tick_lists_str[i])
            else:
                sheet_str = section.render_type.value(sections_tick_list_sheets_str[i], section.row_lengths)
                tick_list_str = section.render_type.value(sections_tick_lists_str[i], section.row_lengths)
        
            regularized_sheet_str = self.regularize(sheet_str)
            regularized_tick_list_str = self.regularize(tick_list_str)

            split_lines = zip(regularized_sheet_str.split("\n"), regularized_tick_list_str.split("\n"))
            sheet_tick_preview_str = "\n".join([x + "  |  " + y for x, y in split_lines])
            sections_str.append(section.name + "\n\n" + sheet_tick_preview_str)

        sections = "\n\n".join(sections_str)

        legal_actions = "legal actions: \n["
        legal_actions += ", ".join([GameMove(i).name for i in self.get_legal_env_actions_indices()]) + "]"

        out = "\n\n".join([title, details, available_dice, sections, legal_actions])
        out += "\n\n"

        return out