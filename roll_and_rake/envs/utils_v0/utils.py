import json
import os
import sys

from roll_and_rake.envs.model_v0.enums import DiceToTickConverter, RenderType

parent_dir_path = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(parent_dir_path)

from model_v0.enums import BonusType, DiceCondition, ScoringType, TickStrategy
from model_v0.classes import DiceRequirement, Bonus, SectionMetadataContinuos, SectionMetadataIrregular

def get_section_metadata(with_name):
    name = with_name

    with open('roll_and_rake/envs/utils/sections_metadata.json') as json_file:
        data = json.load(json_file)
        sections = data["sections"]

        for section in sections:

            if section["name"] == name:
                tick_strategy = TickStrategy[section["tick_strategy"]]
                dice_to_tick_converter = DiceToTickConverter[section["dice_to_tick_converter"]]
                ticks_number = section["ticks_number"]

                dice_requirements = []
                for dice_requirement in section["dice_requirements"]:
                    dice_condition = DiceCondition[dice_requirement["condition"]]
                    dice_requirements.append(DiceRequirement(dice_requirement["colors"], dice_condition.value))

                bonuses = []
                for bonus in section["bonuses"]:
                    bonus_type = BonusType[bonus["type"]]
                    bonus_value = bonus["value"]
                    location = bonus["location"]
                    bonuses.append(Bonus(bonus_type, bonus_value, location))

                scoring_type = ScoringType[section["scoring_type"]]
                render_type = RenderType[section["render_type"]]

                if tick_strategy.value == TickStrategy.ContinuosTick.value:
                    row_lengths = section["row_lengths"]
                    min_ticks_to_fullfill_row = section["min_ticks_to_fullfill_row"]
                    
                    return SectionMetadataContinuos(name, tick_strategy, dice_to_tick_converter, ticks_number, row_lengths, min_ticks_to_fullfill_row, dice_requirements, bonuses, scoring_type, render_type)

                elif tick_strategy.value == TickStrategy.IrregularTick.value:
                    tick_conditions = list(map(lambda x: DiceCondition[x], section["tick_conditions"]))
                    
                    return SectionMetadataIrregular(name, tick_strategy, dice_to_tick_converter, ticks_number, dice_requirements, tick_conditions, bonuses, scoring_type, render_type)
                
def get_sections_metadata():
    sections_metadata = []

    with open('roll_and_rake/envs/utils_v0/sections_metadata.json') as json_file:
        data = json.load(json_file)
        sections = data["sections"]

        for section in sections:
        
            name = section["name"]
            tick_strategy = TickStrategy[section["tick_strategy"]]
            dice_to_tick_converter = DiceToTickConverter[section["dice_to_tick_converter"]]
            ticks_number = section["ticks_number"]

            dice_requirements = []
            for dice_requirement in section["dice_requirements"]:
                dice_condition = DiceCondition[dice_requirement["condition"]]
                dice_requirements.append(DiceRequirement(dice_requirement["colors"], dice_condition.value))

            bonuses = []
            for bonus in section["bonuses"]:
                bonus_type = BonusType[bonus["type"]]
                bonus_value = bonus["value"]
                location = bonus["location"]
                bonuses.append(Bonus(bonus_type, bonus_value, location))

            scoring_type = ScoringType[section["scoring_type"]]
            render_type = RenderType[section["render_type"]]

            if tick_strategy.value == TickStrategy.ContinuosTick.value:
                row_lengths = section["row_lengths"]
                min_ticks_to_fullfill_row = section["min_ticks_to_fullfill_row"]
                
                sections_metadata.append(SectionMetadataContinuos(name, tick_strategy, dice_to_tick_converter, ticks_number, row_lengths, min_ticks_to_fullfill_row, dice_requirements, bonuses, scoring_type, render_type))

            elif tick_strategy.value == TickStrategy.IrregularTick.value:
                tick_conditions = list(map(lambda x: DiceCondition[x], section["tick_conditions"]))
                
                sections_metadata.append(SectionMetadataIrregular(name, tick_strategy, dice_to_tick_converter, ticks_number, dice_requirements, tick_conditions, bonuses, scoring_type, render_type))
            
    return sections_metadata