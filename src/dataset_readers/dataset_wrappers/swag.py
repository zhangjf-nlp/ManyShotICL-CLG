from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *
import logging
logger = logging.getLogger(__name__)
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    question = entry['startphrase']
    choices = [entry[f"ending{i}"] for i in range(4)]
    return f"{question}...\t" + " ".join([f"({chr(65+i)}) {choices[i]}" for i in range(len(choices))])


@field_getter.add("qa")
def get_qa(entry):
    return f"Choose an ending: {get_q(entry)}\tAnswer: {get_a(entry)}"


@field_getter.add("a")
def get_a(entry):
    return chr(ord("A") + entry['label'])


@field_getter.add("gen_a")
def get_gen_a(entry):
    return "{ice_prompt}Choose an ending: {question}\tAnswer: ".format(
        question=get_q(entry),
        ice_prompt="{ice_prompt}")


@field_getter.add("choices")
def get_choices(entry):
    return ["A", "B", "C", "D"]


class DatasetWrapper(ABC):
    name = "swag"
    ice_separator = "\n"
    question_field = "startphrase"
    answer_field = "label"
    hf_dataset = "swag"
    hf_dataset_name = "regular"
    field_getter = field_getter