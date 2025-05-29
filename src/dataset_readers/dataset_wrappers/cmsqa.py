from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *
import logging

logger = logging.getLogger(__name__)
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    question = entry["question"]
    choices = entry["choices"]['text']
    return f"{question}\t" + " ".join([f"({chr(65+i)}) {choices[i]}" for i in range(len(choices))])


@field_getter.add("qa")
def get_qa(entry):
    return f"Question: {get_q(entry)}\tAnswer: {get_a(entry)}"


@field_getter.add("a")
def get_a(entry):
    return entry['answerKey']


@field_getter.add("gen_a")
def get_gen_a(entry):
    # hypothesis, premise = get_q(entry)
    return "{ice_prompt}Question: {text}\tAnswer: ".format(
        ice_prompt="{ice_prompt}",
        text=get_q(entry))


@field_getter.add("choices")
def get_choices(entry):
    return ["A", "B", "C", "D", "E"]


class DatasetWrapper(ABC):
    name = "cmsqa"
    ice_separator = "\n"
    question_field = "question"
    answer_field = "answerKey"
    hf_dataset = "commonsense_qa"
    hf_dataset_name = None
    field_getter = field_getter
