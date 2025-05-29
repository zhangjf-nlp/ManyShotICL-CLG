from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *
import logging

logger = logging.getLogger(__name__)
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    question = entry["Question"].strip()
    choices = entry["Choices"].strip()
    explanation = entry["Explanation"].strip()
    answer = entry["Answer"].strip()
    return (f"<question>\n{question}\n</question>\n"
            f"<choices>\n{choices}\n</choices>\n"
            f"<explanation>\n")


@field_getter.add("qa")
def get_qa(entry):
    question = entry["Question"].strip()
    choices = entry["Choices"].strip()
    explanation = entry["Explanation"].strip()
    answer = entry["Answer"].strip()
    return (f"<question>\n{question}\n</question>\n"
            f"<choices>\n{choices}\n</choices>\n"
            f"<explanation>\n{explanation}\n</explanation>\n"
            f"<answer>\n{answer}\n</answer>")


@field_getter.add("a")
def get_a(entry):
    question = entry["Question"].strip()
    choices = entry["Choices"].strip()
    explanation = entry["Explanation"].strip()
    answer = entry["Answer"].strip()
    return (f"{explanation}\n</explanation>\n"
            f"<answer>\n{answer}\n</answer>")


@field_getter.add("gen_a")
def get_gen_a(entry):
    question = entry["Question"].strip()
    choices = entry["Choices"].strip()
    explanation = entry["Explanation"].strip()
    answer = entry["Answer"].strip()
    return "{ice_prompt}" + (f"<question>\n{question}\n</question>\n"
                             f"<choices>\n{choices}\n</choices>\n"
                             f"<explanation>\n")


class DatasetWrapper(ABC):
    name = "gpqa"
    ice_separator = "\n\n"
    question_field = "Question"
    answer_field = "Answer"
    hf_dataset = "src/hf_datasets/gpqa.py"
    hf_dataset_name = None
    field_getter = field_getter
    stop_strings = ["</answer>"]
