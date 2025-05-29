from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *
import logging

logger = logging.getLogger(__name__)
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    problem = entry["problem"].strip()
    solution = entry["solution"].strip()
    return f"<problem>\n{problem}\n</problem>\n<solution>\n"


@field_getter.add("qa")
def get_qa(entry):
    problem = entry["problem"].strip()
    solution = entry["solution"].strip()
    return f"<problem>\n{problem}\n</problem>\n<solution>\n{solution}\n</solution>"


@field_getter.add("a")
def get_a(entry):
    problem = entry["problem"].strip()
    solution = entry["solution"].strip()
    return f"{solution}\n</solution>"


@field_getter.add("gen_a")
def get_gen_a(entry):
    problem = entry["problem"].strip()
    solution = entry["solution"].strip()
    return "{ice_prompt}" + f"<problem>\n{problem}\n</problem>\n<solution>\n"


class DatasetWrapper(ABC):
    name = "math"
    ice_separator = "\n\n"
    question_field = "problem"
    answer_field = "solution"
    hf_dataset = "src/hf_datasets/math.py"
    hf_dataset_name = None
    field_getter = field_getter
    stop_strings = ["</solution>"]
