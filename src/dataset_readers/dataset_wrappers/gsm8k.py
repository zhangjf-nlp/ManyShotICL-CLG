from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *
import logging

logger = logging.getLogger(__name__)
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    return entry["question"].strip()


@field_getter.add("qa")
def get_qa(entry):
    question = entry["question"].strip()
    answer = entry["answer"].strip()
    return f"<question>\n{question}\n</question>\n<answer>\n{answer}\n</answer>"


@field_getter.add("a")
def get_a(entry):
    return entry["answer"].strip() + "\n</answer>"


@field_getter.add("gen_a")
def get_gen_a(entry):
    return "{ice_prompt}<question>\n{text}\n</question>\n<answer>\n".format(
        ice_prompt="{ice_prompt}",
        text=get_q(entry)
    )


class DatasetWrapper(ABC):
    name = "gsm8k"
    ice_separator = "\n\n"
    question_field = "question"
    answer_field = "answer"
    hf_dataset = "openai/gsm8k"
    hf_dataset_name = "socratic"
    field_getter = field_getter
    stop_strings = ["</answer>"]
