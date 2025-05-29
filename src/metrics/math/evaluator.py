import numpy as np
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

from math_code_utils.grader import math_equal
from math_code_utils.latex import last_boxed_only_string, remove_boxed


class EvaluateTool(object):
    def __init__(self):
        pass

    def evaluate(self, preds, golds):
        golds = [gold["solution"] for gold in golds]
        bar = tqdm(zip(preds, golds), total=len(preds), desc="evaluating math solutions")
        results = []
        for pred, gold in bar:
            pred, gold = remove_boxed(last_boxed_only_string(pred)), remove_boxed(last_boxed_only_string(gold))
            result = math_equal(pred, gold)
            results.append(result)
            bar.set_postfix({
                "pred": pred,
                "gold": gold,
                "result": result
            })
        return {"acc": np.mean(results)}
